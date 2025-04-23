import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from torch.amp import autocast, GradScaler
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import copy

class PlantDiseaseTrainer:
    def __init__(self):
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.criterion = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.logger = self.setup_logging()
        self.batch_size = 32  # Default batch size
        self.epochs = 30
        
    def setup_logging(self):
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(log_file, maxBytes=1e6, backupCount=3, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.log_file = log_file
        return logging.getLogger(__name__)

    def setup_device(self):
        self.logger.info("Initializing CUDA...")
        start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            # Warm up CUDA
            torch.zeros(1).to(self.device)
            self.logger.info(f"CUDA initialized in {time.time()-start_time:.2f}s")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.warning("No GPU available, using CPU")

    def setup_data(self):
        self.logger.info("Setting up data transformations...")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Added vertical flip
            transforms.RandomRotation(45),  # Increased rotation range
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translations
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # Optional blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),  # Random erasing regularization
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.logger.info("Loading datasets from ./PlantVillage/train and ./PlantVillage/val...")
        try:
            train_data = datasets.ImageFolder('./training/PlantVillage/train', transform=train_transform)
            val_data = datasets.ImageFolder('./training/PlantVillage/val', transform=val_test_transform)
            self.logger.info(f"Training set loaded with {len(train_data)} images")
            self.logger.info(f"Validation set loaded with {len(val_data)} images")
            
            val_size = int(0.8 * len(val_data))
            test_size = len(val_data) - val_size
            val_data, test_data = random_split(val_data, [val_size, test_size])
            self.logger.info(f"Final splits - Train: {len(train_data)}, Val: {val_size}, Test: {test_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            exit()

        test_data.dataset.transform = val_test_transform

        batch_size = 32
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            prefetch_factor=2
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            prefetch_factor=2
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            prefetch_factor=2
        )

        self.class_names = train_data.classes
        self.num_classes = len(self.class_names)
        self.logger.info(f"\nFound {self.num_classes} classes:\n{self.class_names}\n")
    
    def setup_model(self):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')

        self.logger.info("Initializing model...")
        self.model = PlantDiseaseCNN(self.num_classes).to(self.device)
        self.logger.info(f"Model architecture:\n{self.model}")

    def setup_optimization(self):
        self.logger.info("Calculating class weights with optimized balancing...")
        def get_class_weights():
            train_labels = [label for _, label in datasets.ImageFolder('./training/PlantVillage/train').imgs]
            class_counts = torch.bincount(torch.tensor(train_labels), minlength=self.num_classes)
            
            median_count = torch.median(class_counts.float())
            weights = (median_count / class_counts)
            weights = torch.clamp(weights, min=0.2, max=8.0)

            low_sample_mask = class_counts < 100
            weights[low_sample_mask] = torch.clamp(weights[low_sample_mask] * 1.5, max=6.0)

            rare_class_mask = class_counts < 50
            weights[rare_class_mask] = torch.clamp(weights[rare_class_mask] * 1.8, max=10.0)
            
            self.logger.info("\n📊 Class Distribution and Optimized Weights:")
            max_name_length = max(len(name) for name in self.class_names)
            for i, (count, weight) in enumerate(zip(class_counts, weights)):
                self.logger.info(
                    f"{self.class_names[i]:<{max_name_length}} | "
                    f"Count: {count:>5} | "
                    f"Weight: {weight:.2f}"
                )
            
            return weights.to(self.device)

        class_weights = get_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3, factor=0.5)
        self.scaler = GradScaler()

        self.logger.info("\n⚙️ Training Configuration:")
        self.logger.info(f"- Optimizer: AdamW")
        self.logger.info(f"- Initial LR: {self.optimizer.param_groups[0]['lr']}")
        self.logger.info(f"- Batch Size: 32")
        self.logger.info(f"- Epochs: 20")
        self.logger.info(f"- Class weights: {class_weights.cpu().numpy()}")
        
    def train(self, num_epochs=20):
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n🏁 Epoch {epoch+1}/{num_epochs} started")
            
            self.model.train()
            self.optimizer.zero_grad()
            running_loss = 0.0
            correct = 0
            total = 0
            accum_steps = 2

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for batch_idx, (images, labels) in enumerate(loop):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(stream)

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / accum_steps
                
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f} - Acc: {100.*correct/total:.2f}%")
            
            val_acc = self.validate()
            history['train_loss'].append(running_loss/len(self.train_loader))
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names,
                }, 'best_model.pth')
                self.logger.info(f"💾 New best model saved with val acc: {best_val_acc:.2f}%")
            
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            self.logger.info(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s")
            self.logger.info(f"📊 Train Loss: {running_loss/len(self.train_loader):.4f} | Acc: {100.*correct/total:.2f}%")
            self.logger.info(f"📈 Val Acc: {val_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
        
            torch.cuda.synchronize()
        
        torch.save(self.model.state_dict(), 'final_model.pth')
        self.plot_training_curves(history)
        return history

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return acc

    def test(self):
        self.logger.info("\n🧪 Starting final evaluation...")
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        with open('classification_report.txt', 'w') as f:
            f.write(classification_report(all_labels, all_preds, target_names=self.class_names))
        self.logger.info("📝 Classification report saved to classification_report.txt")
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        self.logger.info("📊 Confusion matrix saved to confusion_matrix.png")

    def plot_training_curves(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.savefig('training_curves.png')
        plt.close()
        self.logger.info("📈 Training curves saved to training_curves.png")

    def run(self):
        try:
            self.logger.info("🚀 Starting training pipeline")
            self.setup_device()
            self.setup_data()
            self.setup_model()
            self.setup_optimization()
            
            start_time = time.time()
            history = self.train(num_epochs=self.epochs)
            self.test()
            
            total_time = (time.time() - start_time) / 3600
            self.logger.info(f"\n🏁 Training completed in {total_time:.2f} hours")
            self.logger.info(f"📄 Full logs saved to: {self.log_file}")
            
        except Exception as e:
            self.logger.exception("🔥 Critical error in training pipeline")
        finally:
            logging.shutdown()

# Model definitionclass PlantDiseaseCNN(nn.Module):
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
          
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 224 -> 112
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 112 -> 56
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 56 -> 28
            nn.Dropout(0.5),
        )

        # This is crucial: adaptive pooling makes the output size fixed (no guessing flatten dims)
        self.pool = nn.AdaptiveAvgPool2d((28, 28))  # Output: 128 x 4 x 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
  
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    trainer = PlantDiseaseTrainer()
    trainer.run()