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
from torch.utils.tensorboard import SummaryWriter

device = None
train_loader = None
val_loader = None
test_loader = None
model = None
class_names = None
num_classes = None
criterion = None
optimizer = None
scaler = None
scheduler = None
train_loader = None
writer = SummaryWriter('runs/experiment_1')

# Configure logging
def setup_logging():
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
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)

# Device configuration
def setup_device():
    logger.info("Initializing CUDA...")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        # Warm up CUDA
        torch.zeros(1).to(device)
        logger.info(f"CUDA initialized in {time.time()-start_time:.2f}s")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available, using CPU")
    
    return device

def init_setup():
    device = setup_device()

    # Data transformations
    logger.info("Setting up data transformations...")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load dataset
    logger.info("Loading datasets from ./train and ./val...")
    try:
        train_data = datasets.ImageFolder('./train', transform=train_transform)
        val_data = datasets.ImageFolder('./val', transform=val_test_transform)
        logger.info(f"Training set loaded with {len(train_data)} images")
        logger.info(f"Validation set loaded with {len(val_data)} images")
        
        # For test set, we'll use a portion of validation or create separately
        # Here we'll split validation into val+test (80/20)
        val_size = int(0.8 * len(val_data))
        test_size = len(val_data) - val_size
        val_data, test_data = random_split(val_data, [val_size, test_size])
        logger.info(f"Final splits - Train: {len(train_data)}, Val: {val_size}, Test: {test_size}")
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        exit()

    # Apply transforms
    val_data.dataset.transform = val_test_transform
    test_data.dataset.transform = val_test_transform

    # DataLoaders
    logger.info("Creating data loaders...")
    batch_size = 32
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increase the number of workers for better parallel data loading
        pin_memory=True,
        prefetch_factor=2  # Preload 2 batches per worker in the background
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Increase number of workers if your CPU can handle it
        pin_memory=True,  # Speed up GPU transfer
        prefetch_factor=2  # Preload 2 batches per worker in the background
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Increase number of workers if your CPU can handle it
        pin_memory=True,  # Speed up GPU transfer
        prefetch_factor=2  # Preload 2 batches per worker in the background
    )

    class_names = train_data.classes
    num_classes = len(class_names)
    logger.info(f"\nFound {num_classes} classes:\n{class_names}\n")

    torch.backends.cudnn.benchmark = True  # Optimizes CUDA ops for performance
    torch.set_float32_matmul_precision('medium')  # Faster matrix math

    logger.info("Initializing model...")
    model = PlantDiseaseCNN(num_classes).to(device)
    logger.info(f"Model architecture:\n{model}")

    # Handle class imbalance
    logger.info("Calculating class weights with optimized balancing...")
    def get_class_weights():
        # Get all labels (before split)
        train_labels  = [label for _, label in train_data.imgs]
        
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes)
        
        # Improved weight calculation
        median_count = torch.median(class_counts.float())
        weights = (median_count / class_counts)  # Stronger scaling than sqrt
        weights = torch.clamp(weights, min=0.2, max=8.0)  # Wider range

        # Additional boost for rare classes (<100 samples)
        low_sample_mask = class_counts < 100
        weights[low_sample_mask] = torch.clamp(weights[low_sample_mask] * 1.5, max=6.0)

        # Special boost for very rare classes
        rare_class_mask = class_counts < 100
        weights[rare_class_mask] = torch.clamp(weights[rare_class_mask] * 1.8, max=10.0)
        
        # Log combined distribution and weights
        logger.info("\n📊 Class Distribution and Optimized Weights:")
        max_name_length = max(len(name) for name in class_names)
        for i, (count, weight) in enumerate(zip(class_counts, weights)):
            logger.info(
                f"{class_names[i]:<{max_name_length}} | "
                f"Count: {count:>5} | "
                f"Weight: {weight:.2f}"
            )
        
        return weights.to(device)

    class_weights = get_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    scaler = GradScaler('cuda')

    logger.info("\n⚙️ Training Configuration:")
    logger.info(f"- Optimizer: AdamW")
    logger.info(f"- Initial LR: {optimizer.param_groups[0]['lr']}")
    logger.info(f"- Batch Size: {batch_size}")
    logger.info(f"- Epochs: 20")
    logger.info(f"- Class weights: {class_weights.cpu().numpy()}")

# Model definition
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
          
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 224 -> 112
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 112 -> 56
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 56 -> 28
            nn.Dropout(0.4),
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

def train(num_epochs=20):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"\n🏁 Epoch {epoch+1}/{num_epochs} started")
        
        model.train()
        optimizer.zero_grad()  # Reset once per epoch
        running_loss = 0.0
        correct = 0
        total = 0
        accum_steps = 2  # Simulates batch_size=96

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(loop):
            # Async data transfer (critical for Windows)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(stream)  # Sync

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Acc: {100.*correct/total:.2f}%")
        
        # Validation
        val_acc = validate()
        history['train_loss'].append(running_loss/len(train_loader))
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, 'best_model.pth')
            logger.info(f"💾 New best model saved with val acc: {best_val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logger.info(f"📊 Train Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")
        logger.info(f"📈 Val Acc: {val_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
    
        # Optional: Sync at epoch end
        torch.cuda.synchronize()
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    plot_training_curves(history)
    return history

def validate():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    return acc

def plot_training_curves(history):
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
    logger.info("📈 Training curves saved to training_curves.png")

def test():
    logger.info("\n🧪 Starting final evaluation...")
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    logger.info("📝 Classification report saved to classification_report.txt")
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    logger.info("📊 Confusion matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting training pipeline")
        init_setup()
        start_time = time.time()
        
        history = train(num_epochs=20)
        test()
        
        total_time = (time.time() - start_time) / 3600
        logger.info(f"\n🏁 Training completed in {total_time:.2f} hours")
        logger.info(f"📄 Full logs saved to: {log_file}")
        
    except Exception as e:
        logger.exception("🔥 Critical error in training pipeline")
    finally:
        logging.shutdown()