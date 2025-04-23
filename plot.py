import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             roc_curve, auc, precision_score, recall_score)
from torch.utils.data import DataLoader
import seaborn as sns
from train_cnn import PlantDiseaseCNN

# Create directory for graphs if it doesn't exist
os.makedirs('./graph', exist_ok=True)

def load_model(model_path, model_class, *args, **kwargs):
    """Load a saved PyTorch model from a checkpoint."""
    model = model_class(*args, **kwargs)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Extract just the model part
    model.eval()
    return model

def get_predictions(model, dataloader, device='cpu'):
    """Get predictions and true labels from model and dataloader."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """Plot a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./graph/confusion_matrix.png')
    plt.close()

def plot_precision_recall(y_true, y_scores, title='Precision-Recall Curve'):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = precision_score(y_true, y_scores > 0.5)
    avg_recall = recall_score(y_true, y_scores > 0.5)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title}\nAvg Precision: {avg_precision:.2f}, Avg Recall: {avg_recall:.2f}')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./graph/precision_recall_curve.png')
    plt.close()

def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig('./graph/roc_curve.png')
    plt.close()

def main():
    # Configuration - YOU NEED TO ADAPT THESE TO YOUR MODEL AND DATA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_CLASSES = 38  # Set this to match your actual number of classes
    
    # Load model with required arguments
    model = load_model('best_model.pth', PlantDiseaseCNN, num_classes=NUM_CLASSES).to(device)
    
    
    # 2. Prepare your test dataset and dataloader
    # test_dataset = YourDataset(...)  # Replace with your test dataset
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Get predictions
    # y_pred, y_true = get_predictions(model, test_loader, device)
    
    # For binary classification:
    # y_scores = ...  # Get your model's prediction scores (before thresholding)
    
    # For demonstration, let's create some dummy data
    # REMOVE THIS IN YOUR ACTUAL IMPLEMENTATION
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    y_scores = np.random.rand(100)
    classes = ['Class 0', 'Class 1']
    
    # 4. Generate and save plots
    plot_confusion_matrix(y_true, y_pred, classes)
    plot_precision_recall(y_true, y_scores)
    plot_roc_curve(y_true, y_scores)
    
    print("Visualizations saved to ./graph directory")

if __name__ == '__main__':
    main()