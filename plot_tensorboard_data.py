import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# ------------------------- Model Definition -------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.pool = nn.AdaptiveAvgPool2d((28, 28))

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


# ------------------------- Evaluation Function -------------------------
def evaluate_model(model, val_loader, class_names, device, writer):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# ------------------------- Logging Function -------------------------
def log_to_tensorboard(labels, preds, class_names, writer):
    report = classification_report(labels, preds, output_dict=True, target_names=class_names)

    for cls in class_names:
        writer.add_scalar(f'Precision/{cls}', report[cls]['precision'])
        writer.add_scalar(f'Recall/{cls}', report[cls]['recall'])
        writer.add_scalar(f'F1-Score/{cls}', report[cls]['f1-score'])

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    writer.add_figure("Confusion Matrix", fig)


# ------------------------- Main Function -------------------------
def main():
    # === You should replace these with your real values ===
    num_classes = 38  # Update to match your dataset
    best_model_path = './best_model.pth'
    log_dir = 'runs/final_eval'

    # Import these from your project:
    from your_dataset_loader import val_loader, class_names, device  # <-- You define this!

    # Setup model
    model = PlantDiseaseCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    writer = SummaryWriter(log_dir)

    labels, preds = evaluate_model(model, val_loader, class_names, device, writer)
    log_to_tensorboard(labels, preds, class_names, writer)

    writer.close()
    print(f"✅ Evaluation complete. View results with: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
