# 🌱 Plant Disease Detection System

![Plant Disease Detection](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-red)
![Flask](https://img.shields.io/badge/Flask-2.3.2-lightgrey)
![License](https://img.shields.io/badge/License-GPL--2.0-green)

A deep learning system for detecting plant diseases from leaf images, featuring:
- CNN model training pipeline
- Flask web API for predictions
- Visualization tools
- Comprehensive logging

## 📂 Project Structure

```
plant-disease-detection/
├── training/                  # Model training scripts
│   ├── train_cnn.py           # Main training script
│   ├── final-model.py         # Final trained model (60 epochs)
│   ├── zion_train_cnn.py      # Early prototype training script
│   └── best_model.pth         # Best performing model weights
├── app/                       # Flask application
│   ├── static/                # Web assets
│   │   └── index.html         # Web interface
│   └── app.py                 # Flask server
├── graphs/                    # Training visualizations
├── logs/                      # Training logs
├── scripts/                   # Utility scripts
│  ├── download.py             # Dataset downloader
   ├── predict.py             # Pridiction 
│  └── plot.py                # Visualization tools
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) NVIDIA GPU with CUDA 11.8

### Installation

#### For CPU-only systems:
```bash
pip install -r requirements.txt
```

#### For CUDA 11.8 systems:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Quickest)
If you just want to run predictions with the included pre-trained model (`./training/best_model.pth`), you can skip straight to starting the Flask server:

```bash
python predict.py
```

### Option 2: Full Pipeline (For Retraining)
If you need to refresh the dataset or retrain the model:

1. **Download the dataset** (optional - skip if using existing data):
```bash
python download.py
```

2. **Train the model** (optional - skip if using pre-trained model):
```bash
python training/train_cnn.py
```

3. **Start the Flask server** (required):
```bash
python predict.py
```

4. **Generate visualizations** (optional):
```bash
python plot.py
```

The system will automatically use whichever model is available in this priority order:
1. `training/best_model.pth` (if exists)
2. Freshly trained model (if you run train_cnn.py)
3. Falls back to included pre-trained model
   
## 🧠 Model Architecture

The system uses a custom CNN architecture with the following structure:
```
PlantDiseaseCNN(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Dropout(p=0.3, inplace=False)
    (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Dropout(p=0.4, inplace=False)
    (10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU()
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Dropout(p=0.5, inplace=False)
  )
  (pool): AdaptiveAvgPool2d(output_size=(28, 28))
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=100352, out_features=512, bias=True)
    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=512, out_features=38, bias=True)
  )
)
```
Through multiple iterations, I carefully optimized the model architecture and training process to achieve optimal performance while preventing overfitting. The final training configuration incorporates several key techniques for balanced learning:

**Training Specifications**:
- Epochs: 60
- Batch Size: 32
- Number of Workers: 3
- Optimizer: AdamW (lr=0.001)
- ClassWeights: Dynamically calucated based on image frequency in each class.
- Loss Function: CrossEntropyLoss
- Early Stopping: Patience of 5 epochs

## 🌐 Flask API Endpoints

### `GET /`
- Serves the web interface (static/index.html)

### `POST /api/predict`
- Accepts: Image file (JPEG/PNG)
- Returns: JSON prediction
```json
{
  "prediction": "Tomato_Early_Blight",
  "confidence": 0.927,
  "status": "success"
}
```

### `GET /health/check`
- Returns: System health information
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device": "cuda:0",
  "model_loaded": true,
  "timestamp": "2023-11-15T12:34:56Z"
}
```

## 📊 Visualization Tools

The `plot.py` script generates the following visualizations in the `graphs/` directory:
- Training/Validation Loss Curve
- Training/Validation Accuracy Curve
- Confusion Matrix
- Class Distribution Pie Chart
- Sample Predictions Grid

## 📝 Logging

Training logs are saved in `logs/` with the following format:
```
[YYYY-MM-DD HH:MM:SS] Epoch 1/30
Train Loss: 1.234 | Train Acc: 45.67%
Val Loss: 1.123 | Val Acc: 50.12%
---
```

## 🏆 Performance Metrics

| Metric          | Training | Validation |
|-----------------|----------|------------|
| Accuracy        | 98.2%    | 96.5%      |
| Precision       | 98.1%    | 96.3%      |
| Recall          | 98.0%    | 96.2%      |
| F1 Score        | 98.0%    | 96.2%      |

## 📜 License

This project is licensed under the GPL 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from [PlantVillage Dataset]([https://www.kaggle.com/emmarex/plantdisease](https://www.kaggle.com/api/v1/datasets/download/mohitsingh1804/plantvillage)

