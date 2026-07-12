# Transfer Learning: Cat vs Dog Classification using PyTorch

A deep learning project implementing transfer learning with ResNet-50 to classify images of cats and dogs. This project demonstrates fine-tuning a pre-trained neural network for binary image classification tasks.

## 📋 Project Overview

This project uses **transfer learning** with ResNet-50 to build a binary classifier that distinguishes between cat and dog images. By leveraging pre-trained ImageNet weights and fine-tuning on a custom dataset, the model achieves robust classification performance with minimal training time.

## 🏗️ Architecture & Approach

### Model: ResNet-50
- **Base Model**: ResNet-50 pre-trained on ImageNet
- **Modification Strategy**: 
  - Frozen initial layers (Layer 1-3) to preserve learned features
  - Unfrozen Layer 4 and fully connected layers for fine-tuning
  - Custom classification head with adaptive pooling

### Custom Classification Head
```python
nn.Linear(2048, 512)  → Batch Normalization → ReLU → Dropout(0.4) → Linear(512, 2)
```

### Data Augmentation Techniques
The project implements three transformation strategies applied to training data:
1. **Normal Transform**: Resize, center crop (224×224), normalization
2. **Vertical Transform**: + vertical flips, random rotation (±10°)
3. **Horizontal Transform**: + horizontal flips, random rotation (±10°)

Training data is tripled using augmentation for improved generalization.

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Weight Decay | 1e-3 |
| Batch Size | 32 |
| Epochs | 10 |
| Loss Function | Cross Entropy Loss |
| Scheduler | Cosine Annealing |
| Gradient Clipping | Max norm = 1.0 |

## 📊 Dataset Structure

```
Cat_And_Dog/
├── train/          # Training images with data augmentation
├── test/           # Test split for model evaluation
└── test1/          # Additional test set (cat/dog subfolders)
```

### Data Processing
- **Train-Test Split**: 80-20 split with stratification
- **Image Format**: JPG
- **Image Resolution**: 224×224 (after resizing and center crop)
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
- **Augmented Training Dataset**: 3× original size due to augmentation

## 📈 Model Performance

The model is evaluated on multiple test sets:
- Accuracy metric on normalized test data
- Support for evaluation on additional test subsets
- Device support: GPU (CUDA) or CPU

## 🛠️ Key Features

- ✅ Transfer learning with pre-trained ResNet-50
- ✅ Multi-strategy data augmentation
- ✅ Mixed precision training support (AdamW optimizer)
- ✅ Gradient clipping for stable training
- ✅ Learning rate scheduling with cosine annealing
- ✅ CUDA/CPU device flexibility
- ✅ Batch normalization in custom head
- ✅ Dropout regularization (0.4)
- ✅ Model checkpointing and weight saving

## 📁 Project Files

- `transfer_learning_on_dogs_vs_cat.ipynb` - Main training notebook
- `models weights/resnet_cats_dogs.pth` - Trained model weights
- `README.md` - Project documentation

## 🚀 Installation & Usage

### Requirements
```bash
pip install torch torchvision
pip install pandas numpy matplotlib scikit-learn pillow tqdm
```

### Training
Run the Jupyter notebook to:
1. Load and preprocess the dataset
2. Configure ResNet-50 with custom classification head
3. Train for 10 epochs with data augmentation
4. Evaluate on test sets
5. Save model weights

### Model Loading & Inference
```python
import torch
from torchvision import models
import torch.nn as nn

# Load model
resnet = models.resnet50()
resnet.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

# Load trained weights
resnet.load_state_dict(torch.load("resnet_cats_dogs.pth"))
resnet.eval()

# Predict on new images
```

## 💾 Model Weights

Pre-trained model weights are available in `models weights/resnet_cats_dogs.pth`. The model uses:
- ResNet-50 backbone (ImageNet pre-trained)
- Custom 2-class output head for cat/dog classification
- 10 epochs of fine-tuning with the configuration above

## 📝 Technical Details

- **Framework**: PyTorch
- **Data Loading**: Custom Dataset class with PIL image loading
- **Device Support**: Automatic GPU (CUDA) or CPU fallback
- **Data Parallel**: Multi-worker data loading with pin_memory for efficiency
- **Evaluation**: No-gradient evaluation mode with torch.no_grad()

## 📚 References

This project demonstrates best practices in transfer learning:
- Selective layer unfreezing for efficient fine-tuning
- Strategic data augmentation for small datasets
- Learning rate scheduling for convergence
- Gradient clipping for training stability