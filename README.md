# Image Classification using CNN — CIFAR-10

> **Assignment Submission — VIT Bhopal**

| Field | Details |
|---|---|
| **Name** | Ayush Ranjan |
| **Roll No** | 23MIP10135 |
| **Branch** | MIP |
| **Institute** | VIT Bhopal |

---

## Project Overview

This project implements an **Image Classification system** using a custom deep Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset — a benchmark dataset for object recognition.

The model classifies 32×32 colour images into 10 object categories with high accuracy using a VGG-inspired deep CNN architecture.

---

## Dataset

**CIFAR-10**
- Source: [CIFAR-10 on Kaggle](https://www.kaggle.com/c/cifar-10) | [Official](https://www.cs.toronto.edu/~kriz/cifar.html)
- 60,000 images (50,000 train + 10,000 test)
- 10 classes, 6,000 images per class
- Image size: 32×32 RGB

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## CNN Architecture

```
Input (32×32×3)
    ↓
Block 1: Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Block 3: Conv2D(256) × 3 → BatchNorm → MaxPool → Dropout(0.35)
    ↓
Block 4: Conv2D(512) × 2 → BatchNorm → Dropout(0.35)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.4)
    ↓
Dense(10, softmax)
```

**Regularisation:** L2 weight decay (1e-4), BatchNormalization, Dropout.

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Batch Size | 128 |
| Max Epochs | 100 (EarlyStopping) |
| Image Size | 32×32 |
| Augmentation | Rotation, shift, horizontal_flip, zoom |

---

## How to Run

1. Open `2_Image_Classification_CIFAR10_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — CIFAR-10 is downloaded automatically (no Kaggle API needed)

---

## Files

```
├── 2_Image_Classification_CIFAR10_CNN.ipynb    # Main Colab notebook
└── README.md
```

---

## Results

The model outputs:
- Test accuracy & Top-3 accuracy
- Training curves (accuracy + loss)
- Confusion matrix (10×10)
- Per-class classification report
- Sample predictions with confidence scores
