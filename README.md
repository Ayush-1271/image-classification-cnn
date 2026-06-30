# Image Classification CNN — CIFAR-10

**Ayush Ranjan | 23MIP10135 | VIT Bhopal**

## Results
| Metric | Score |
|---|---|
| Test Accuracy | **91.86%** |
| Top-3 Accuracy | **98.84%** |
| Parameters | 5,682,762 |

## Dataset
- **Source:** CIFAR-10 (via Hugging Face `uoft-cs/cifar10`)
- **Train:** 50,000 images · **Test:** 10,000 images
- **Classes:** 10 · **Image size:** 32 × 32 px

### Classes
airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

## Architecture — Deep CNN (4 Conv Blocks, VGG-style)
```
Input (32×32×3)
├── Block 1: Conv2D(64)×2  → BatchNorm → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(128)×2 → BatchNorm → MaxPool → Dropout(0.30)
├── Block 3: Conv2D(256)×3 → BatchNorm → MaxPool → Dropout(0.35)
├── Block 4: Conv2D(512)×2 → BatchNorm            → Dropout(0.35)
├── GlobalAveragePooling2D
├── Dense(512) → BatchNorm → Dropout(0.50)
├── Dense(256) → Dropout(0.40)
└── Dense(10, softmax)
```
All Conv layers: ReLU · same-padding · L2(1e-4) regularization

## Training Config
| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, EarlyStopping patience=15) |
| Batch Size | 128 |
| Pipeline | tf.data with GPU-side augmentation |
| Augmentation | random flip, pad+crop, brightness, contrast |
| Framework | TensorFlow 2.20 / Keras |
| Hardware | Google Colab T4 GPU |

## How to Run
1. Open `2_Image_Classification_CIFAR10_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime → **T4 GPU**
3. Run all cells top to bottom
