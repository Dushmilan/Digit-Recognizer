# 🔢 Digit Recognizer (CNN)

An optimized Deep Learning system designed to classify handwritten digits (0-9). This project implements a Convolutional Neural Network (CNN) using **Keras 3** and **TensorFlow**, trained on a curated subset of the MNIST dataset. It features modern data augmentation techniques and robust evaluation metrics to ensure high generalization accuracy.

---

## 📈 Performance & Results

The final model successfully overcame early training challenges (model collapse) to achieve state-of-the-art classification performance:

- **Test Accuracy:** **97%** (300 test samples)
- **Validation Accuracy:** **97.08%** (final epoch)
- **Training Accuracy:** **96.46%** (final epoch)
- **Confusion Matrix:** Near-perfect diagonal alignment with minimal confusion between visually similar digits (e.g., 3 vs 9, 4 vs 9)
- **Perfect Recall (100%):** Digits 0, 1, 2, 6, and 8
- **Convergence:** Training and validation curves demonstrate healthy learning with no signs of overfitting

---

## 🚀 Key Features

- **Modern CNN Architecture:** Multi-layered convolutional design with Batch Normalization and Dropout for stability and regularization
- **Keras 3 Augmentation Pipeline:** GPU-accelerated data augmentation using `RandomRotation`, `RandomZoom`, and `RandomTranslation` layers
- **Intelligent Training:** 
  - `EarlyStopping` to halt training at peak performance (patience: 8 epochs)
  - `ReduceLROnPlateau` for adaptive learning rate decay
- **Comprehensive Reporting:** 
  - Full `classification_report` with precision, recall, and F1-scores
  - Confusion matrix heatmap (`confusion_matrix.png`)
  - Training history visualization (`training_curves.png`)

---

## 📁 Repository Structure

```
Digit-Recognizer/
├── Data/
│   └── mnist1.5k.csv          # 1,500 labeled MNIST samples
├── Model.py                    # Complete training pipeline
├── digit_recognizer_Model.h5   # Trained Keras model
├── training_curves.png         # Loss/accuracy visualization
├── confusion_matrix.png        # Detailed error analysis
└── README.md                   # Project documentation
```

---

## 🛠️ Model Architecture

The network is optimized for 28×28 grayscale digit images:

```
Input (28×28×1)
    ↓
[Data Augmentation Block]
├── RandomRotation(±3°)
├── RandomZoom(±5%)
└── RandomTranslation(±5%)
    ↓
[Conv Block 1]
├── Conv2D(32, 3×3, ReLU)
├── BatchNormalization
├── Conv2D(32, 3×3, ReLU)
├── BatchNormalization
├── MaxPooling2D(2×2)
└── Dropout(20%)
    ↓
[Conv Block 2]
├── Conv2D(64, 3×3, ReLU)
├── BatchNormalization
├── Conv2D(64, 3×3, ReLU)
├── BatchNormalization
└── MaxPooling2D(2×2)
    ↓
[Classifier]
├── Flatten
├── Dense(128, ReLU)
├── BatchNormalization
├── Dropout(40%)
└── Dense(10, Softmax)
```

**Total Parameters:** ~200K  
**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy

---

## 💻 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed, then install dependencies:

```bash
pip install tensorflow pandas scikit-learn seaborn matplotlib
```

### Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dushmilan/Digit-Recognizer.git
   cd Digit-Recognizer
   ```

2. **Verify dataset location:**  
   Ensure `mnist1.5k.csv` is in the `Data/` directory.

3. **Train the model:**
   ```bash
   python Model.py
   ```

4. **Review outputs:**
   - `digit_recognizer_Model.h5` – Trained model
   - `training_curves.png` – Accuracy/loss plots
   - `confusion_matrix.png` – Error analysis heatmap

---

## 📉 Troubleshooting & Development Insights

### Initial Challenge: Model Collapse
During early development, the model experienced **catastrophic collapse**, predicting only a single class (digit 4) across all inputs. This was resolved through:

1. **Proper Normalization:** Ensuring pixel values are scaled to `[0, 1]` via division by 255.0
2. **Learning Rate Adjustment:** Implementing `ReduceLROnPlateau` to dynamically lower the learning rate when validation loss plateaus
3. **Increased Patience:** Setting `EarlyStopping` patience to 8 epochs to allow the model to escape local minima
4. **Keras 3 Migration:** Replacing deprecated `ImageDataGenerator` with modern preprocessing layers for better GPU utilization

### Common Issues

**Issue:** `ImportError: cannot import name 'ImageDataGenerator'`  
**Solution:** This project uses Keras 3, which deprecated `ImageDataGenerator`. The code now uses `layers.RandomRotation`, `layers.RandomZoom`, etc.

**Issue:** `KeyError: 'label'`  
**Solution:** The CSV header uses capitalized `Label`. The code automatically detects both `Label` and `label`.

---

## 🎯 Future Enhancements

- [ ] Deploy as a web application using TensorFlow.js
- [ ] Implement real-time digit recognition via webcam
- [ ] Experiment with deeper architectures (ResNet, EfficientNet)
- [ ] Add support for the full MNIST dataset (60K training samples)

---

## 👤 Author

**Dushmilan**  
� [GitHub Profile](https://github.com/Dushmilan)

**Project Category:** Computer Vision | Deep Learning | Image Classification

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset:** MNIST (Modified National Institute of Standards and Technology)
- **Framework:** TensorFlow/Keras
- **Inspiration:** Classic computer vision benchmarks
