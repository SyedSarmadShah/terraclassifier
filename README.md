# 🌍 TerraClassifier - Land Use and Land Cover Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**TerraClassifier** is an advanced deep learning system for **automated Land Use and Land Cover (LULC) classification** using satellite imagery. It uses a custom CNN architecture with explainable AI (Grad-CAM) to classify satellite images into 5 different land cover categories with 96.15% accuracy.

### Key Features ✨

- ✅ **Advanced CNN Architecture** - Custom designed with batch normalization and dropout
- ✅ **High Accuracy** - 96.15% test accuracy on 5 classes
- ✅ **Multi-Class Classification** - 5 land cover types
- ✅ **Explainable AI** - Grad-CAM visualizations for interpretability
- ✅ **Data Augmentation** - Rotation, flipping, zoom, and shift transformations
- ✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ **Easy Prediction** - Simple interface to classify new satellite images
- ✅ **Production Ready** - Trained model included, ready for deployment

---

## 🎯 Supported Land Cover Classes

The model can classify satellite images into 5 categories:

| Class | Description | Symbol |
|-------|-------------|--------|
| Forest | Dense vegetation areas with trees | 🌲 |
| Residential | Urban residential areas | 🏘️ |
| Highway | Road networks and transportation infrastructure | 🛣️ |
| Industrial | Industrial/manufacturing zones | 🏭 |
| River | Water bodies and flowing rivers | 🌊 |

---

## 📊 Dataset

- **Source**: EuroSAT RGB Satellite Dataset
- **Resolution**: 64×64 RGB images
- **Total Samples**: 13,500 images
- **Train/Val/Test Split**: 9,450 / 1,350 / 2,700 (70% / 10% / 20%)
- **Classes**: 5 land cover categories
- **Data Augmentation**: Applied during training (rotation, flips, zoom, shifts)

---

## 🏗️ Project Structure

```
terraclassifier/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── EuroSAT_RGB/
│   │       ├── Forest/
│   │       ├── Highway/
│   │       ├── Industrial/
│   │       └── ...
│   ├── processed/
│   │   └── preprocessed_data.npz     # Preprocessed data
│   └── splits/
│
├── src/
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   ├── model_architecture.py         # CNN model design
│   ├── train.py                      # Training pipeline
│   ├── evaluate.py                   # Evaluation metrics
│   └── explainability.py             # Grad-CAM visualization
│
├── models/
│   └── saved_models/
│       └── best_model.h5             # Trained model (40MB)
│
├── results/
│   ├── graphs/
│   │   ├── training_history.png      # Training curves
│   │   └── per_class_metrics.png     # Performance metrics
│   ├── confusion_matrix/
│   │   ├── confusion_matrix.png
│   │   └── confusion_matrix_normalized.png
│   ├── predictions/
│   │   ├── sample_predictions.png
│   │   └── misclassified_samples.png
│   ├── augmented_samples.png
│   └── sample_images.png
│
├── main.py                           # Complete pipeline orchestration
├── predict_image.py                  # Image prediction script
├── analyze_mistakes.py               # Mistake analysis tool
├── run_predict.sh                    # Wrapper script
│
├── requirements.txt                  # Python dependencies
├── REQUIREMENTS_COMPLETION.md        # Requirements checklist
├── DEMO_GUIDE.md                     # How to demo the model
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SyedSarmadShah/terraclassifier.git
cd terraclassifier
```

2. **Create virtual environment**
```bash
python -m venv lulc_env
source lulc_env/bin/activate  # On Windows: lulc_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Complete Pipeline
```bash
python main.py
```
This will:
- Load and preprocess satellite images
- Build and compile the CNN model
- Train the model on augmented data
- Evaluate on test set
- Generate visualizations and metrics

#### Option 2: Predict on New Images
```bash
./run_predict.sh
```
Then choose:
- **Option 1**: Classify a single image
- **Option 2**: Classify multiple images from a folder
- **Option 3**: Auto-test on sample images
- **Option 4**: Exit

Example:
```bash
./run_predict.sh
# Enter choice: 1
# Enter image path: path/to/satellite/image.jpg
```

#### Option 3: Analyze Model Mistakes
```bash
python analyze_mistakes.py
```
Shows:
- Common classification mistakes
- Why the model makes errors
- Improvement suggestions

---

## 🧠 Model Architecture

### CNN Design

```
Input: (64, 64, 3) RGB Image
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Conv Block 4: Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten
    ↓
Dense(512) → ReLU → BatchNorm → Dropout(0.5) → L2 Regularization
    ↓
Dense(256) → ReLU → BatchNorm → Dropout(0.5) → L2 Regularization
    ↓
Dense(5) → Softmax (Output)
```

### Key Components

- **Batch Normalization**: Stabilizes training and reduces overfitting
- **Progressive Dropout**: 0.25 → 0.4 → 0.5 (increases with depth)
- **L2 Regularization**: Weight decay (0.001) in dense layers
- **Max Pooling**: Reduces spatial dimensions
- **ReLU Activation**: Non-linearity
- **Softmax Output**: 5-class probability distribution

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 50 |
| Early Stopping | Yes (patience=15) |
| Learning Rate Reduction | Yes (factor=0.5) |
| Data Augmentation | Yes |

---

## 📈 Performance Metrics

### Actual Results

- **Accuracy**: 96.15%
- **Precision**: 96.51%
- **Recall**: 96.15%
- **F1-Score**: 96.14%
- **Test Samples**: 2,700 images
- **Correctly Classified**: 2,597 / 2,700
- **Error Rate**: 3.85%

### Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall classification correctness
- **Precision**: True positives vs false positives (per class)
- **Recall**: True positives vs false negatives (per class)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Results Visualization

- ✅ `results/graphs/training_history.png` - Accuracy and loss curves
- ✅ `results/graphs/per_class_metrics.png` - Per-class performance
- ✅ `results/confusion_matrix/` - Confusion matrices
- ✅ `results/predictions/` - Sample and misclassified predictions

---

## 🔍 Explainable AI - Grad-CAM

The model includes Grad-CAM (Gradient-weighted Class Activation Mapping) for interpretability.

**What it shows:**
- Which regions of the satellite image influenced the prediction
- Attention maps highlighting important features
- Visual explanation of model decisions

---

## 📚 Data Preprocessing

### Steps Performed

1. **Loading**: Load RGB satellite images from EuroSAT dataset
2. **Resizing**: Resize to 64×64 pixels
3. **Normalization**: Pixel values scaled to [0, 1]
4. **Data Splitting**: Train (70%) / Validation (10%) / Test (20%)
5. **Augmentation**: 
   - Rotation: 20 degrees
   - Horizontal/Vertical flip
   - Width/Height shift: 20%
   - Zoom: 20%

---

## ⚠️ Model Limitations

### Current Challenges

1. **Image Resolution**
   - 64×64 pixels is small
   - Fine details may be lost
   - **Solution**: Use higher resolution (256×256+)

2. **Limited Dataset Size**
   - EuroSAT has ~10,000 images
   - Limited geographic diversity
   - **Solution**: Combine with Sentinel-2, UC Merced datasets

3. **Class Imbalance**
   - Some classes have more training examples
   - Model may be biased
   - **Solution**: Use weighted loss or data balancing

4. **Similar Classes**
   - Highway vs Roads vs Urban areas
   - Forest vs Herbaceous vegetation
   - **Solution**: Use attention mechanisms

5. **Temporal Changes**
   - Dataset snapshot at one time
   - Seasonal variations not captured
   - **Solution**: Use temporal series

---

## 🚀 Future Improvements

### Short-term
- [ ] Increase image resolution to 128×128
- [ ] Implement transfer learning (ResNet, EfficientNet)
- [ ] Add class weighting for imbalanced data
- [ ] Ensemble methods combining multiple models

### Medium-term
- [ ] Multi-spectral data (beyond RGB)
- [ ] Temporal analysis (seasonal changes)
- [ ] Attention mechanisms for better interpretability
- [ ] Real-world deployment optimization

### Long-term
- [ ] Multi-temporal change detection
- [ ] Automated mapping pipeline
- [ ] Mobile app for field classification
- [ ] Integration with satellite APIs

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy>=1.24.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
tensorflow>=2.15.0
matplotlib>=3.7.0
pillow>=9.5.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Code Examples

### Example 1: Classify a Single Image

```python
from predict_image import ImageClassifier

# Initialize classifier
classifier = ImageClassifier()

# Classify image
result = classifier.predict('satellite_image.jpg')

# View result
classifier.print_result(result)
```

### Example 2: Train Custom Model

```python
from src.model_architecture import LULCClassifier
from src.train import ModelTrainer

# Build model
classifier = LULCClassifier(input_shape=(64, 64, 3), num_classes=5)
model = classifier.build_custom_cnn()
model = classifier.compile_model(model, learning_rate=0.001)

# Train
trainer = ModelTrainer(model)
history = trainer.train_with_augmentation(epochs=50, batch_size=32)

# Evaluate and plot
trainer.plot_training_history()
```

### Example 3: Evaluate Model

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model_path='models/saved_models/best_model.h5')
evaluator.evaluate_model()
evaluator.plot_confusion_matrix()
```

---

## 🎓 Educational Use

This project is perfect for:

- 🎓 **University Projects**: Complete ML pipeline with documentation
- 📊 **Research**: Satellite imagery classification baseline
- 🧑‍💼 **Portfolio**: Demonstrates full ML workflow
- 👨‍🏫 **Teaching**: Shows data preprocessing, model design, evaluation

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and create pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Syed Sarmad Shah**
- GitHub: [@SyedSarmadShah](https://github.com/SyedSarmadShah)
- Project: [TerraClassifier](https://github.com/SyedSarmadShah/terraclassifier)

---

## 📞 Support

For issues, questions, or suggestions, please create an issue on GitHub.

---

**Status**: ✅ Production Ready | **Model Trained**: ✅ Yes | **Ready for Demo**: ✅ Yes | **Accuracy**: 96.15%

Last Updated: January 18, 2026
- scikit-learn: Machine learning utilities
- Pandas & NumPy: Data manipulation
- Pillow: Image handling
- LIME: Explainable AI
- Jupyter: Interactive notebooks

## Usage

1. Place your raw data in `data/raw/`
2. Run preprocessing: `python src/data_preprocessing.py`
3. Train the model: `python src/train.py`
4. Evaluate results: `python src/evaluate.py`
5. Generate explanations: `python src/explainability.py`

## License

MIT License
