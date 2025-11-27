# Yoga Pose Classification System

A comprehensive machine learning project for real-time yoga pose classification and correctness evaluation using computer vision and deep learning techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Dataset Structure](#dataset-structure)
- [Results](#results)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a complete pipeline for yoga pose recognition that can:
- Extract and normalize skeletal keypoints from images using MediaPipe Pose
- Train multiple machine learning models (KNN, SVM, Random Forest, MLP, CNN, GNN)
- Perform real-time pose classification via webcam
- Evaluate pose correctness (right vs wrong posture)
- Identify specific step sequences in yoga poses

## ✨ Features

- **Multi-Model Architecture**: Implements and compares 6 different ML/DL approaches
- **Real-time Inference**: Webcam-based pose detection with live FPS counter
- **Pose Normalization**: Advanced keypoint normalization for scale/position invariance
- **Angle Feature Engineering**: Calculates 12 critical joint angles for enhanced accuracy
- **Graph Neural Networks**: Novel GNN approach treating pose skeleton as a graph structure
- **Comprehensive Evaluation**: Confusion matrices and precision metrics for all models
- **Parallel Processing**: Multi-threaded feature extraction for faster training

## 📁 Project Structure

```
miniproject/
│
├── keypointextractor.py          # Extract and cache MediaPipe keypoints from dataset
├── webcam.py                      # Real-time pose classification using webcam
│
├── training and evaluation/
│   ├── knn.py                    # K-Nearest Neighbors classifier
│   ├── svm.py                    # Support Vector Machine variants
│   ├── rf.py                     # Random Forest classifier
│   ├── mlp.py                    # Multi-Layer Perceptron (Neural Network)
│   ├── cnn.py                    # 1D Convolutional Neural Network
│   └── modelgnn.py               # Graph Neural Network (GCN)
│
└── confusion matrix/             # Stored confusion matrix visualizations
    ├── knn.png
    ├── svm.png
    ├── random_forest.png
    ├── mlp.png
    ├── cnn.png
    └── gnn.png
```

## 🛠 Technologies Used

### Core Libraries
- **MediaPipe**: Pose landmark detection (33 body keypoints)
- **OpenCV**: Image processing and webcam capture
- **NumPy**: Numerical computations and array operations

### Machine Learning
- **scikit-learn**: KNN, SVM, Random Forest, MLP implementations
- **PyTorch**: CNN and GNN deep learning models
- **PyTorch Geometric**: Graph neural network operations

### Visualization & Analysis
- **Matplotlib & Seaborn**: Confusion matrices and performance plots
- **Pandas**: Data organization and CSV export
- **Joblib**: Model serialization and caching

## 🚀 Installation

### Prerequisites
```bash
python >= 3.8
CUDA-compatible GPU (optional, for faster training)
```

### Install Dependencies
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn
pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib seaborn joblib
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd miniproject

# Extract keypoints from your dataset
python keypointextractor.py

# Train a model (example: GNN)
python "training and evaluation/modelgnn.py"

# Run real-time inference
python webcam.py
```

## 📖 Usage

### 1. Dataset Preparation
Organize your yoga pose images in the following structure:
```
dataset/
└── images/
    └── [PoseName]/
        ├── right/
        │   ├── step 1/
        │   ├── step 2/
        │   └── step 3/
        └── wrong/
            ├── step 1/
            ├── step 2/
            └── step 3/
```

### 2. Extract Keypoints
```bash
python keypointextractor.py
```
This script:
- Scans the dataset directory recursively
- Extracts 33 pose landmarks per image (x, y, z, visibility)
- Normalizes keypoints relative to hip center and body scale
- Caches results for faster subsequent training
- Outputs: `processed_keypoints/X_keypoints.npy`, `y_labels.npy`, `pose_classes.pkl`

### 3. Train Models
Each training script in `training and evaluation/` follows this pattern:
```bash
python "training and evaluation/[model_name].py"
```

**Available Models:**
- `knn.py`: Tests multiple k-values, distance metrics, and weighting schemes
- `svm.py`: Compares linear, polynomial, RBF, and sigmoid kernels
- `rf.py`: Tunes n_estimators and max_depth parameters
- `mlp.py`: Experiments with different hidden layer configurations
- `cnn.py`: 1D convolutions over pose feature sequences
- `modelgnn.py`: Graph convolutional networks on skeletal structure

### 4. Real-time Inference
```bash
python webcam.py
```
**Controls:**
- Press `q` to quit
- Real-time display shows:
  - Pose name (e.g., "Warrior")
  - Step number (e.g., "Step 1")
  - Correctness (Right/Wrong)
  - FPS counter

## 🤖 Models Implemented

### 1. K-Nearest Neighbors (KNN)
- **Hyperparameters Tuned**: n_neighbors (3, 5, 7, 9), weights (uniform, distance), metric (euclidean, manhattan, minkowski)
- **Feature Set**: Normalized keypoints + 12 joint angles
- **Best Config**: Varies by dataset, typically k=5 with distance weighting

### 2. Support Vector Machine (SVM)
- **Kernels Tested**: Linear, Polynomial, RBF, Sigmoid
- **Variants**: SVC and LinearSVC with different loss functions
- **Feature Set**: Standardized keypoints + angles
- **Strengths**: Effective for high-dimensional feature spaces

### 3. Random Forest (RF)
- **Hyperparameters**: n_estimators (50, 100, 200, 300), max_depth (None, 10, 20, 30)
- **Feature Set**: Raw normalized features
- **Strengths**: Robust to overfitting, handles non-linear relationships

### 4. Multi-Layer Perceptron (MLP)
- **Architectures**: (50,), (100,), (50,50) hidden layers
- **Activations**: ReLU, Tanh, Logistic
- **Optimizers**: Adam, SGD
- **Strengths**: Captures complex non-linear patterns

### 5. 1D Convolutional Neural Network (CNN)
- **Architecture**: Conv1D → BatchNorm → MaxPool → Dense layers
- **Hyperparameters**: Filters (32, 64, 128), kernel sizes (3, 5), dropout rates
- **Input**: Pose features as 1D sequences
- **Strengths**: Learns local patterns in feature space

### 6. Graph Neural Network (GNN) ⭐
**Most Advanced Model**
- **Architecture**: 2-layer GCN + angle feature fusion + dense classifier
- **Node Features**: (x, y, z, visibility) for each of 33 landmarks
- **Edge Structure**: Anatomically-based skeleton connectivity (15 edges)
- **Angle Features**: 12 critical joint angles processed separately
- **Training**: 80 epochs with early stopping, learning rate scheduling
- **Strengths**: 
  - Respects skeletal structure through graph representation
  - Combines graph convolutions with engineered angle features
  - Achieves highest accuracy among all models
  - Naturally handles pose topology

**GNN Technical Details:**
```python
# Graph structure: 33 nodes (pose landmarks), 15 bidirectional edges
# Node features: [x, y, z, visibility] (4D)
# Global features: 12 joint angles
# Output: Classification over pose classes
```

## 📊 Dataset Structure

The system expects images organized by:
1. **Pose Name**: Type of yoga pose (e.g., Warrior, Tree, Downward Dog)
2. **Correctness**: `right` or `wrong` execution
3. **Step**: Sequential stages (step 1, step 2, step 3)

**Label Format**: `{PoseName}_{right/wrong}_{step#}`
Example: `Warrior_right_step1`, `Tree_wrong_step2`

## 📈 Results

Each training script generates:
- **CSV Reports**: Precision/accuracy comparisons for different configurations
- **Bar Plots**: Visual performance comparisons
- **Confusion Matrices**: Per-class prediction analysis (saved as PNG in `confusion matrix/`)
- **Trained Models**: Serialized model files (`.pt`, `.pkl`)

**Typical Output Files:**
```
knn_precision_comparison.csv
mlp_precision_barplot.png
confusion_matrix.png
pose_gcn_best.pt
pose_classes.pkl
```

## 🔧 How It Works

### Feature Extraction Pipeline

1. **Landmark Detection**
   - MediaPipe Pose detects 33 3D body landmarks
   - Each landmark: (x, y, z, visibility)

2. **Normalization**
   - Center: Mid-point between left and right hips
   - Scale: Max of (vertical span, shoulder-hip distance)
   - Result: Translation and scale-invariant features

3. **Angle Calculation**
   - 12 critical angles computed:
     - Elbow angles (left/right)
     - Knee angles (left/right)
     - Shoulder-hip angles (left/right)
     - Hip-knee angles (left/right)
     - Additional torso/limb angles
   - Provides rotation-invariant features

4. **Feature Vector**
   - Flattened keypoints: 33 × 4 = 132 features
   - Joint angles: 12 features
   - **Total: 144-dimensional feature vector per pose**

### Graph Construction (GNN-specific)

```
Nodes: 33 pose landmarks
Edges: Anatomical connections
  - Arms: shoulder → elbow → wrist
  - Legs: hip → knee → ankle → foot
  - Torso: shoulders ↔ hips
  - etc.

Node Features: [x, y, z, visibility]
Global Features: 12 computed angles
```

### Real-time Inference

1. Capture webcam frame
2. Extract pose landmarks using MediaPipe
3. Normalize keypoints and calculate angles
4. Feed to trained model (GNN)
5. Predict: pose name, step, correctness
6. Display results with skeleton overlay

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional pose datasets
- New model architectures
- Performance optimizations
- Documentation enhancements
- Bug fixes and testing

## 📝 Notes

- **GPU Recommended**: GNN and CNN training benefit significantly from CUDA
- **Dataset Size**: Minimum 100 images per class recommended for good performance
- **Threading**: Adjust `NUM_WORKERS` based on your CPU cores
- **Memory**: Large datasets may require caching to disk (already implemented)
- **Mediapipe Version**: Ensure compatible version for consistent landmark detection

## 📄 License

This project is provided for educational and research purposes.

## 👏 Acknowledgments

- MediaPipe team for excellent pose detection library
- PyTorch Geometric for graph neural network tools
- scikit-learn for traditional ML implementations

---
