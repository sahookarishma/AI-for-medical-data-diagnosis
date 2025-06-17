# AI-for-medical-data-diagnosis

This repository implements a complete deep learning pipeline using **Tensorflow** to classify images of cats and dogs. The model architecture is based on **DenseNet**, with support for class imbalance via a **weighted loss function**. The code is modularized into different scripts for clarity and reusability.

---

## 🧠 Project Overview

This binary classification project covers the entire workflow:

- 📊 Data Exploration and Preprocessing
- 🧮 Label Counting and Weighted Loss Handling
- 🧱 Model Definition using DenseNet
- 🏋️ Model Training and Evaluation
- 📈 Accuracy Metrics and Confusion Matrix
- 📦 Model Saving/Loading and Inference

---

#Install dependencies 
pip install torch torchvision matplotlib opencv-python tqdm

# data exploration and Pre-processing
python data_exploration_and_image_preprocessing.py

#Class imabalance Handling 
python counting_labels_and_weighted_loss_function.py

#Define the model 
from densenet_architecture import DenseNetClassifier
model = DenseNetClassifier(num_classes=2)


