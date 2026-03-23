# Hand Gesture Recognition Using ConvNeXt and Attention

## Overview

Hand Gesture Recognition Using ConvNeXt and Attention is a deep learning-based computer vision project designed to classify infrared hand gesture images into 10 gesture categories. The system combines transfer learning with ConvNeXtTiny, attention mechanisms, and a structured TensorFlow input pipeline to improve recognition accuracy, focus on gesture-relevant regions, and maintain reproducible training and evaluation.

The project is built with a practical machine learning workflow suitable for research, academic demonstration, and portfolio presentation. It emphasizes subject-wise evaluation, modular training, and robust performance analysis.

---

## Key Highlights

* End-to-end hand gesture classification pipeline
* Transfer learning with ConvNeXtTiny pretrained on ImageNet
* Channel attention for feature recalibration
* Spatial attention for region-level focus
* Subject-wise train, validation, and test split
* Efficient `tf.data` pipeline for scalable preprocessing
* Data augmentation to improve generalization
* Two-stage training with frozen backbone and fine-tuning
* Evaluation using accuracy, top-3 accuracy, classification report, and confusion matrix
* Best model checkpointing with early stopping and learning rate scheduling

---

## System Architecture

### Training Workflow

The project follows a structured machine learning lifecycle:

1. Dataset loading from Google Drive
2. ZIP extraction and directory traversal
3. Metadata creation with subject and label information
4. Label cleaning and numerical encoding
5. Subject-wise dataset splitting
6. TensorFlow data pipeline construction
7. Data augmentation and preprocessing
8. ConvNeXtTiny backbone initialization
9. Attention-enhanced classification head construction
10. Transfer learning training
11. Backbone fine-tuning
12. Final evaluation and visualization

Generated model artifacts:

* `best_hand_model.keras` — Best checkpoint from the initial transfer learning phase
* `best_finetuned_hand_model.keras` — Best checkpoint from the fine-tuning phase

This structure ensures a clear separation between data handling, modeling, training, and evaluation.

---

## Dataset

This project uses the Leap Motion Hand Gesture Recognition Database available on Kaggle.

### Dataset Description

* 10 gesture classes
* 10 subjects
* Infrared images captured using a Leap Motion sensor
* Folder structure organized by subject and gesture

### Gesture Classes

* palm
* l
* fist
* fist_moved
* thumb
* index
* ok
* palm_moved
* c
* down

### Citation

T. Mantecón, C. R. del Blanco, F. Jaureguizar, and N. García, “Hand Gesture Recognition using Infrared Imagery Provided by Leap Motion Controller,” in *Int. Conf. on Advanced Concepts for Intelligent Vision Systems (ACIVS)*, 2016.

### License

Please review the original Kaggle dataset page and associated license terms before redistribution, reuse, or commercial application.

---

## Machine Learning Methodology

### Data Preparation

The dataset is extracted from a compressed archive and organized into a dataframe containing:

* subject identifier
* raw gesture label
* image file path

The pipeline is designed to robustly handle common image formats and build a clean metadata table for training.

### Label Standardization

Raw dataset folder names are mapped to clean class names before encoding. This creates a consistent label space for classification.

### Subject-Wise Split Strategy

A subject-wise split is used instead of a random split to better measure generalization to unseen individuals:

* Training set: subjects 00 to 07
* Validation set: subject 08
* Test set: subject 09

This reduces identity leakage and makes the evaluation more realistic.

### Input Pipeline

The TensorFlow pipeline performs:

* image decoding
* resizing to 224 × 224
* batching
* prefetching
* shuffling for training data

This approach improves training efficiency and keeps preprocessing consistent.

### Data Augmentation

The model uses lightweight augmentation during training:

* horizontal flip
* random rotation
* random zoom

These transformations help reduce overfitting and improve robustness.

---

## Model Architecture

The final model combines a pretrained convolutional backbone with attention-based refinement.

### Architecture Components

* Input layer for 224 × 224 infrared images
* Data augmentation layer
* ConvNeXtTiny backbone with ImageNet weights
* Channel attention block
* Spatial attention block
* Global average pooling
* Batch normalization
* Fully connected dense layers
* Dropout regularization
* Softmax output layer for 10-class prediction

### Why This Design Works

* ConvNeXtTiny provides strong feature extraction with a modern convolutional design
* Channel attention helps the model emphasize the most informative feature maps
* Spatial attention improves focus on relevant gesture regions
* Dense layers and dropout support robust classification while reducing overfitting

---

## Training Strategy

The project uses a two-phase training approach.

### Phase 1: Transfer Learning

* ConvNeXt backbone is frozen
* Adam optimizer with learning rate `1e-4`
* Early stopping is enabled
* Learning rate reduction on plateau is applied
* Best checkpoint is saved automatically

### Phase 2: Fine-Tuning

* Upper layers of ConvNeXt are unfrozen
* Lower layers remain frozen for stability
* Learning rate is reduced to `1e-5`
* Fine-tuned checkpoint is saved separately

This approach allows the model to first learn task-specific classification behavior and then adapt pretrained visual representations to infrared gesture imagery.

---

## Training Configuration

| Parameter                 | Value                           |
| ------------------------- | ------------------------------- |
| Input Image Size          | 224 × 224                       |
| Batch Size                | 32                              |
| Initial Optimizer         | Adam                            |
| Initial Learning Rate     | 1e-4                            |
| Fine-Tuning Learning Rate | 1e-5                            |
| Initial Epochs            | 10                              |
| Fine-Tuning Epochs        | 10                              |
| Loss Function             | Sparse Categorical Crossentropy |
| Metrics                   | Accuracy, Top-3 Accuracy        |

---

## Evaluation

The model is evaluated using both numerical metrics and visual analysis.

### Evaluation Outputs

* training and validation accuracy curves
* training and validation loss curves
* fine-tuning accuracy curves
* fine-tuning loss curves
* classification report with precision, recall, and F1-score
* confusion matrix for class-level performance analysis
* combined validation accuracy summary across both training stages

### Evaluation Goals

The evaluation phase is intended to assess:

* convergence behavior
* overfitting tendencies
* class-wise performance consistency
* generalization to unseen subjects

---

## Installation

### Clone the Repository

```bash
https://github.com/sushantkothari/Hand-Gesture-Recognition-ConvNeXt-Attention.git
cd Hand-Gesture-Recognition-ConvNeXt-Attention
```

### Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Mount Google Drive and place the dataset ZIP file in the expected path.
3. Run the notebook cells in order:

   * import libraries
   * extract the dataset
   * build the metadata dataframe
   * create subject-wise splits
   * construct the TensorFlow datasets
   * train the model
   * fine-tune the backbone
   * evaluate results
4. Review the saved `.keras` model files and plots.

---

## Technology Stack

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

## Engineering Principles

* Reproducible training and evaluation workflow
* Consistent preprocessing with `tf.data`
* Subject-wise evaluation for stronger experimental integrity
* Transfer learning for efficient convergence
* Attention-based refinement for improved discrimination
* Fine-tuning for domain adaptation
* Modular design that supports experimentation and extension

---

## Potential Extensions

* Experimenting with larger ConvNeXt variants
* Stronger augmentation policies such as MixUp or RandAugment
* Real-time webcam-based gesture recognition
* Flask or FastAPI deployment
* TensorFlow Lite export for edge devices
* ONNX export for cross-platform inference
* Grad-CAM based explainability

---

## License

This project is licensed under the [LICENSE](LICENSE). See the LICENSE file for details.

---

## Author

Sushant Kothari
