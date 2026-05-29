# Beware of the Dog: CNNs for Dog-Breed Identification

**Deep Learning Course Project (2025/2026)**

- Francisco Guimarães
- Sofia Reia
- Jorge Caldeira

## Overview
This project investigates the impact of data augmentation and fine-tuning on fine-grained image classification. 
Specifically, we tackle the **Kaggle Dog Breed Identification** challenge, which requires classifying images into 120 visually similar dog breeds. 

The primary scientific question addressed is: *How much does data augmentation help in improving the model's robustness and generalization?*

## Architecture & Methodology
- **Base Model:** ResNet50 (pretrained on ImageNet)
- **Modifications:** The classification head is replaced to output 120 classes.
- **Data Pipeline:** We use the Kaggle dataset, splitting it 80/20 for training and validation using stratified sampling. 
- **Experiments Conducted:**
  1. Frozen backbone without data augmentation (Baseline).
  2. Frozen backbone with training data augmentation.
  3. Robustness testing evaluating the baseline on augmented data.
  4. Partial fine-tuning of the ResNet50 backbone (unfreezing `layer4`).
- **Interpretability:** Grad-CAM is used to generate heatmaps and understand morphological breed cues utilized by the model. 
- **Metrics:** Top-1 Accuracy, Top-5 Accuracy, Macro-F1 Score, Log Loss, and breed-level Confusion Matrices.

## Installation and Execution
1. Install requirements:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn opencv-python grad-cam
   ```
2. Download the [Kaggle Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data) and place the `train.zip`, `test.zip`, and `labels.csv` files in the appropriate directories as specified in the notebook.
3. Run the cells in `DL_Project.ipynb` in order. The notebook is designed to run on Google Colab with a GPU (T4 recommended).
