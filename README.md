
# 🐟 Fish U-Net Segmentation

A deep learning image segmentation project for automatically separating fish from the background using a U-Net convolutional neural network.

The application allows users to upload a fish image and obtain a predicted binary segmentation mask and visualization overlay through a Streamlit web application.

---

## 📌 Project Overview

Fish image segmentation is the process of identifying the pixels belonging to a fish while separating them from the background.

In this project, a U-Net architecture was trained to perform binary semantic segmentation on fish images.

The complete workflow consists of:

Dataset → Preprocessing → Train/Validation/Test Split → U-Net Training → Evaluation → Prediction → Streamlit Deployment

---

## 🎯 Objective

The main objective is to develop a deep learning model that can accurately segment fish from their backgrounds and provide the segmentation result through an interactive Streamlit application.

---

## 📂 Dataset

The original Fish Dataset contained multiple fish species with corresponding ground-truth segmentation masks.

Due to computational limitations, the dataset was reduced for training and experimentation.

### Final Dataset

- Total image-mask pairs: 6,000
- Number of fish species: 3
- Image size: 256 × 256
- Image channels: RGB
- Mask type: Binary segmentation mask

### Selected Species

1. Black Sea Sprat
2. Gilt-Head Bream
3. Hourse Mackerel

Each fish image has a corresponding ground-truth mask.

---

## 📊 Dataset Split

The dataset was divided using a stratified split:

| Dataset | Number of Pairs | Percentage |
|---|---:|---:|
| Training | 4,200 | 70% |
| Validation | 900 | 15% |
| Testing | 900 | 15% |
| **Total** | **6,000** | **100%** |

Stratification was performed using the fish species to maintain class distribution across the splits.

---

## 🔧 Preprocessing

The following preprocessing steps were applied:

### Image preprocessing

- Images converted to RGB
- Resized to 256 × 256 pixels
- Pixel values normalized from `[0, 255]` to `[0, 1]`

### Mask preprocessing

- Masks loaded as single-channel images
- Resized to 256 × 256 using nearest-neighbor interpolation
- Converted to binary values:
  - 0 → Background
  - 1 → Fish

---

## 🧠 U-Net Architecture

A U-Net architecture was used for semantic segmentation.

The model consists of:

- Encoder
- Bottleneck
- Decoder
- Skip connections
- Final 1 × 1 convolution
- Sigmoid activation

### Architecture flow

Input

→ Encoder Block 1: 32 filters

→ Encoder Block 2: 64 filters

→ Encoder Block 3: 128 filters

→ Encoder Block 4: 256 filters

→ Bottleneck: 512 filters

→ Decoder Block 1: 256 filters

→ Decoder Block 2: 128 filters

→ Decoder Block 3: 64 filters

→ Decoder Block 4: 32 filters

→ Output: 1-channel binary mask

Input shape:

`256 × 256 × 3`

Output shape:

`256 × 256 × 1`

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Model | U-Net |
| Image Size | 256 × 256 |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss | Binary Cross Entropy + Dice Loss |
| Output Activation | Sigmoid |
| Prediction Threshold | 0.5 |
| Hardware | Google Colab T4 GPU |

---

## 📐 Evaluation Metrics

### Dice Coefficient

Dice measures the overlap between the predicted segmentation and the ground-truth mask.

Higher Dice values indicate better overlap.

### Intersection over Union (IoU)

IoU measures the intersection between the predicted and ground-truth masks divided by their union.

Higher IoU values indicate better segmentation overlap.

---

## 📈 Overall Test Results

The trained model was evaluated on the held-out test dataset.

| Metric | Score |
|---|---:|
| Dice Coefficient | **0.9692** |
| IoU | **0.9448** |

---

## 🐟 Species-wise Test Results

| Fish Species | Test Images | Average Dice | Average IoU |
|---|---:|---:|---:|
| Gilt-Head Bream | 150 | **0.9804** | **0.9616** |
| Hourse Mackerel | 150 | **0.9763** | **0.9540** |
| Black Sea Sprat | 150 | **0.9509** | **0.9188** |

These values represent the average segmentation performance for the evaluated test images of each species.

---

## 🖼️ Segmentation Results

The model produces:

1. Original fish image
2. Predicted binary segmentation mask
3. Segmentation overlay

The representative results demonstrate that the predicted segmentation closely follows the fish shape.

---

## 🚀 Streamlit Application

The trained model is integrated into a Streamlit application.

The application allows users to:

- Upload a fish image
- Resize and preprocess the image
- Generate a segmentation prediction
- View the predicted binary mask
- View the segmentation overlay
- Download the segmentation mask
- Download the segmentation overlay

### Application workflow

Upload Image

↓

Resize to 256 × 256

↓

Normalize pixel values

↓

U-Net prediction

↓

Apply threshold of 0.5

↓

Generate binary segmentation mask

↓

Display overlay

---

## 📁 Project Structure

```text
fish-unet-project/
│
├── app.py
├── requirements.txt
├── fish_unet_final.keras
├── README.md
└── results/
    └── fish_unet_presentation_results.png
