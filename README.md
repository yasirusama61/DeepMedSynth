# 🏥 DeepMedSynth - Synthetic Medical Image Generator & Segmenter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/github/license/yasirusama61/DeepMedSynth?style=flat-square)
![Issues](https://img.shields.io/github/issues/yasirusama61/DeepMedSynth?color=critical)
![Stars](https://img.shields.io/github/stars/yasirusama61/DeepMedSynth?style=social)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

**DeepMedSynth** is an AI research project that combines **GAN-based synthetic image generation** with **U-Net-based segmentation** of brain tumors on MRI. It is designed for medical imaging R&D and serves two key purposes:

- 🧪 **Data Augmentation:** Generate realistic synthetic MRI slices using GANs  
- 🧠 **Tumor Segmentation:** Train and evaluate segmentation models on real MRI data (BraTS2020)

The project enhances medical dataset availability, privacy-preserving AI research, and visual interpretability for clinical insights.

---

## 🚀 Features

✅ Generate synthetic **X-ray** and **MRI** images using GANs  
✅ Train U-Net models for **tumor segmentation** using BraTS2020 FLAIR images  
✅ Support for **DCGAN, CycleGAN, and StyleGAN** architectures (in progress)  
✅ 📈 Visualizations for tumor heatmaps, slice-wise tumor areas, segmentation overlays  
✅ Fully reproducible **preprocessing, training, and evaluation pipelines**  

---

## 📂 Project Structure

```plaintext
DeepMedSynth/
├── data/              # 🧠 Brain tumor MRI datasets (e.g., BraTS2020)
├── models/            # 🤖 Trained GAN & segmentation models (Generator, Discriminator, U-Net)
├── results/           # 🖼️ Generated synthetic images and segmentation outputs
├── assets/            # 📊 Visualizations and overlay plots for README
├── src/               # 🧪 Python scripts (training, augmentation, preprocessing, evaluation)
├── README.md          # 📘 Project overview and usage instructions
├── .gitignore         # 🚫 Files/folders excluded from version control
├── requirements.txt   # 📦 Python dependencies and package versions


## 🔧 Installation
First, **clone** the repository:

```bash
    git clone https://github.com/yasirusama61/DeepMedSynth.git
    cd DeepMedSynthaa
```

## 🧠 BraTS2020 Dataset

The **Brain Tumor Segmentation (BraTS2020)** dataset is used in this project to train GAN-based models for generating synthetic brain MRI images with tumor labels. It serves as a standard benchmark in the field of medical image synthesis and segmentation.

### 📁 Dataset Location
```
   DeepMedSynth/data/BraTS2020/`
```

### 📦 Contents

Each case contains five volumes:

- `FLAIR`: Fluid Attenuated Inversion Recovery
- `T1w`: T1-weighted image
- `T1CE`: Post-contrast T1-weighted image
- `T2w`: T2-weighted image
- `seg`: Ground truth segmentation mask

Each volume is in **NIfTI (.nii.gz)** format.

### 🧪 Modalities Shape

All MRI volumes are preprocessed to:

- Shape: **240 × 240 × 155**
- Aligned to the same anatomical space
- Intensity-normalized per modality

### 🏷️ Segmentation Mask Labels

- `0`: Background  
- `1`: Necrotic and Non-Enhancing Tumor Core (NCR/NET)  
- `2`: Peritumoral Edema (ED)  
- `4`: Enhancing Tumor (ET)

> ⚠️ **Note**: The dataset is stored locally and not included in this repository due to size and privacy restrictions.  
You can download the dataset from the official [BraTS 2020 Challenge page](https://www.med.upenn.edu/sbia/brats2020/data.html).

---

## 🧠 BraTS2020 Preprocessing Pipeline

To prepare the BraTS2020 dataset for deep learning workflows (e.g., GANs, segmentation), we preprocess all MRI modalities into standardized `.npy` tensors.

### ✅ Preprocessing Steps

1. **Dataset Source**  
   - Located at:  
     `brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`
   - Each patient folder (e.g., `BraTS20_Training_001`) contains:
     - `t1.nii`
     - `t1ce.nii`
     - `t2.nii`
     - `flair.nii`
     - `seg.nii`

2. **Processing Details**
   - Each modality is:
     - Loaded using `nibabel`
     - Normalized to `[0, 1]` (except segmentation masks)
     - Resized to a standard shape: **(128, 128, 128)** using `scipy.ndimage.zoom`
   - Modalities are stacked into a tensor of shape: **(4, 128, 128, 128)**
   - Segmentation masks are processed separately and saved in the same target shape.

3. **Saved Outputs**
   - Saved in: `/deepmedsynth_preprocessed/`
   - File structure:
     ```
     BraTS20_Training_001_image.npy  # Contains T1, T1ce, T2, FLAIR
     BraTS20_Training_001_mask.npy   # Contains segmentation mask
     ```

4. **Tools Used**
   - `nibabel`, `numpy`, `scipy`, `tqdm`, `matplotlib`

---

⚠️ **Note:** This preprocessing ensures uniform volume dimensions and intensity ranges, which is essential for training deep generative models like GANs on 3D medical data.

## 🧪 Sample Visualization of Preprocessed BraTS2020 Data

The following image showcases a single sample from the BraTS2020 dataset after preprocessing (resized to `128×128×128` and normalized). It includes 4 MRI modalities and the corresponding segmentation mask:

<p align="center">
  <img src="assets/sample_brats_slice.png" alt="BraTS Modalities and Segmentation" width="90%">
</p>

- **T1** – T1-weighted MRI  
- **T1CE** – T1-weighted contrast-enhanced MRI  
- **T2** – T2-weighted MRI  
- **FLAIR** – Fluid-attenuated inversion recovery  
- **Segmentation** – Tumor mask with label classes encoded as pixel values

## 📊 Visualizations

Below is a slice grid of the **FLAIR modality** for a BraTS2020 subject. This grid showcases anatomical and pathological structures across different axial slices.

<p align="center">
  <img src="assets/sample_flair_grid.png" alt="FLAIR Slice Grid" width="85%">
</p>

### 🧠 Segmentation Overlay

<p align="center">
  <img src="assets/segmentation_overlay.png" alt="Segmentation Overlay on FLAIR" width="70%">
</p>

This visualization shows a FLAIR slice overlaid with its corresponding segmentation mask.  
The tumor region is clearly highlighted, useful for visually validating preprocessing quality.

## 🧠 Tumor Insights and Visualization

To better understand the spatial structure and characteristics of the BraTS dataset, we include advanced visualizations:

### 🎯 Tumor Location Heatmap

<p align="center">
  <img src="assets/tumor_location_heatmap.png" alt="Tumor Location Heatmap" width="60%">
</p>

This heatmap represents the spatial distribution of tumor voxels across all patients. Brighter areas indicate common regions where tumors tend to appear, helping to visualize spatial priors useful for generative models.

---

### 📉 Tumor Area (Voxel Count) per Slice

<p align="center">
  <img src="assets/tumor_area_slicewise.png" alt="Tumor Area per Slice" width="60%">
</p>

This plot shows how the tumor volume varies across axial slices for a sample. It helps identify slices with maximum tumor presence, useful for selecting representative slices for 2D GAN training.

---

## 🧠 Tumor Segmentation Results (BraTS2020)

We trained a U-Net on preprocessed FLAIR slices to segment tumor regions from the BraTS2020 dataset. The model achieved robust performance after 20 epochs using a combined **Dice + BCE loss**.

- 📊 **Mean Dice Score (non-empty slices):** `0.82`  
- 📈 **Loss Function:** Combo Loss (0.5 × Dice + 0.5 × BCE)  
- 🧪 **Evaluation Set:** 300 random FLAIR slices with non-empty ground truth  

---

### 🔍 Segmentation Overlay Grid

This grid shows input FLAIR slices alongside their corresponding ground truth and predicted masks. Useful for qualitative evaluation of segmentation accuracy.

<p align="center">
  <img src="assets/segmentation_overlay_grid.png" alt="Segmentation Overlay Grid" width="80%">
</p>

---

### ✅ Ground Truth vs Prediction Overlap

This visualization compares ground truth masks (🟢 green), predicted masks (🔴 red), and areas of agreement (🟡 yellow). It gives a pixel-level match/mismatch between model output and true labels.

<p align="center">
  <img src="assets/segmentation_gt_vs_pred.png" alt="Ground Truth vs Prediction Overlap" width="60%">
</p>

