# 🏥 DeepMedSynth - Synthetic Medical Image Generator

DeepMedSynth is a **GAN-based deep learning project** designed to generate **synthetic medical images** (X-rays, MRIs) for research and AI model training. The goal is to enhance dataset availability while ensuring **data privacy and security**.

##  Features
✅ Generate synthetic **X-ray** and **MRI** images using GANs  
✅ Support for **DCGAN, CycleGAN, and StyleGAN** architectures  
✅ Open-source and privacy-preserving AI-generated medical images  
✅ Preprocessing pipeline for dataset preparation  
✅ Model training and evaluation with visualization  

## 📂 Project Structure
DeepMedSynth/  
├── data/              # Training datasets (DO NOT upload real patient data)  
├── models/            # Trained GAN models  
├── results/           # Generated synthetic images  
├── src/               # Python scripts (training, preprocessing, etc.)  
├── README.md          # Project documentation  
├── .gitignore         # Ignored files (large datasets, cache, etc.)  
├── requirements.txt   # Dependencies for the project  


## 🔧 Installation
First, **clone** the repository:

```bash
    git clone https://github.com/yasirusama61/DeepMedSynth.git
    cd DeepMedSynthaa
````

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


