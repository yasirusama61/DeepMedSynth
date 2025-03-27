"""
Author: Yasir Usama
Role: Medical AI Researcher
Project: DeepMedSynth – Synthetic Medical Image Generation using GANs

📌 Description:
This script preprocesses the BraTS2020 MRI dataset by:
- Loading all modalities (T1, T1ce, T2, FLAIR, Segmentation)
- Normalizing volumes (excluding segmentation)
- Resizing all volumes to (128, 128, 128)
- Saving them as NumPy arrays (.npy) for deep learning pipelines
"""

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

# ============================== Configuration ==============================
# Set default dataset root and output directory
BRATS_ROOT = "brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUTPUT_DIR = "deepmedsynth_preprocessed"
TARGET_SHAPE = (128, 128, 128)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================ Helper Functions =============================

def load_process_nifti(filepath, target_shape=TARGET_SHAPE, normalize=True):
    """Load NIfTI file, normalize (optional), and resize to target shape."""
    volume = nib.load(filepath).get_fdata()
    if normalize:
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, zoom=zoom_factors, order=1)
    return resized.astype(np.float32)

# ============================= Main Pipeline ==============================

def preprocess_brats2020():
    patients = sorted(os.listdir(BRATS_ROOT))
    processed_count = 0

    for patient_id in tqdm(patients, desc="📦 Processing BraTS2020 samples"):
        patient_path = os.path.join(BRATS_ROOT, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # Identify expected modality file names
        modalities = {
            "t1": os.path.join(patient_path, f"{patient_id}_t1.nii"),
            "t1ce": os.path.join(patient_path, f"{patient_id}_t1ce.nii"),
            "t2": os.path.join(patient_path, f"{patient_id}_t2.nii"),
            "flair": os.path.join(patient_path, f"{patient_id}_flair.nii"),
            "seg": os.path.join(patient_path, f"{patient_id}_seg.nii"),
        }

        if not all(os.path.exists(f) for f in modalities.values()):
            print(f"❌ Missing modality for: {patient_id}")
            continue

        try:
            # Load and process all image modalities
            image_modalities = [
                load_process_nifti(modalities["t1"]),
                load_process_nifti(modalities["t1ce"]),
                load_process_nifti(modalities["t2"]),
                load_process_nifti(modalities["flair"]),
            ]
            image_tensor = np.stack(image_modalities, axis=0)  # Shape: (4, D, H, W)

            # Load segmentation mask (no normalization)
            mask_tensor = load_process_nifti(modalities["seg"], normalize=False)

            # Save outputs
            np.save(os.path.join(OUTPUT_DIR, f"{patient_id}_image.npy"), image_tensor)
            np.save(os.path.join(OUTPUT_DIR, f"{patient_id}_mask.npy"), mask_tensor)

            processed_count += 1

        except Exception as e:
            print(f"⚠️ Error processing {patient_id}: {e}")
            continue

    print(f"✅ Done! Processed {processed_count} patients and saved to {OUTPUT_DIR}")


# ============================== Entry Point ==============================

if __name__ == "__main__":
    preprocess_brats2020()
