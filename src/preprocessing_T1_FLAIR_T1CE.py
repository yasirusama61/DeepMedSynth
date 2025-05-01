import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

# Paths
brats_root = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_dir = "/kaggle/working/deepmedsynth_preprocessed"
os.makedirs(output_dir, exist_ok=True)

TARGET_SHAPE = (128, 128, 128)

def load_process_nifti(path, target_shape=TARGET_SHAPE, normalize=True):
    data = nib.load(path).get_fdata()
    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom=zoom_factors, order=1)
    return resized.astype(np.float32)

patients = sorted(os.listdir(brats_root))
processed = 0

for patient_id in tqdm(patients, desc="Processing BraTS2020 samples"):
    patient_path = os.path.join(brats_root, patient_id)
    if not os.path.isdir(patient_path): continue

    t1 = os.path.join(patient_path, f"{patient_id}_t1.nii")
    t1ce = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
    flair = os.path.join(patient_path, f"{patient_id}_flair.nii")
    seg = os.path.join(patient_path, f"{patient_id}_seg.nii")

    if not all(os.path.exists(p) for p in [t1, t1ce, flair, seg]):
        print(f"❌ Missing modality for: {patient_id}")
        continue

    modalities = {
        "t1": load_process_nifti(t1),
        "t1ce": load_process_nifti(t1ce),
        "flair": load_process_nifti(flair),
        "seg": load_process_nifti(seg, normalize=False)
    }

    image_tensor = np.stack([modalities["t1"], modalities["t1ce"], modalities["flair"]], axis=0)
    np.save(os.path.join(output_dir, f"{patient_id}_image.npy"), image_tensor)
    np.save(os.path.join(output_dir, f"{patient_id}_mask.npy"), modalities["seg"])

    processed += 1

print(f"✅ Done! Total processed samples: {processed}")
