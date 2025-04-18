import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# === Paths ===
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
MODEL_PATH = "/kaggle/working/segmentation_results/unet_brats_flair_segmentation.keras"

# === Load model ===
model = load_model(MODEL_PATH, compile=False)

# === Params ===
IMG_SIZE = 128
NUM_SAMPLES = 300
THRESHOLD = 0.5

# === Get image and mask paths ===
image_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

# === Sample subset (optional shuffle for random eval) ===
np.random.seed(42)
sample_indices = np.random.choice(len(image_paths), NUM_SAMPLES, replace=False)
sample_image_paths = [image_paths[i] for i in sample_indices]
sample_mask_paths = [mask_paths[i] for i in sample_indices]

# === Dice score function ===
def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# === Evaluation loop ===
dice_scores = []
for img_path, mask_path in tqdm(zip(sample_image_paths, sample_mask_paths), total=NUM_SAMPLES):
    flair = np.load(img_path)[3]  # Use FLAIR channel
    mask = np.load(mask_path)

    for i in range(IMG_SIZE):
        slice_img = flair[:, :, i]
        slice_mask = mask[:, :, i]

        # Skip empty slices
        if np.sum(slice_mask) == 0:
            continue

        # Preprocess slice
        slice_input = np.expand_dims(slice_img, axis=(0, -1))  # (1, 128, 128, 1)
        pred = model.predict(slice_input, verbose=0)[0, :, :, 0]
        pred_binary = (pred > THRESHOLD).astype(np.float32)

        score = dice_score(slice_mask, pred_binary)
        dice_scores.append(score)

# === Final score ===
mean_dice = np.mean(dice_scores)
print(f"📊 Mean Dice Score (non-empty slices only): {mean_dice:.4f}")
