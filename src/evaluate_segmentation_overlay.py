import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === Configuration ===
MODEL_PATH = "segmentation_results/unet_brats_flair_segmentation.keras"
DATA_DIR = "deepmedsynth_preprocessed"
SAVE_PATH = "segmentation_results/overlay_debug.png"
IMG_SIZE = 128

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# === Load sample data ===
image_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

# Choose a random volume
idx = np.random.randint(len(image_paths))
image_volume = np.load(image_paths[idx])[3]  # Flair modality
mask_volume = np.load(mask_paths[idx])

# Choose a non-empty slice
for slice_index in range(IMG_SIZE):
    if np.sum(mask_volume[:, :, slice_index]) > 0:
        flair = image_volume[:, :, slice_index]
        true_mask = mask_volume[:, :, slice_index]
        break

# Prepare input
input_tensor = np.expand_dims(flair, axis=(0, -1))  # Shape: (1, 128, 128, 1)
pred_mask = model.predict(input_tensor)[0, :, :, 0]
pred_binary = (pred_mask > 0.5).astype(np.uint8)

# === Overlay Visualization ===
def plot_prediction_overlay(flair_slice, gt_mask, pred_mask, save_path=None):
    tp = np.logical_and(gt_mask == 1, pred_mask == 1)   # True Positive → Yellow
    fp = np.logical_and(gt_mask == 0, pred_mask == 1)   # False Positive → Red
    fn = np.logical_and(gt_mask == 1, pred_mask == 0)   # False Negative → Green

    # Normalize FLAIR for RGB visualization
    flair_norm = (flair_slice - np.min(flair_slice)) / (np.max(flair_slice) - np.min(flair_slice))
    rgb = np.stack([flair_norm]*3, axis=-1)

    rgb[fp] = [1.0, 0.0, 0.0]  # Red
    rgb[fn] = [0.0, 1.0, 0.0]  # Green
    rgb[tp] = [1.0, 1.0, 0.0]  # Yellow

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title("GT vs Prediction Overlay")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

plot_prediction_overlay(flair, true_mask, pred_binary, SAVE_PATH)
