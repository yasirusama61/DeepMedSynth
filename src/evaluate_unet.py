import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from matplotlib import pyplot as plt
#from train_unet_improved import SliceDataset  # make sure same class used

# === Paths ===
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
MODEL_PATH = "/kaggle/working/segmentation_results_v4/best_model.keras"
RESULTS_DIR = "/kaggle/working/segmentation_results_v4/test_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load test data ===
image_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

from sklearn.model_selection import train_test_split
_, test_img, _, test_mask = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)
test_gen = SliceDataset(test_img, test_mask, batch_size=16)

# === Load Model ===
custom_objects = {
    "dice_coef": lambda y_true, y_pred: tf.keras.backend.mean(y_true * y_pred),
    "combo_tversky_loss": lambda y_true, y_pred: 0.0
}  # if needed
model = load_model(MODEL_PATH, custom_objects=custom_objects)
print("✅ Model loaded.")

# === Evaluation ===
y_true_all, y_pred_all = [], []
for x_batch, y_batch in test_gen:
    pred_batch = model.predict(x_batch)
    y_true_all.append(y_batch.flatten())
    y_pred_all.append((pred_batch > 0.5).astype(np.uint8).flatten())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

# === Metrics ===
precision = precision_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)
iou = jaccard_score(y_true_all, y_pred_all)

with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"IoU:       {iou:.4f}\n")

print(f"📊 Test metrics saved to {RESULTS_DIR}/metrics.txt")

# === Visualize Predictions ===
x_vis, y_vis = test_gen[0]
y_pred_vis = model.predict(x_vis)

for i in range(min(8, len(x_vis))):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(x_vis[i,...,0], cmap='gray')
    axs[0].set_title("Input")
    axs[1].imshow(y_vis[i,...,0], cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(y_pred_vis[i,...,0] > 0.5, cmap='gray')
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"sample_{i}.png"))
    plt.close()

print(f"🖼️ Visual results saved to {RESULTS_DIR}")
