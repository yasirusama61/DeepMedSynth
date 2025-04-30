import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train_multimodal_v2 import MultimodalSliceDataset, combo_loss, dice_coef  # reuse class/loss

# === Paths ===
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
MODEL_PATH = "/kaggle/working/segmentation_results_multimodal_v2/best_model.keras"
SAVE_DIR = "/kaggle/working/segmentation_results_multimodal_v2/test_visuals"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load file paths ===
flair_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_flair.npy")])
t1_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_t1.npy")])
mask_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

# === Reproduce test split ===
from sklearn.model_selection import train_test_split
_, test_flair, _, test_t1, _, test_mask = train_test_split(flair_paths, t1_paths, mask_paths, test_size=0.1, random_state=42)

# === Load test generator ===
test_gen = MultimodalSliceDataset(test_flair, test_t1, test_mask, batch_size=16)

# === Load model ===
model = load_model(MODEL_PATH, custom_objects={'combo_loss': combo_loss, 'dice_coef': dice_coef})

# === Evaluate ===
loss, dice = model.evaluate(test_gen, verbose=2)
print(f"\n📊 Full Test Loss: {loss:.4f}")
print(f"📈 Full Test Dice Coefficient: {dice:.4f}")

# === Visualize predictions ===
X_test, Y_true = test_gen[0]
Y_pred = model.predict(X_test)

for i in range(min(8, len(X_test))):
    img = X_test[i]              # shape: (128,128,2)
    flair_img = img[:, :, 0]
    t1_img = img[:, :, 1]
    true_mask = Y_true[i, :, :, 0]
    pred_mask = Y_pred[i, :, :, 0]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(flair_img, cmap='gray')
    axs[0].set_title("Flair Input")
    axs[1].imshow(t1_img, cmap='gray')
    axs[1].set_title("T1 Input")
    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title("Ground Truth")
    axs[3].imshow(pred_mask > 0.5, cmap='gray')
    axs[3].set_title("Prediction (0.5)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sample_{i}.png"))
    plt.close()

print(f"\n✅ Test visualizations saved to: {SAVE_DIR}")
