## 📅 Update: April 25, 2025 — U-Net Segmentation Stability & Overfitting Fixes

### ✨ Key Improvements

- 🔧 **Added Dropout and BatchNorm** to all U-Net convolutional blocks to improve feature generalization and reduce memorization.

- 🧠 **Reduced Batch Size** from `64 → 16` to increase update frequency and mitigate overfitting on small slice-wise batches.

- 🧪 **Upgraded Data Augmentation** using **Albumentations** for spatially-aware transforms:
  - Horizontal flip
  - Shift/scale/rotate
  - Random brightness/contrast

- 📉 **Loss Function Tweaks**:
  - Continued with **ComboLoss** (`0.5 × BCE + 0.5 × Dice`)
  - Smoothed hard 0/1 masks with clipping to stabilize early epochs

- 📏 **Replaced Accuracy with Dice Coefficient** as the primary metric for segmentation performance.

- 📈 **Visualization Improvements**:
  - Loss curves (train, val)
  - Dice coefficient curves (train, val)

- 🔁 **Training Strategy Enhanced**:
  - Added `EarlyStopping` with `patience = 10`
  - Added `ReduceLROnPlateau` to dynamically lower learning rate if validation loss plateaus
