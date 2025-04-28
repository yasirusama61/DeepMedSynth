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


## 📈 Final Training Results (Early Stopping at Epoch 20)

- ✅ Training automatically stopped after epoch **20** due to plateau in validation loss.
- 📉 **Final Loss Values**:
  - Train Loss: ~0.425
  - Val Loss: ~0.430

- 📈 **Final Dice Coefficient**:
  - Train Dice: ~0.23
  - Val Dice: ~0.22

- 📊 Visualizations:
  ![Loss Plot](segmentation_results/loss_plot.png)
  ![Dice Plot](segmentation_results/dice_plot.png)

# 📝 CHANGELOG

## 📅 [April 26, 2025] - Final Model and Training Strategy Updates

### 🧠 Final Model Architecture Changes
- ✅ 2D U-Net Backbone (input: 128×128, single-channel Flair slices)
- ✅ Dropout (rate = 0.2) after each convolutional block to prevent overfitting
- ✅ Batch Normalization after each convolutional block
- ✅ Output Activation: Sigmoid for binary mask prediction
- ✅ Loss Function: ComboLoss (0.5 × Binary Crossentropy + 0.5 × Dice Loss)
- ✅ Evaluation Metric: Dice Coefficient
- ✅ Single output channel for whole tumor segmentation

---

### 🛠️ Training Strategy Improvements
- 🔽 Reduced Batch Size: from 64 → 16
- 🧪 Data Augmentation via Albumentations:
  - Horizontal flip
  - Shift-Scale-Rotate
  - Random brightness/contrast
- 🔁 EarlyStopping (patience = 20) on validation loss to prevent overfitting
- 📉 ReduceLROnPlateau (patience = 5) with LR decay factor 0.5
- 🔄 Resumed training from checkpoint after early stopping triggered
- 📈 Completed full training to 100 epochs

---

### 📈 Final Evaluation Metrics
- **Training Loss**: ~0.425
- **Validation Loss**: ~0.430
- **Test Loss**: ~0.4154
- **Test Dice Coefficient**: ~0.2596
