import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import albumentations as A
from albumentations.core.composition import OneOf
from matplotlib import pyplot as plt

# ========= Paths =========
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
SAVE_DIR = "/kaggle/working/segmentation_results_v4"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= Dataset =========
class SliceDataset(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(image_paths))

        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                OneOf([
                    A.ElasticTransform(p=0.5),
                    A.GridDistortion(p=0.5)
                ], p=0.3)
            ])

    def __len__(self):
        return len(self.image_paths) * 128 // self.batch_size

    def __getitem__(self, idx):
        batch_imgs, batch_masks = [], []
        vol_idx = idx * self.batch_size // 128
        slice_idx = (idx * self.batch_size) % 128

        img_vol = np.load(self.image_paths[vol_idx])
        mask_vol = np.load(self.mask_paths[vol_idx])

        for i in range(self.batch_size):
            s_idx = (slice_idx + i) % 128
            img_slice = img_vol[:, :, :, s_idx].transpose(1, 2, 0)
            raw_mask_slice = mask_vol[:, :, s_idx]
            mask_slice = (raw_mask_slice > 0).astype(np.float32)[..., np.newaxis]

            if self.augment:
                augmented = self.aug(image=img_slice, mask=mask_slice)
                img_slice, mask_slice = augmented["image"], augmented["mask"]

            batch_imgs.append(img_slice)
            batch_masks.append(mask_slice)

        return np.array(batch_imgs), np.array(batch_masks)

# ========= Loss =========
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    TP = tf.keras.backend.sum(y_true_f * y_pred_f)
    FP = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    FN = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    return 1 - (TP + smooth) / (TP + alpha * FP + (1 - alpha) * FN + smooth)

def combo_tversky_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    tversky = tversky_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * tversky

# ========= U-Net Deep =========
def build_unet(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters, dropout_rate=0):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256, dropout_rate=0.3)
    p4 = layers.MaxPooling2D()(c4)
    c5 = conv_block(p4, 512, dropout_rate=0.3)

    u1 = layers.UpSampling2D()(c5); u1 = layers.concatenate([u1, c4]); c6 = conv_block(u1, 256)
    u2 = layers.UpSampling2D()(c6); u2 = layers.concatenate([u2, c3]); c7 = conv_block(u2, 128)
    u3 = layers.UpSampling2D()(c7); u3 = layers.concatenate([u3, c2]); c8 = conv_block(u3, 64)
    u4 = layers.UpSampling2D()(c8); u4 = layers.concatenate([u4, c1]); c9 = conv_block(u4, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)
    return models.Model(inputs, outputs)

# ========= Load Files =========
image_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

from sklearn.model_selection import train_test_split
train_img, test_img, train_mask, test_mask = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)
train_img, val_img, train_mask, val_mask = train_test_split(train_img, train_mask, test_size=0.1, random_state=42)

train_gen = SliceDataset(train_img, train_mask, batch_size=16, augment=True)
val_gen   = SliceDataset(val_img, val_mask, batch_size=16)
test_gen  = SliceDataset(test_img, test_mask, batch_size=16)

# ========= Train =========
model = build_unet(input_shape=(128,128,3))
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
              loss=combo_tversky_loss, metrics=[dice_coef])

callbacks = [
    ModelCheckpoint(f"{SAVE_DIR}/best_model.keras", save_best_only=True, monitor="val_dice_coef", mode="max"),
    EarlyStopping(monitor="val_dice_coef", patience=30, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks)

# ========= Save Plots =========
plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss Curve"); plt.grid(); plt.savefig(f"{SAVE_DIR}/loss_plot.png")

plt.figure(figsize=(10,5))
plt.plot(history.history["dice_coef"], label="Train Dice")
plt.plot(history.history["val_dice_coef"], label="Val Dice")
plt.legend(); plt.title("Dice Coefficient Curve"); plt.grid(); plt.savefig(f"{SAVE_DIR}/dice_plot.png")

print(f"✅ Finished training. Model + plots saved in {SAVE_DIR}")
