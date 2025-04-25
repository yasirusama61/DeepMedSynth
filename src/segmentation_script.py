import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
SAVE_DIR = "/kaggle/working/segmentation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100

# ==== Dice + BCE Loss ====
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combo_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dsc

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-6)

# ==== Data Generator ====
class SliceDataset(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, is_train=True, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.is_train = is_train
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            ])

    def __len__(self):
        return len(self.image_paths) * IMG_SIZE // self.batch_size

    def __getitem__(self, idx):
        batch_imgs = []
        batch_masks = []

        vol_idx = idx * self.batch_size // IMG_SIZE
        slice_idx = (idx * self.batch_size) % IMG_SIZE

        img = np.load(self.image_paths[vol_idx])[3]  # Use Flair
        mask = np.load(self.mask_paths[vol_idx])

        for i in range(self.batch_size):
            s_idx = (slice_idx + i) % IMG_SIZE
            image_slice = img[:, :, s_idx]
            mask_slice = mask[:, :, s_idx]

            image_slice = np.expand_dims(image_slice, axis=-1)
            mask_slice = np.expand_dims(mask_slice, axis=-1)
            mask_slice = np.clip(mask_slice, 0.01, 0.99)

            if self.augment:
                augmented = self.aug(image=image_slice, mask=mask_slice)
                image_slice = augmented['image']
                mask_slice = augmented['mask']

            batch_imgs.append(image_slice)
            batch_masks.append(mask_slice)

        return np.array(batch_imgs), np.array(batch_masks)

# ==== U-Net Model with Dropout + BatchNorm ====
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        return x

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = conv_block(u1, 64)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = conv_block(u2, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return models.Model(inputs, outputs)

# ==== Load Data ====
image_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

train_img, test_img, train_mask, test_mask = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)
train_img, val_img, train_mask, val_mask = train_test_split(train_img, train_mask, test_size=0.1, random_state=42)

train_gen = SliceDataset(train_img, train_mask, BATCH_SIZE, is_train=True, augment=True)
val_gen = SliceDataset(val_img, val_mask, BATCH_SIZE, is_train=False)
test_gen = SliceDataset(test_img, test_mask, BATCH_SIZE, is_train=False)

# ==== Build & Train ====
model = build_unet()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss=combo_loss, metrics=[dice_coef])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# ==== Save Model ====
model.save(os.path.join(SAVE_DIR, "unet_brats_flair_segmentation.keras"))

# ==== Plotting ====
plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))

plt.figure(figsize=(10,5))
plt.plot(history.history["dice_coef"], label="Train Dice")
plt.plot(history.history["val_dice_coef"], label="Val Dice")
plt.title("Training vs Validation Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dice_plot.png"))
