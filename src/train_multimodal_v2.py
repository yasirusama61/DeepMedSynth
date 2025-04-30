import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import albumentations as A

# Constants
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
SAVE_DIR = "/kaggle/working/segmentation_results_multimodal_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100

# Losses
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combo_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = 1 - dice_coef(y_true, y_pred)
    return 0.5 * bce + 0.5 * dsc

# Dataset
class MultimodalSliceDataset(Sequence):
    def __init__(self, flair_paths, t1_paths, mask_paths, batch_size, augment=False):
        self.flair_paths = flair_paths
        self.t1_paths = t1_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.slice_count = 128

        if self.augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])

    def __len__(self):
        return len(self.flair_paths) * self.slice_count // self.batch_size

    def __getitem__(self, idx):
        batch_imgs, batch_masks = [], []
        vol_idx = (idx * self.batch_size) // self.slice_count
        slice_idx = (idx * self.batch_size) % self.slice_count

        flair = np.load(self.flair_paths[vol_idx])
        t1 = np.load(self.t1_paths[vol_idx])
        mask = np.load(self.mask_paths[vol_idx])

        for i in range(self.batch_size):
            s = (slice_idx + i) % self.slice_count
            x = np.stack([flair[:, :, s], t1[:, :, s]], axis=-1)
            y = np.expand_dims(mask[:, :, s], axis=-1)

            if self.augment:
                augmented = self.aug(image=x, mask=y)
                x, y = augmented["image"], augmented["mask"]

            batch_imgs.append(x)
            batch_masks.append(y)

        return np.array(batch_imgs, dtype=np.float32), np.array(batch_masks, dtype=np.float32)

# U-Net model
def build_unet(input_shape=(128, 128, 2)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = conv_block(u1, 64)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = conv_block(u2, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return models.Model(inputs, outputs)

# Load file paths
flair_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_flair.npy")])
t1_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_t1.npy")])
mask_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

# Split data
train_flair, test_flair, train_t1, test_t1, train_mask, test_mask = train_test_split(flair_paths, t1_paths, mask_paths, test_size=0.1, random_state=42)
train_flair, val_flair, train_t1, val_t1, train_mask, val_mask = train_test_split(train_flair, train_t1, train_mask, test_size=0.1, random_state=42)

# Generators
train_gen = MultimodalSliceDataset(train_flair, train_t1, train_mask, BATCH_SIZE, augment=True)
val_gen = MultimodalSliceDataset(val_flair, val_t1, val_mask, BATCH_SIZE)
test_gen = MultimodalSliceDataset(test_flair, test_t1, test_mask, BATCH_SIZE)

# Compile model
model = build_unet()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss=combo_loss, metrics=[dice_coef])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.keras"), monitor="val_dice_coef", mode="max", save_best_only=True)
]

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# Save & plot
model.save(os.path.join(SAVE_DIR, "final_model.keras"))

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(), plt.grid(), plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))

plt.figure(figsize=(10, 5))
plt.plot(history.history["dice_coef"], label="Train Dice")
plt.plot(history.history["val_dice_coef"], label="Val Dice")
plt.legend(), plt.grid(), plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dice_plot.png"))
