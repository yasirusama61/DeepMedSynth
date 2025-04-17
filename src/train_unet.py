import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
SAVE_DIR = "/kaggle/working/segmentation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants
IMG_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 20

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

# ==== Data Generator ====
class SliceDataset(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.is_train = is_train
        self.indices = np.arange(len(self.image_paths))

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

            batch_imgs.append(np.expand_dims(image_slice, axis=-1))
            batch_masks.append(np.expand_dims(mask_slice, axis=-1))

        return np.array(batch_imgs), np.array(batch_masks)

# ==== U-Net Model with BatchNorm ====
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
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

train_img, val_img, train_mask, val_mask = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)
train_gen = SliceDataset(train_img, train_mask, BATCH_SIZE)
val_gen = SliceDataset(val_img, val_mask, BATCH_SIZE)

# ==== Build & Train ====
model = build_unet()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss=combo_loss, metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ==== Save Model ====
model.save(os.path.join(SAVE_DIR, "unet_brats_flair_segmentation.keras"))
