import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==== Paths ====
DATA_DIR = "/kaggle/working/deepmedsynth_preprocessed"
SAVE_DIR = "/kaggle/working/segmentation_results_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Constants ====
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100

# ==== Loss Functions ====
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, gamma=0.75):
    tversky = tversky_loss(y_true, y_pred, alpha)
    return K.pow(tversky, gamma)

# ==== Data Generator ====
class SliceDataset(Sequence):
    def __init__(self, flair_paths, mask_paths, batch_size, is_train=True):
        self.flair_paths = flair_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.is_train = is_train
        self.indices = np.arange(len(self.flair_paths))
        self.IMG_SIZE = 128

    def __len__(self):
        return len(self.flair_paths) * self.IMG_SIZE // self.batch_size

    def __getitem__(self, idx):
        batch_imgs = []
        batch_masks = []

        vol_idx = idx * self.batch_size // self.IMG_SIZE
        slice_idx = (idx * self.batch_size) % self.IMG_SIZE

        flair = np.load(self.flair_paths[vol_idx])[3]  # Flair only
        mask = np.load(self.mask_paths[vol_idx])

        for i in range(self.batch_size):
            s_idx = (slice_idx + i) % self.IMG_SIZE
            flair_slice = flair[:, :, s_idx]
            mask_slice = mask[:, :, s_idx]

            input_slice = np.expand_dims(flair_slice, axis=-1)
            mask_slice = np.expand_dims(mask_slice, axis=-1)

            batch_imgs.append(input_slice)
            batch_masks.append(mask_slice)

        return np.array(batch_imgs), np.array(batch_masks)

# ==== U-Net Model ====
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
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
flair_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_image.npy")])
mask_paths  = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_mask.npy")])

# Split into train, val, test
train_flair, test_flair, train_mask, test_mask = train_test_split(flair_paths, mask_paths, test_size=0.1, random_state=42)
train_flair, val_flair, train_mask, val_mask = train_test_split(train_flair, train_mask, test_size=0.1, random_state=42)

train_gen = SliceDataset(train_flair, train_mask, BATCH_SIZE, is_train=True)
val_gen = SliceDataset(val_flair, val_mask, BATCH_SIZE, is_train=False)
test_gen = SliceDataset(test_flair, test_mask, BATCH_SIZE, is_train=False)

# ==== Build & Compile Model ====
model = build_unet(input_shape=(128, 128, 1))
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss=focal_tversky_loss,
    metrics=[dice_coef]
)

# ==== Callbacks ====
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "best_model_v2.keras"),
        monitor='val_dice_coef',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# ==== Train ====
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==== Save Model ====
model.save(os.path.join(SAVE_DIR, "final_model_v2.keras"))

# ==== Plot Loss and Dice ====
plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss (Train vs Val)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_plot_v2.png"))
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history["dice_coef"], label="Train Dice")
plt.plot(history.history["val_dice_coef"], label="Val Dice")
plt.title("Dice Coefficient (Train vs Val)")
plt.xlabel("Epochs")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dice_plot_v2.png"))
plt.close()

print("✅ Training finished and results saved successfully.")
