import numpy as np
import tensorflow as tf
from utils import (
    dice_loss, jaccard_loss, combo_loss,
    precision_score, recall_score,
    multiclass_dice, multiclass_jaccard
)

# Simulated binary masks
y_true = tf.constant(np.random.randint(0, 2, (1, 128, 128, 1)), dtype=tf.float32)
y_pred = tf.constant(np.random.rand(1, 128, 128, 1), dtype=tf.float32)

print("🧪 Binary Metrics and Losses")
print("Dice Loss:", dice_loss(y_true, y_pred).numpy())
print("Jaccard Loss:", jaccard_loss(y_true, y_pred).numpy())
print("Combo Loss:", combo_loss(y_true, y_pred).numpy())
print("Precision:", precision_score(y_true, y_pred).numpy())
print("Recall:", recall_score(y_true, y_pred).numpy())

# Simulated multiclass prediction and ground truth
y_true_mc = tf.one_hot(tf.random.uniform((1, 128, 128), minval=0, maxval=4, dtype=tf.int32), depth=4)
y_pred_mc = tf.one_hot(tf.random.uniform((1, 128, 128), minval=0, maxval=4, dtype=tf.int32), depth=4)

print("
🧪 Multiclass Metrics")
print("Multiclass Dice:", multiclass_dice(y_true_mc, y_pred_mc).numpy())
print("Multiclass Jaccard:", multiclass_jaccard(y_true_mc, y_pred_mc).numpy())
