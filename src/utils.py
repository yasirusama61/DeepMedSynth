import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# ==============================
# ⚙️ General Utilities
# ==============================

def flatten_tensor(y_true, y_pred):
    """
    Flattens prediction and ground truth for loss computation.
    """
    return K.flatten(y_true), K.flatten(y_pred)

# ==============================
# 🎯 Binary Losses and Metrics
# ==============================

def binary_dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for binary segmentation.
    """
    y_true_f, y_pred_f = flatten_tensor(y_true, y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss (1 - Dice coefficient).
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return 1 - binary_dice_coefficient(y_true, y_pred, smooth)

def jaccard_loss(y_true, y_pred, smooth=1e-6):
    """
    Jaccard loss (IoU loss) for binary segmentation.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_true_f, y_pred_f = flatten_tensor(y_true, y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1 - (intersection + smooth) / (union + smooth)

def safe_binary_crossentropy(y_true, y_pred, epsilon=1e-7):
    """
    Numerically stable binary cross-entropy loss.
    """
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return K.mean(-y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))

def combo_loss(y_true, y_pred, alpha=0.5, smooth=1e-6):
    """
    Combined BCE and Dice loss (numerically stable).
    """
    bce = safe_binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred, smooth)
    return alpha * bce + (1 - alpha) * dsc

def precision_score(y_true, y_pred):
    """
    Precision metric for binary segmentation.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    return (true_positives + 1e-6) / (predicted_positives + 1e-6)

def recall_score(y_true, y_pred):
    """
    Recall metric for binary segmentation.
    """
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positives = K.sum(y_true * y_pred)
    actual_positives = K.sum(y_true)
    return (true_positives + 1e-6) / (actual_positives + 1e-6)

# ==============================
# 🧠 Multiclass Dice and IoU (Optional)
# ==============================

def multiclass_dice(y_true, y_pred, num_classes=4, smooth=1e-6):
    """
    Multiclass Dice score (averaged across channels).
    """
    dice_per_class = []
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes)
    for i in range(num_classes):
        dice_i = binary_dice_coefficient(y_true[..., i], y_pred[..., i], smooth)
        dice_per_class.append(dice_i)
    return tf.reduce_mean(tf.stack(dice_per_class))

def multiclass_jaccard(y_true, y_pred, num_classes=4, smooth=1e-6):
    """
    Multiclass Jaccard (IoU) score.
    """
    iou_per_class = []
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes)
    for i in range(num_classes):
        true_f = K.flatten(y_true[..., i])
        pred_f = K.flatten(y_pred[..., i])
        intersection = K.sum(true_f * pred_f)
        union = K.sum(true_f) + K.sum(pred_f) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou)
    return tf.reduce_mean(tf.stack(iou_per_class))


