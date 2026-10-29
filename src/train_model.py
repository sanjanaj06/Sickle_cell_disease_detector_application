# ================================================================
# 1. IMPORTS
# ================================================================

import os
import numpy as np
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# New library in this script: matplotlib, for the training curve plots
# at the end (pre-installed on Colab, no pip install needed)
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# ================================================================
# 2. SET RANDOM SEEDS
# ================================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# 3. PATHS — matches BASE_DIR convention used in
#    train_experiment2.py / train_experiment2_rebalanced.py
# ================================================================

BASE_DIR = "/content/drive/MyDrive/Sickle Cell Detection"

DATASET_ROOT = os.path.join(BASE_DIR, "dataset_rebalanced")

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "val")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("\nDataset:")
print(DATASET_ROOT)

print("\nTrain:", TRAIN_DIR)
print("Val  :", VAL_DIR)
print("Test :", TEST_DIR)


# ================================================================
# 4. PARAMETERS
# ================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 0.001

print("\n========================================")
print("PAPER-STYLE EXPERIMENT")
print("========================================")
print("Model: MobileNetV1")
print("Weights: ImageNet")
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)
print("Optimizer: Adam")
print("Loss: Binary Crossentropy")
print("========================================")


# ================================================================
# 5. HISTOGRAM EQUALIZATION
# ================================================================
#
# The paper states that images were converted:
#
# RGB → YUV
# equalize Y channel
# YUV → RGB
#
# We implement that here.
# ================================================================

def histogram_equalization(image):
    """
    Apply histogram equalization to the Y channel
    after converting RGB -> YUV.
    """

    # ImageDataGenerator gives floating-point images.
    image = np.clip(image, 0, 255).astype(np.uint8)

    # RGB -> YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Equalize luminance channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

    # YUV -> RGB
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    return result.astype(np.float32)


# ================================================================
# 6. CUSTOM PREPROCESSING FUNCTION
# ================================================================
#
# We use ImageDataGenerator because the paper explicitly reports
# using the Keras ImageDataGenerator framework.
#
# Histogram equalization is applied after the augmentation.
# ================================================================

def paper_preprocessing(image):

    image = histogram_equalization(image)

    return image


# ================================================================
# 7. TRAINING DATA GENERATOR
# ================================================================
#
# Paper-reported augmentation:
#
# Rotation: 20 degrees
# Width shift: 20%
# Height shift: 20%
# Shear: 20%
# Zoom: 20%
# Horizontal flip
# Brightness: [0.8, 1.2]
# Channel shifts: ±20
#
# ================================================================

train_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    shear_range=0.20,

    zoom_range=0.20,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2],

    channel_shift_range=20.0,

    preprocessing_function=paper_preprocessing
)


# ================================================================
# 8. VALIDATION GENERATOR
# ================================================================
#
# IMPORTANT:
# We do NOT augment validation images.
#
# The paper discusses augmentation for the training dataset.
# Keeping validation/test untouched gives us a cleaner evaluation.
# ================================================================

val_datagen = ImageDataGenerator(
    preprocessing_function=paper_preprocessing
)


# ================================================================
# 9. TEST GENERATOR
# ================================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=paper_preprocessing
)


# ================================================================
# 10. LOAD TRAINING DATA
# ================================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True,

    seed=SEED
)


# ================================================================
# 11. LOAD VALIDATION DATA
# ================================================================

val_generator = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


# ================================================================
# 12. LOAD TEST DATA
# ================================================================

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


print("\n========================================")
print("CLASS INDICES")
print("========================================")

print(train_generator.class_indices)

print("\nTrain samples:", train_generator.samples)
print("Val samples:", val_generator.samples)
print("Test samples:", test_generator.samples)


# ================================================================
# 13. BUILD MOBILENETV1
# ================================================================
#
# MobileNetV1 with ImageNet weights.
#
# IMPORTANT:
# The paper says all convolution layers were loaded with ImageNet
# weights and the models were subsequently fine-tuned for 20 epochs.
#
# Therefore, unlike your previous conservative experiment,
# we make the MobileNet backbone trainable.
# ================================================================

print("\nBuilding MobileNetV1...")


base_model = MobileNet(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"
)


# Make the backbone trainable
base_model.trainable = True


# ================================================================
# 14. CLASSIFICATION HEAD
# ================================================================
#
# The paper does not specify the exact classification-head code.
# We use a standard GAP + sigmoid binary classifier.
# ================================================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = base_model(inputs, training=True)

x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = keras.Model(
    inputs,
    outputs
)


# ================================================================
# 15. COMPILE
# ================================================================

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(

    optimizer=optimizer,

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ================================================================
# 16. MODEL SUMMARY
# ================================================================

model.summary()


# ================================================================
# 17. COUNT TRAINABLE PARAMETERS
# ================================================================

trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.trainable_variables
])

non_trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.non_trainable_variables
])

print("\n========================================")
print("PARAMETERS")
print("========================================")

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Non-trainable parameters:",
    f"{non_trainable_params:,}"
)


# ================================================================
# 18. CALLBACKS
# ================================================================
#
# We intentionally do NOT use ReduceLROnPlateau here because the
# paper explicitly reports Adam with LR=0.001 and 20 epochs.
#
# CHANGED: model now saves to your project's models/ folder on
# Drive (like experiment2_best.keras and experiment2_rebalanced_best.keras)
# instead of Colab's ephemeral /content/ storage, so it persists
# after the runtime disconnects.
# ================================================================

MODEL_PATH = os.path.join(MODEL_DIR, "paper_style_mobilenetv1_best.keras")


checkpoint = keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


# ================================================================
# 19. TRAIN
# ================================================================

print("\n")
print("=" * 70)
print("STARTING PAPER-STYLE MOBILENETV1 TRAINING")
print("=" * 70)

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[checkpoint],

    verbose=1
)


# ================================================================
# 20. LOAD BEST MODEL
# ================================================================

print("\nLoading best model...")

best_model = keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded.")


# ================================================================
# 21. FINAL TRAINING PERFORMANCE
# ================================================================

train_loss, train_accuracy = best_model.evaluate(
    train_generator,
    verbose=1
)

print("\nTrain accuracy:", train_accuracy)
print("Train loss:", train_loss)


# ================================================================
# 22. FINAL VALIDATION PERFORMANCE
# ================================================================

val_loss, val_accuracy = best_model.evaluate(
    val_generator,
    verbose=1
)

print("\nValidation accuracy:", val_accuracy)
print("Validation loss:", val_loss)


# ================================================================
# 23. FINAL TEST PERFORMANCE
# ================================================================

test_loss, test_accuracy = best_model.evaluate(
    test_generator,
    verbose=1
)

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test loss: {test_loss:.4f}"
)


# ================================================================
# 24. PREDICTIONS
# ================================================================

test_generator.reset()

probabilities = best_model.predict(
    test_generator,
    verbose=1
).ravel()

y_true = test_generator.classes

y_pred = (probabilities >= 0.5).astype(int)


# ================================================================
# 25. CLASSIFICATION REPORT
# ================================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"],
        digits=4
    )
)


# ================================================================
# 26. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ================================================================
# 27. MEDICAL METRICS
# ================================================================

TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN) if (TP + FN) else 0

specificity = TN / (TN + FP) if (TN + FP) else 0

precision = TP / (TP + FP) if (TP + FP) else 0

npv = TN / (TN + FN) if (TN + FN) else 0

f1 = (
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (precision + sensitivity)
    else 0
)

balanced_accuracy = (
    sensitivity + specificity
) / 2


print("\n")
print("=" * 70)
print("MEDICAL CLASSIFICATION METRICS")
print("=" * 70)

print("True Positives :", TP)
print("False Negatives:", FN)
print("True Negatives :", TN)
print("False Positives:", FP)

print(
    f"\nSensitivity : {sensitivity:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"PPV         : {precision:.4f}"
)

print(
    f"NPV         : {npv:.4f}"
)

print(
    f"F1 score    : {f1:.4f}"
)

print(
    f"Balanced Acc: {balanced_accuracy:.4f}"
)


# ================================================================
# 28. ROC-AUC / PR-AUC
# ================================================================

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

pr_auc = average_precision_score(
    y_true,
    probabilities
)

print(
    f"\nROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# ================================================================
# 29. TRAINING CURVES
# ================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Paper-Style MobileNetV1 - Accuracy"
)

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Paper-Style MobileNetV1 - Loss"
)

plt.legend()
plt.grid(True)

plt.show()


# ================================================================
# 30. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print("Model                : MobileNetV1")
print("Weights              : ImageNet")
print("Fine-tuning          : Full backbone")
print("Optimizer            : Adam")
print("Learning rate        : 0.001")
print("Epochs               : 20")
print("Loss                 : Binary Crossentropy")
print("Augmentation         : Paper-style")
print("Histogram equalize   : Y-channel")
print()
print(f"Train accuracy       : {train_accuracy * 100:.2f}%")
print(f"Validation accuracy  : {val_accuracy * 100:.2f}%")
print(f"Test accuracy        : {test_accuracy * 100:.2f}%")
print(f"Sensitivity          : {sensitivity * 100:.2f}%")
print(f"Specificity          : {specificity * 100:.2f}%")
print(f"F1                   : {f1 * 100:.2f}%")
print(f"ROC-AUC              : {roc_auc:.4f}")
print(f"PR-AUC               : {pr_auc:.4f}")

print("\nModel saved at:")
print(MODEL_PATH)
