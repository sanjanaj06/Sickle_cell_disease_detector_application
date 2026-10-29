"""
evaluate.py

Loads a saved model and evaluates it against a given test set directory,
including medically-relevant classification metrics (sensitivity,
specificity, PPV, NPV, ROC-AUC) rather than just plain accuracy.

Usage (from Colab, after cd'ing into the project root):
    !python src/evaluate.py
"""

import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)

# Importing model.py runs its @register_keras_serializable-decorated
# functions (e.g. mobilenet_preprocess), needed ONLY for models from the
# Experiment 2 family that bake preprocessing into a Lambda layer inside
# the model graph. Harmless to import even when evaluating a model that
# doesn't use it.
import model  # noqa: F401

# --- UPDATE these two paths to match your Drive folder structure ---
TEST_DIR = "/content/drive/MyDrive/Sickle Cell Detection/dataset_rebalanced/test"
MODEL_PATH = "/content/drive/MyDrive/Sickle Cell Detection/models/paper_style_lr_experiment_best.keras"

# IMPORTANT: set this to match how the model being evaluated was trained.
#   - Experiment 2 family (experiment2_best.keras, experiment2_rebalanced_best.keras):
#     preprocessing is baked into the model's own Lambda layer -> False
#   - Paper-style family (paper_style_*.keras):
#     preprocessing (Y-channel histogram equalization) was applied
#     externally via ImageDataGenerator during training -> True
APPLY_HISTOGRAM_EQUALIZATION = True

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def _histogram_equalization_numpy(image):
    """Exact port of paper_preprocessing() from
    train_paper_style_lr_experiment.py, so evaluation matches training."""
    image = image.numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return result.astype(np.float32)


def _tf_histogram_equalization(image, label):
    """Wraps the cv2-based function above so it can run inside a
    tf.data pipeline. tf.py_function is needed because cv2 operations
    aren't native TensorFlow ops."""
    image = tf.py_function(
        func=_histogram_equalization_numpy, inp=[image], Tout=tf.float32
    )
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label


def load_test_set(test_dir):
    """Loads the test set the same way training data was loaded, with
    class_names set explicitly so negative=0, positive=1 always matches
    how the model was trained.

    If APPLY_HISTOGRAM_EQUALIZATION is True, images are unbatched,
    histogram-equalized per-image (matching ImageDataGenerator's
    preprocessing_function behavior during training), then rebatched.
    If False, no external preprocessing is applied — the saved model is
    assumed to handle it internally via its own Lambda layer."""
    if APPLY_HISTOGRAM_EQUALIZATION:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            class_names=["negative", "positive"],
            image_size=IMG_SIZE,
            batch_size=None,  # unbatched so equalization runs per-image
            shuffle=False,
        )
        test_ds = test_ds.map(
            _tf_histogram_equalization, num_parallel_calls=tf.data.AUTOTUNE
        )
        test_ds = test_ds.batch(BATCH_SIZE)
    else:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            class_names=["negative", "positive"],
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    return test_ds


def compute_medical_metrics(y_true, y_pred, y_pred_probs, threshold=0.5):
    """
    Computes classification metrics framed for a diagnostic screening
    context, where positive = sickle cell present.

    Sensitivity (Recall+): of all TRUE positive cases, how many did the
        model correctly catch? This is the most clinically important
        number for a screening tool — a low sensitivity means the model
        is missing real sickle cell cases (false negatives), which is
        the costlier type of error in a medical context.

    Specificity: of all TRUE negative cases, how many did the model
        correctly clear? A low specificity means healthy samples get
        flagged unnecessarily (false positives) — an inconvenience, but
        generally less dangerous than a missed positive.

    PPV (Precision): of everything the model FLAGGED as positive, how
        many really are positive? Matters for how much to trust a
        positive result before further testing.

    NPV: of everything the model CLEARED as negative, how many really
        are negative? Matters for how much to trust a "clear" result.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)

    print(f"\n========== MEDICAL CLASSIFICATION METRICS (threshold={threshold}) ==========")
    print(f"True Positives (correctly caught positive cases):  {tp}")
    print(f"False Negatives (MISSED positive cases):           {fn}")
    print(f"True Negatives (correctly cleared negative cases): {tn}")
    print(f"False Positives (false alarms):                    {fp}")
    print()
    print(f"Sensitivity (Recall+):  {sensitivity:.4f}  <- catches this fraction of real positive cases")
    print(f"Specificity:            {specificity:.4f}  <- correctly clears this fraction of real negative cases")
    print(f"PPV (Precision):        {ppv:.4f}  <- of predicted positives, this fraction are truly positive")
    print(f"NPV:                    {npv:.4f}  <- of predicted negatives, this fraction are truly negative")
    print(f"F1 score:               {f1:.4f}")
    print(f"Balanced accuracy:      {bal_acc:.4f}  <- (sensitivity + specificity) / 2, robust to class imbalance")
    print(f"ROC-AUC:                {roc_auc:.4f}  <- threshold-independent separability of the two classes")
    print(f"PR-AUC:                 {pr_auc:.4f}  <- precision/recall tradeoff, useful given class imbalance")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def threshold_sweep(y_true, y_pred_probs, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """
    Shows how sensitivity/specificity trade off as the decision threshold
    changes. Default classification uses threshold=0.5, but for a
    screening tool it's common to deliberately lower the threshold to
    prioritize sensitivity (catch more real cases) at the cost of more
    false alarms — since missing a case is usually worse than a false
    alarm that gets ruled out on follow-up.
    """
    print("\n========== THRESHOLD SWEEP ==========")
    print(f"{'Threshold':<12}{'Sensitivity':<14}{'Specificity':<14}{'PPV':<10}")
    for t in thresholds:
        y_pred_t = (y_pred_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"{t:<12}{sens:<14.4f}{spec:<14.4f}{ppv:<10.4f}")


def evaluate_model(model_path, test_dir):
    test_ds = load_test_set(test_dir)
    loaded_model = tf.keras.models.load_model(model_path)

    loss, accuracy = loaded_model.evaluate(test_ds)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).flatten()
    y_pred_probs = loaded_model.predict(test_ds).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    medical_metrics = compute_medical_metrics(y_true, y_pred, y_pred_probs)
    threshold_sweep(y_true, y_pred_probs)

    return accuracy, loss, y_true, y_pred, medical_metrics


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, TEST_DIR)
