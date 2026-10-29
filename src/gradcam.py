
"""
gradcam.py — Grad-CAM / CAM explainability for the MobileNetV1
sickle cell detection model.

============================================================
BACKGROUND / KEY FINDINGS (documented here so future-you or a
reader of the report understands why the code looks like this)
============================================================

1. NESTED BACKBONE BUG (Keras 3):
   The trained model wraps MobileNet as a single nested Functional
   layer ("mobilenet_1.00_224"). Building one Model that spans from
   the OUTER input straight into an INNER layer's output triggers a
   Keras 3 graph-tracing bug (KeyError during Functional.call()).
   Fix: build the conv sub-model entirely INSIDE the backbone
   (inputs=backbone.input), then manually re-apply GAP + Dense
   outside the tape instead of relying on a single spanning Model.

2. SIGMOID SATURATION:
   Backpropagating from the sigmoid probability (not the logit)
   produces exactly-zero gradients whenever the model is very
   confident (|logit| large enough that sigmoid'(x) underflows to
   0.0 in float32). This showed up as flat, uninformative heatmaps
   for the model's most confident predictions — exactly the ones
   we most needed to inspect. Fix: compute gradients w.r.t. the
   pre-sigmoid LOGIT, obtained by manually applying the dense
   layer's weights instead of calling the sigmoid-activated layer.

3. GAP + LINEAR HEAD == CLASSIC CAM:
   Because GlobalAveragePooling2D + Dense are both linear, the
   gradient of the logit w.r.t. each spatial position of the last
   conv layer is CONSTANT across the image (= dense_weight / (H*W)).
   This means Grad-CAM here mathematically reduces to the older
   "CAM" method — pooled_grads never varies by image, only
   conv_output does. This is expected and correct, not a bug.

4. DIAGNOSTIC CONCLUSION (MobileNetV1, paper_style_mobilenetv1_best,
   91.19% test acc): high-confidence false negatives show near-zero
   or zero spatial evidence for the positive class almost everywhere
   in the image — NOT a diluted-but-present signal outvoted by GAP.
   This does NOT support the "needle in haystack" / GAP-dilution
   hypothesis. It's more consistent with (a) genuinely low-signal /
   hard positive cases, and/or (b) possible label noise on a subset
   of the dataset. See project report for full writeup.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


# ================================================================
# CONFIG — matches training pipeline in train_paper_style_mobilenetv1.py
# ================================================================

DEFAULT_BACKBONE_NAME = "mobilenet_1.00_224"
DEFAULT_LAST_CONV_NAME = "conv_pw_13_relu"
DEFAULT_GAP_NAME = "global_average_pooling2d"
DEFAULT_DENSE_NAME = "dense"
IMG_SIZE = (224, 224)


class GradCAM:
    """
    Wraps a trained MobileNetV1 sickle-cell model and exposes
    Grad-CAM / CAM heatmap generation, matched exactly to the
    paper-style preprocessing used at training time.

    Usage:
        gc = GradCAM(model_path)
        result = gc.predict_and_explain("path/to/image.png")
        # result["probability"], result["label"],
        # result["heatmap"], result["overlay"]
    """

    def __init__(
        self,
        model_path,
        backbone_name=DEFAULT_BACKBONE_NAME,
        last_conv_name=DEFAULT_LAST_CONV_NAME,
        gap_name=DEFAULT_GAP_NAME,
        dense_name=DEFAULT_DENSE_NAME,
    ):
        self.model = keras.models.load_model(model_path)

        backbone = self.model.get_layer(backbone_name)
        last_conv_layer = backbone.get_layer(last_conv_name)

        # Built entirely inside the backbone — avoids the Keras 3
        # nested-Functional tracing bug (see module docstring, point 1).
        self.conv_grad_model = keras.Model(
            inputs=backbone.input,
            outputs=last_conv_layer.output
        )

        self.gap_layer = self.model.get_layer(gap_name)
        dense_layer = self.model.get_layer(dense_name)

        dense_weights, dense_bias = dense_layer.get_weights()
        self.dense_weights = tf.convert_to_tensor(dense_weights)
        self.dense_bias = tf.convert_to_tensor(dense_bias)

    # ------------------------------------------------------------
    # PREPROCESSING — must exactly match training (paper-style)
    # ------------------------------------------------------------

    @staticmethod
    def _histogram_equalization(image):
        """RGB -> YUV -> equalize Y channel -> RGB. Matches training exactly."""
        image = np.clip(image, 0, 255).astype(np.uint8)
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return result.astype(np.float32)

    def preprocess(self, img_path, target_size=IMG_SIZE):
        """
        Loads an image and applies the same preprocessing used at
        training time. NOTE: no rescale/normalize step — the model
        was trained on raw [0,255] float32 + histogram equalization
        only. Do not add external normalization here.
        """
        img = tf.keras.utils.load_img(img_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = self._histogram_equalization(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # ------------------------------------------------------------
    # CORE CAM COMPUTATION
    # ------------------------------------------------------------

    def _conv_output_and_grads(self, img_array):
        """
        Runs the forward + backward pass once and returns the raw
        conv feature map and the gradient of the LOGIT (not the
        sigmoid probability — see module docstring, point 2) w.r.t.
        that feature map.
        """
        img_tensor = tf.convert_to_tensor(img_array)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_output = self.conv_grad_model(img_tensor)
            x = self.gap_layer(conv_output)
            logit = tf.matmul(x, self.dense_weights) + self.dense_bias
            class_channel = logit[:, 0]

        grads = tape.gradient(class_channel, conv_output)
        return conv_output[0], grads, logit.numpy()[0][0]

    def raw_signed_cam(self, img_array):
        """
        DIAGNOSTIC use. Returns the signed (pre-ReLU) CAM — positive
        values indicate genuine evidence FOR the positive class at
        that spatial location, negative values indicate evidence
        against. Do not normalize/display this directly with a
        single-ended colormap; use a diverging one centered at 0.
        """
        conv_output, _, logit = self._conv_output_and_grads(img_array)
        cam = tf.tensordot(conv_output, self.dense_weights[:, 0], axes=[[2], [0]])
        return cam.numpy(), logit

    def heatmap(self, img_array):
        """
        VISUALIZATION use. Standard Grad-CAM heatmap: ReLU'd and
        normalized to [0, 1]. Suitable for the JET-colormap overlay
        shown to clinicians/end users in the Flask app.
        """
        conv_output, grads, logit = self._conv_output_and_grads(img_array)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), logit

    # ------------------------------------------------------------
    # OVERLAY / DISPLAY
    # ------------------------------------------------------------

    @staticmethod
    def make_overlay(img_path, heatmap, alpha=0.4, target_size=IMG_SIZE):
        """Returns an RGB numpy array with the heatmap overlaid on the original image."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        heatmap_resized = cv2.resize(heatmap, target_size)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    def save_overlay(self, img_path, output_path, alpha=0.4):
        """Convenience method: compute heatmap, overlay, and write to disk (e.g. for Flask to serve)."""
        img_array = self.preprocess(img_path)
        heatmap, logit = self.heatmap(img_array)
        overlay = self.make_overlay(img_path, heatmap, alpha=alpha)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay_bgr)
        return output_path

    # ------------------------------------------------------------
    # HIGH-LEVEL CONVENIENCE (for inference.py in the Flask app)
    # ------------------------------------------------------------

    def predict_and_explain(self, img_path, threshold=0.5):
        """
        Runs prediction + Grad-CAM in one call. Returns a dict ready
        to be consumed by a Flask route:
            {
                "probability": float,
                "label": "positive" | "negative",
                "heatmap": np.ndarray (H, W), values in [0, 1],
                "overlay": np.ndarray (H, W, 3), RGB uint8
            }
        """
        img_array = self.preprocess(img_path)
        heatmap, logit = self.heatmap(img_array)
        probability = float(1 / (1 + np.exp(-logit)))
        label = "positive" if probability >= threshold else "negative"
        overlay = self.make_overlay(img_path, heatmap)

        return {
            "probability": probability,
            "label": label,
            "heatmap": heatmap,
            "overlay": overlay,
        }