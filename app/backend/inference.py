"""
app/backend/inference.py - Unified inference pipeline for the Sickle Cell Web App.

Chains:
  1. MobileNetV1 CNN (Binary Disease Presence + Confidence)
  2. Grad-CAM (Spatial Visual Attention Heatmap)
  3. OpenCV Cell Analysis (RBC counting, sickle flagging, clustering, SCI%, CD%)
  4. Clinical Severity Logic (IBSI, Estimated Hb range, VORS Risk Score)
"""
import base64
import os
import sys
from typing import Any, Dict

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.cell_analysis import analyze_image, draw_overlay
from src.gradcam import GradCAM
from src.severity import assess


def _np_to_base64(image_bgr: np.ndarray, quality: int = 85) -> str:
    """Encodes a BGR OpenCV image into a base64 JPEG data URI for web rendering."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


class SickleCellPipeline:
    """
    Production wrapper loaded once at server startup.
    Keeps the TensorFlow model and Grad-CAM graph cached in memory.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "paper_style_mobilenetv1_best.keras")

        print(f"[Pipeline] Initializing Grad-CAM and loading MobileNetV1 from:\n  {model_path}")
        self.gradcam = GradCAM(model_path)
        print("[Pipeline] Model loaded and ready for inference.")

    def run(
        self,
        image_path: str,
        age: float,
        gender: str,
        pain_crises_per_year: float,
        spo2: float,
    ) -> Dict[str, Any]:
        """
        Runs the full diagnostic and severity pipeline on a blood smear image.
        """
        # 1. Load original image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        # 2. Gate 1: Deep Learning Classification & Grad-CAM
        cam_result = self.gradcam.predict_and_explain(image_path)
        prob = float(cam_result["probability"])
        is_positive = prob >= 0.5
        label = "Positive" if is_positive else "Negative"
        confidence = prob * 100.0 if is_positive else (1.0 - prob) * 100.0

        # Grad-CAM overlay comes back in RGB, convert to BGR for encoding
        gradcam_bgr = cv2.cvtColor(cam_result["overlay"], cv2.COLOR_RGB2BGR)
        gradcam_b64 = _np_to_base64(gradcam_bgr)

        # 3. Gate 2: OpenCV Cell Morphology & Clustering
        cell_result = analyze_image(image_bgr)
        cell_summary = cell_result.summary()
        cell_overlay_bgr = draw_overlay(image_bgr, cell_result)
        cell_overlay_b64 = _np_to_base64(cell_overlay_bgr)

        # 4. Gate 3: Clinical Severity & Vaso-Occlusion Risk Engine
        # Only run severity assessment if CNN says Positive AND morphology found sickle cells
        disease_detected = is_positive and (cell_summary["sickle_count"] > 0)

        if disease_detected:
            clinical_report = assess(
                sci=cell_summary["sci"],
                cd=cell_summary["cd"],
                age=float(age),
                gender=str(gender),
                pain_crises_per_year=float(pain_crises_per_year),
                spo2=float(spo2),
            )
        else:
            clinical_report = None

        return {
            "prediction": {
                "label": label,
                "is_positive": is_positive,
                "disease_detected": disease_detected,
                "confidence_percent": round(confidence, 2),
                "raw_probability": round(prob, 4),
            },
            "cell_analysis": cell_summary,
            "clinical_assessment": clinical_report,
            "visualizations": {
                "gradcam_overlay": gradcam_b64,
                "cell_overlay": cell_overlay_b64,
            },
        }


# Global singleton instance for Flask
_PIPELINE_INSTANCE = None


def get_pipeline() -> SickleCellPipeline:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = SickleCellPipeline()
    return _PIPELINE_INSTANCE