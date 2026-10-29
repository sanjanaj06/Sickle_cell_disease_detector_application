"""
project_walkthrough.py

Complete end-to-end inference walkthrough for the
Sickle Cell Detection project.

Pipeline:
    Original Image
        ↓
    Resize to 224x224
        ↓
    RGB Conversion
        ↓
    Histogram Equalization
        ↓
    MobileNetV1 Prediction
        ↓
    OpenCV Cell Analysis
        ↓
    Sickle-like Cell Detection
        ↓
    Cell Clustering
        ↓
    Grad-CAM Explanation
        ↓
    Final Result

Model:
    models/paper_style_mobilenetv1_best.keras

Input:
    scripts/scd.jpg

Output:
    outputs/project_walkthrough/scd/
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ================================================================
# PROJECT PATH SETUP
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# PROJECT IMPORTS
# ================================================================

from src.cell_analysis import analyze_image, draw_overlay
from src.gradcam import GradCAM


# ================================================================
# CONFIGURATION
# ================================================================

IMAGE_PATH = PROJECT_ROOT / "scripts" / "scd.jpg"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "paper_style_mobilenetv1_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "project_walkthrough"
    / "scd"
)

IMG_SIZE = (224, 224)


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def ensure_output_directory():
    """Create walkthrough output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image(filename, image):
    """Save image safely."""
    path = OUTPUT_DIR / filename

    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Could not save image: {path}")

    return path


def histogram_equalization_rgb(image_rgb):
    """
    Apply histogram equalization to the luminance channel.

    This matches the preprocessing used by src/gradcam.py
    and the paper-style evaluation pipeline.
    """

    yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    return equalized


def put_title(image, title):
    """
    Add a title bar to an image for easier PPT visualization.
    """

    image = image.copy()

    h, w = image.shape[:2]

    bar_height = 40

    output = np.zeros(
        (h + bar_height, w, 3),
        dtype=np.uint8
    )

    output[bar_height:] = image

    cv2.putText(
        output,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ================================================================
# CELL VISUALIZATION FUNCTIONS
# ================================================================

def create_detected_cells_image(image_bgr, result):
    """
    Visualize accepted RBC candidates.

    Green:
        Accepted RBC

    Gray:
        Background
    """

    output = image_bgr.copy()

    for cell in result.cells:

        # Only visualize cells that survived filtering.
        color = (0, 255, 0)

        cv2.rectangle(
            output,
            (
                int(cell.bbox[0]),
                int(cell.bbox[1]),
            ),
            (
                int(cell.bbox[0] + cell.bbox[2]),
                int(cell.bbox[1] + cell.bbox[3]),
            ),
            color,
            2,
        )

        cx, cy = cell.centroid

        cv2.circle(
            output,
            (int(cx), int(cy)),
            3,
            color,
            -1,
        )

    return output


def create_sickle_detection_image(image_bgr, result):
    """
    Highlight sickle-like cells.

    Green:
        Normal RBC

    Red:
        Sickle-like cell
    """

    output = image_bgr.copy()

    for cell in result.cells:

        if cell.is_sickle:
            color = (0, 0, 255)
            thickness = 3
        else:
            color = (0, 255, 0)
            thickness = 1

        x, y, w, h = cell.bbox

        cv2.rectangle(
            output,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            thickness,
        )

    return output


def create_clustering_image(image_bgr, result):
    """
    Visualize clustered sickle-like cells.

    Clustered sickle cells:
        Magenta

    Isolated sickle-like cells:
        Red

    Normal RBCs:
        Green
    """

    output = image_bgr.copy()

    for cell in result.cells:

        if not cell.is_sickle:
            color = (0, 255, 0)
            thickness = 1

        elif cell.cluster_id is not None:
            color = (255, 0, 255)
            thickness = 3

        else:
            color = (0, 0, 255)
            thickness = 3

        x, y, w, h = cell.bbox

        cv2.rectangle(
            output,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            thickness,
        )

        if cell.cluster_id is not None:

            cx, cy = cell.centroid

            cv2.putText(
                output,
                f"C{cell.cluster_id}",
                (
                    int(cx),
                    int(cy),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return output


# ================================================================
# MAIN WALKTHROUGH
# ================================================================

def main():

    ensure_output_directory()

    print()
    print("=" * 68)
    print("          SICKLE CELL DETECTION - IMAGE PIPELINE")
    print("=" * 68)
    print()

    # ------------------------------------------------------------
    # CHECK FILES
    # ------------------------------------------------------------

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"\nInput image not found:\n{IMAGE_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    print(f"Input image : {IMAGE_PATH}")
    print(f"Model       : {MODEL_PATH}")
    print(f"Output      : {OUTPUT_DIR}")
    print()

    # ============================================================
    # 1. ORIGINAL IMAGE
    # ============================================================

    print("-" * 68)
    print("[1] ORIGINAL IMAGE")
    print("-" * 68)

    original_bgr = cv2.imread(str(IMAGE_PATH))

    if original_bgr is None:

        raise RuntimeError(
            f"Could not read image:\n{IMAGE_PATH}"
        )

    original_h, original_w = original_bgr.shape[:2]

    print(f"    Original size : {original_w} x {original_h}")
    print("    Color format  : BGR")

    save_image(
        "01_original.png",
        put_title(
            original_bgr,
            "01 - Original Image"
        ),
    )

    print(f"    Saved         : 01_original.png")
    print()

    # ============================================================
    # 2. RESIZE
    # ============================================================

    print("-" * 68)
    print("[2] IMAGE RESIZING")
    print("-" * 68)

    resized_bgr = cv2.resize(
        original_bgr,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    print(
        f"    Resize        : "
        f"{original_w} x {original_h} -> "
        f"{IMG_SIZE[0]} x {IMG_SIZE[1]}"
    )

    save_image(
        "02_resized.png",
        put_title(
            resized_bgr,
            "02 - Resized to 224 x 224"
        ),
    )

    print("    Saved         : 02_resized.png")
    print()

    # ============================================================
    # 3. RGB CONVERSION
    # ============================================================

    print("-" * 68)
    print("[3] COLOR CONVERSION")
    print("-" * 68)

    resized_rgb = cv2.cvtColor(
        resized_bgr,
        cv2.COLOR_BGR2RGB,
    )

    print("    Conversion    : BGR -> RGB")
    print("    Channels      : 3")

    # Convert back to BGR only for OpenCV saving.
    rgb_display_bgr = cv2.cvtColor(
        resized_rgb,
        cv2.COLOR_RGB2BGR,
    )

    save_image(
        "03_rgb.png",
        put_title(
            rgb_display_bgr,
            "03 - RGB Image"
        ),
    )

    print("    Saved         : 03_rgb.png")
    print()

    # ============================================================
    # 4. HISTOGRAM EQUALIZATION
    # ============================================================

    print("-" * 68)
    print("[4] HISTOGRAM EQUALIZATION")
    print("-" * 68)

    equalized_rgb = histogram_equalization_rgb(
        resized_rgb
    )

    equalized_bgr = cv2.cvtColor(
        equalized_rgb,
        cv2.COLOR_RGB2BGR,
    )

    print("    Method        : Y-channel histogram equalization")
    print("    Pixel range   : 0 - 255")
    print("    Data type     : float32 when supplied to model")

    save_image(
        "04_histogram_equalized.png",
        put_title(
            equalized_bgr,
            "04 - Histogram Equalized"
        ),
    )

    print(
        "    Saved         : "
        "04_histogram_equalized.png"
    )
    print()

    # ============================================================
    # 5. CELL SEGMENTATION + ANALYSIS
    # ============================================================

    print("-" * 68)
    print("[5] CELL SEGMENTATION & ANALYSIS")
    print("-" * 68)

    print("    Running OpenCV cell analysis...")

    result = analyze_image(original_bgr)

    summary = result.summary()

    # Segmentation mask
    mask = result.mask

    if mask.dtype != np.uint8:

        mask = cv2.normalize(
            mask,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    save_image(
        "05_segmentation_mask.png",
        put_title(
            mask_bgr,
            "05 - RBC Segmentation Mask"
        ),
    )

    print("    Segmentation completed")
    print(f"    Raw blobs     : {summary['n_blobs']}")
    print(
        f"    Median area   : "
        f"{summary['median_blob_area_px']} px"
    )
    print(
        f"    Rejected      : "
        f"{summary['rejected_blobs']}"
    )

    print(
        "    Saved         : "
        "05_segmentation_mask.png"
    )
    print()

    # ============================================================
    # 6. RBC DETECTION
    # ============================================================

    print("-" * 68)
    print("[6] RBC DETECTION")
    print("-" * 68)

    detected_cells = create_detected_cells_image(
        original_bgr,
        result,
    )

    save_image(
        "06_detected_cells.png",
        put_title(
            detected_cells,
            "06 - Detected RBC Candidates"
        ),
    )

    print(
        f"    Accepted RBCs : "
        f"{summary['total_rbc']}"
    )

    print(
        f"    Avg diameter  : "
        f"{summary['avg_rbc_diameter_px']} px"
    )

    print("    Saved         : 06_detected_cells.png")
    print()

    # ============================================================
    # 7. SICKLE-LIKE CELL DETECTION
    # ============================================================

    print("-" * 68)
    print("[7] SICKLE-LIKE CELL DETECTION")
    print("-" * 68)

    sickle_image = create_sickle_detection_image(
        original_bgr,
        result,
    )

    save_image(
        "07_sickle_detection.png",
        put_title(
            sickle_image,
            "07 - Sickle-like Cell Detection"
        ),
    )

    print(
        f"    Sickle-like   : "
        f"{summary['sickle_count']}"
    )

    print(
        f"    SCI           : "
        f"{summary['sci']}%"
    )

    print("    Saved         : 07_sickle_detection.png")
    print()

    # ============================================================
    # 8. CLUSTERING
    # ============================================================

    print("-" * 68)
    print("[8] CELL CLUSTERING")
    print("-" * 68)

    clustering_image = create_clustering_image(
        original_bgr,
        result,
    )

    save_image(
        "08_clustering.png",
        put_title(
            clustering_image,
            "08 - Sickle Cell Clustering"
        ),
    )

    print(
        f"    Clustered     : "
        f"{summary['clustered_sickle']}"
    )

    print(
        f"    Clusters      : "
        f"{summary['n_clusters']}"
    )

    print(
        f"    Cluster CD    : "
        f"{summary['cd']}%"
    )

    print("    Saved         : 08_clustering.png")
    print()

    # ============================================================
    # 9. MOBILENETV1 + GRAD-CAM
    # ============================================================

    print("-" * 68)
    print("[9] MOBILENETV1 + GRAD-CAM")
    print("-" * 68)

    print("    Loading MobileNetV1 model...")
    print(f"    Model: {MODEL_PATH.name}")

    gradcam = GradCAM(
        str(MODEL_PATH)
    )

    explanation = gradcam.predict_and_explain(
        str(IMAGE_PATH)
    )

    probability = float(
        explanation["probability"]
    )

    label = explanation["label"]

    heatmap = explanation["heatmap"]

    gradcam_overlay = explanation["overlay"]

    # The GradCAM overlay returned by the module is RGB.
    if gradcam_overlay.ndim == 3:

        gradcam_bgr = cv2.cvtColor(
            gradcam_overlay,
            cv2.COLOR_RGB2BGR,
        )

    else:

        gradcam_bgr = gradcam_overlay

    save_image(
        "09_gradcam.png",
        put_title(
            gradcam_bgr,
            "09 - Grad-CAM"
        ),
    )

    print("    Model loaded")
    print("    Feature extraction completed")
    print("    Classification completed")
    print()
    print(
        f"    Probability   : "
        f"{probability:.4f}"
    )

    print(
        f"    Confidence    : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"    Prediction    : "
        f"{label}"
    )

    print("    Grad-CAM generated")
    print("    Saved         : 09_gradcam.png")
    print()

    # ============================================================
    # 10. FINAL RESULT
    # ============================================================

    print("=" * 68)
    print("                         FINAL RESULT")
    print("=" * 68)

    if label.lower() == "positive":

        final_prediction = "Sickle Cell Positive"

    else:

        final_prediction = "Sickle Cell Negative"

    print()
    print(
        f"Prediction       : "
        f"{final_prediction}"
    )

    print(
        f"Confidence       : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"RBC count        : "
        f"{summary['total_rbc']}"
    )

    print(
        f"Sickle-like      : "
        f"{summary['sickle_count']}"
    )

    print(
        f"SCI              : "
        f"{summary['sci']}%"
    )

    print(
        f"Clustered sickle : "
        f"{summary['clustered_sickle']}"
    )

    print(
        f"Cluster density  : "
        f"{summary['cd']}%"
    )

    print(
        f"Clusters         : "
        f"{summary['n_clusters']}"
    )

    print(
        f"Avg RBC diameter : "
        f"{summary['avg_rbc_diameter_px']} px"
    )

    print()
    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )

    print()
    print("=" * 68)
    print("              WALKTHROUGH COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print()

    # ============================================================
    # SAVE TEXT SUMMARY
    # ============================================================

    summary_path = OUTPUT_DIR / "pipeline_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SICKLE CELL DETECTION - PROJECT WALKTHROUGH\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Input image: {IMAGE_PATH}\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n\n"
        )

        f.write("PREPROCESSING\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Original size: "
            f"{original_w} x {original_h}\n"
        )

        f.write(
            "Model input size: 224 x 224\n"
        )

        f.write(
            "Color conversion: BGR -> RGB\n"
        )

        f.write(
            "Histogram equalization: Y channel\n"
        )

        f.write(
            "Model pixel range: 0 - 255\n\n"
        )

        f.write("MODEL PREDICTION\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Model: {MODEL_PATH.name}\n"
        )

        f.write(
            f"Prediction: {final_prediction}\n"
        )

        f.write(
            f"Probability: {probability:.4f}\n"
        )

        f.write(
            f"Confidence: {probability * 100:.2f}%\n\n"
        )

        f.write("CELL ANALYSIS\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Raw blobs: {summary['n_blobs']}\n"
        )

        f.write(
            f"Rejected blobs: "
            f"{summary['rejected_blobs']}\n"
        )

        f.write(
            f"Accepted RBCs: "
            f"{summary['total_rbc']}\n"
        )

        f.write(
            f"Sickle-like cells: "
            f"{summary['sickle_count']}\n"
        )

        f.write(
            f"SCI: {summary['sci']}%\n"
        )

        f.write(
            f"Clustered sickle cells: "
            f"{summary['clustered_sickle']}\n"
        )

        f.write(
            f"Number of clusters: "
            f"{summary['n_clusters']}\n"
        )

        f.write(
            f"Cluster density: "
            f"{summary['cd']}%\n"
        )

        f.write(
            f"Average RBC diameter: "
            f"{summary['avg_rbc_diameter_px']} px\n"
        )

        f.write("\nOUTPUT FILES\n")
        f.write("-" * 60 + "\n")

        for filename in [
            "01_original.png",
            "02_resized.png",
            "03_rgb.png",
            "04_histogram_equalized.png",
            "05_segmentation_mask.png",
            "06_detected_cells.png",
            "07_sickle_detection.png",
            "08_clustering.png",
            "09_gradcam.png",
        ]:

            f.write(f"{filename}\n")

    print(
        f"Summary saved: {summary_path}"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()