import os
import sys

# Ensure src/ is importable
sys.path.append(os.path.join(os.path.dirname(__file__)))

from gradcam import GradCAM

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "paper_style_mobilenetv1_best.keras"
)

# CHANGE this to an actual image path you pulled locally
TEST_IMAGE =  r"E:\Major project\Sickle Cell Detection\dataset_rebalanced\test\positive\1129.png"

print("Loading model from:", MODEL_PATH)
gc = GradCAM(MODEL_PATH)
print("Model loaded successfully.")

print("\nRunning prediction + Grad-CAM on:", TEST_IMAGE)
result = gc.predict_and_explain(TEST_IMAGE)

print("\nProbability:", result["probability"])
print("Label:", result["label"])
print("Heatmap shape:", result["heatmap"].shape)
print("Overlay shape:", result["overlay"].shape)

# Save the overlay so you can visually confirm it looks right
import os
import cv2
import numpy as np

# ------------------------------------------------
# Save Grad-CAM overlay
# ------------------------------------------------

output_dir = os.path.join(BASE_DIR, "test_samples")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.abspath(
    os.path.join(output_dir, "smoke_test_overlay_positive.png")
)

print("\nSaving overlay...")
print("Output directory:", output_dir)
print("Output path:", output_path)

# Make sure overlay is a valid NumPy image
overlay = np.asarray(result["overlay"])

print("Overlay dtype:", overlay.dtype)
print("Overlay shape:", overlay.shape)
print("Overlay min:", overlay.min())
print("Overlay max:", overlay.max())

# Convert RGB -> BGR
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# Save
success = cv2.imwrite(output_path, overlay_bgr)

print("cv2.imwrite result:", success)

if success and os.path.exists(output_path):
    print("\nSUCCESS!")
    print("Overlay exists at:")
    print(output_path)
    print("File size:", os.path.getsize(output_path), "bytes")
else:
    print("\nERROR: Overlay was not written.")