r"""
config.py - centralized settings for the Sickle Cell Detection project.

Lives in the project root:  E:\Major project\Sickle Cell Detection\config.py
Currently holds the settings for src/cell_analysis.py and src/severity.py.
Add training settings (image size, batch size, etc.) here later if you want them centralized.
"""
import os

# Project root = the folder this file sits in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "dataset_rebalanced"      # folder name inside the project root
DATASET_DIR = os.path.join(BASE_DIR, DATASET_NAME)
MODELS_DIR = os.path.join(BASE_DIR, "models")




# ---------------------------------------------------------------------
# Cell analysis (src/cell_analysis.py)
# !! Every value here is a STARTING GUESS, not a validated setting.
# !! Tune them by running scripts/verify_cell_analysis.py on real images.
# ---------------------------------------------------------------------

# Segmentation
SEG_BLUR_KSIZE = 5             # Gaussian blur kernel (must be odd)
SEG_MORPH_KSIZE = 3            # morphological opening kernel
SEG_SEED_RATIO = 0.5           # watershed seeds = pixels >= this fraction of a blob's max distance
ABS_MIN_BLOB_AREA_PX = 40      # ignore specks smaller than this before anything else

# Which blobs count as RBCs (relative to the median blob area of the image,
# so it adapts to different magnifications / image sizes)
MIN_CELL_AREA_FACTOR = 0.25    # smaller than this x median  -> platelet / debris
MAX_CELL_AREA_FACTOR = 1.8     # larger than this x median   -> WBC or unsplit clump
EXCLUDE_BORDER_CELLS = True    # cells cut by the image edge have unreliable shapes

# Sickle-like shape rule: elongated AND not round
SICKLE_MIN_ASPECT_RATIO = 2.0  # long side / short side of the min-area rectangle
SICKLE_MAX_CIRCULARITY = 0.65  # 4*pi*area / perimeter^2  (1.0 = perfect circle)

# Clustering: two sickle cells share a cluster if they touch OR their centroids
# are closer than CLUSTER_DISTANCE_FACTOR x the average RBC diameter
CLUSTER_DISTANCE_FACTOR = 1.1
TOUCH_TOLERANCE_PX = 2         # gap (pixels) still counted as "touching"


# ---------------------------------------------------------------------
# Severity: IBSI = W_SCI * SCI + W_CD * CD        (src/severity.py)
# ---------------------------------------------------------------------
IBSI_W_SCI = 0.7
IBSI_W_CD = 0.3

# Lower bounds of each band (half-open, so no gaps between e.g. 33 and 34):
#   Mild: < 34 | Moderate: 34 to < 67 | Severe: >= 67
SEVERITY_MODERATE_MIN = 34
SEVERITY_SEVERE_MIN = 67


# ---------------------------------------------------------------------
# Age groups and Hb lookup (g/dL) - from your doc
# ---------------------------------------------------------------------
AGE_GROUPS = ("child", "adolescent", "adult", "older_adult")
# child 0-12 | adolescent 13-18 | adult 19-40 | older_adult >40

HB_RANGES = {
    "Mild": {
        "child": (9.5, 11.5), "adolescent": (9.5, 11.5),
        "adult": (9.0, 11.0), "older_adult": (9.0, 11.0),
    },
    "Moderate": {
        "child": (7.5, 9.4), "adolescent": (7.5, 9.4),
        "adult": (7.0, 8.9), "older_adult": (7.0, 8.9),
    },
    "Severe": {
        "child": (5.5, 7.4), "adolescent": (5.5, 7.4),
        "adult": (5.0, 6.9), "older_adult": (5.0, 6.9),
    },
}


# ---------------------------------------------------------------------
# Vaso-occlusion risk score (VORS)
# ---------------------------------------------------------------------
# Maximum base value for pain crises per year, by age group and gender
PAIN_CRISIS_MAX_BASE = {
    "child":       {"male": 6,  "female": 5},
    "adolescent":  {"male": 10, "female": 8},
    "adult":       {"male": 12, "female": 9},
    "older_adult": {"male": 4,  "female": 4},
}

SPO2_FLOOR = 88                # SpO2 score = (100 - SpO2) / (100 - SPO2_FLOOR) * 100

VORS_WEIGHTS = {"spo2": 0.40, "ibsi": 0.30, "hb": 0.20, "pain": 0.10}

# Same half-open band convention as severity: Low < 34 | Moderate 34 to < 67 | High >= 67
VORS_MODERATE_MIN = 34
VORS_HIGH_MIN = 67

# Guard against typos when editing weights
assert abs((IBSI_W_SCI + IBSI_W_CD) - 1.0) < 1e-9, "IBSI weights must sum to 1"
assert abs(sum(VORS_WEIGHTS.values()) - 1.0) < 1e-9, "VORS weights must sum to 1"