"""
cell_analysis.py - OpenCV cell counting, sickle-like shape flagging, clustering.

Produces the two image-derived measurements the severity logic needs:
    SCI = (sickle cells / total RBCs) * 100
    CD  = (sickle cells that sit in a cluster / total sickle cells) * 100

Pipeline:
    1. Segment RBCs      (green channel -> Otsu -> fill holes -> watershed split)
    2. Filter blobs      (drop debris, WBCs, unsplit clumps, border-cut cells)
    3. Flag sickle-like  (elongated + non-circular; thresholds live in config.py)
    4. Cluster           (touching OR centroids closer than 1 avg RBC diameter)

External libraries: opencv-python, scipy (both flagged in requirements.txt).
All tunable numbers come from config.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import config


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
@dataclass
class CellRecord:
    label: int
    centroid: Tuple[float, float]        # (x, y)
    bbox: Tuple[int, int, int, int]      # (y0, y1, x0, x1)
    area: float
    aspect_ratio: float
    circularity: float
    solidity: float
    touches_border: bool = False
    is_sickle: bool = False
    cluster_id: int = -1                 # -1 = not in a cluster
    contour: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class CellAnalysisResult:
    total_rbc: int
    sickle_count: int
    sci: float                            # 0-100
    clustered_sickle: int
    cd: float                             # 0-100
    n_clusters: int
    avg_rbc_diameter_px: float
    rejected_blobs: int
    cells: List[CellRecord] = field(default_factory=list, repr=False)
    labels: Optional[np.ndarray] = field(default=None, repr=False)
    # --- diagnostics (help tune thresholds; not used by the scoring logic) ---
    image_shape: Tuple[int, int] = (0, 0)          # (height, width)
    n_blobs: int = 0                               # blobs found before any filtering
    median_blob_area_px: float = 0.0               # reference size used for the area filter
    mask: Optional[np.ndarray] = field(default=None, repr=False)   # filled foreground mask

    def summary(self) -> dict:
        """JSON-serialisable summary (no arrays) - this is what Flask returns."""
        return {
            "total_rbc": self.total_rbc,
            "sickle_count": self.sickle_count,
            "sci": round(self.sci, 2),
            "clustered_sickle": self.clustered_sickle,
            "cd": round(self.cd, 2),
            "n_clusters": self.n_clusters,
            "avg_rbc_diameter_px": round(self.avg_rbc_diameter_px, 2),
            "rejected_blobs": self.rejected_blobs,
            "image_size": f"{self.image_shape[1]}x{self.image_shape[0]}",
            "n_blobs": self.n_blobs,
            "median_blob_area_px": round(self.median_blob_area_px, 1),
        }


# ---------------------------------------------------------------------
# Step 1: segmentation
# ---------------------------------------------------------------------
def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _segment(image_bgr: np.ndarray):
    """Return (labels, filled_mask). labels: int32, 0 = background/boundary, 1..N = blobs."""
    k = config.SEG_BLUR_KSIZE
    if k % 2 == 0:
        raise ValueError("config.SEG_BLUR_KSIZE must be odd")

    # Stained RBCs absorb green light, so they are darkest in the green channel
    green = image_bgr[:, :, 1]
    blur = cv2.GaussianBlur(green, (k, k), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    m = config.SEG_MORPH_KSIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill central-pallor holes so each RBC becomes a solid blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Watershed seeds: thresholded PER BLOB (not globally), so thin sickle cells
    # keep a seed while touching cells inside a clump still get separated.
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    seeds = np.zeros_like(filled)
    for i in range(1, n_cc):
        x, y, w, h, area = stats[i]
        if area < config.ABS_MIN_BLOB_AREA_PX:
            continue
        in_blob = cc[y:y + h, x:x + w] == i
        sub_dist = dist[y:y + h, x:x + w]
        dmax = sub_dist[in_blob].max()
        seeds[y:y + h, x:x + w][in_blob & (sub_dist >= config.SEG_SEED_RATIO * dmax)] = 255

    _, markers = cv2.connectedComponents(seeds)
    markers = markers.astype(np.int32) + 1          # background seed = 1, cells >= 2
    sure_bg = cv2.dilate(filled, kernel, iterations=2)
    markers[(sure_bg == 255) & (seeds == 0)] = 0    # unknown region for watershed to decide
    markers = cv2.watershed(image_bgr, markers)     # -1 = boundary lines between cells

    labels = np.where((markers >= 2) & (filled == 255), markers - 1, 0)
    return labels.astype(np.int32), filled


# ---------------------------------------------------------------------
# Step 2: per-cell features
# ---------------------------------------------------------------------
def _extract_cells(labels: np.ndarray) -> List[CellRecord]:
    height, width = labels.shape
    records: List[CellRecord] = []

    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop

        sub = (labels[sl] == lab).astype(np.uint8)
        area = float(sub.sum())
        if area < 5:
            continue

        padded = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(
            padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x0 - 1, y0 - 1)
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if cnt_area <= 0 or perimeter <= 0:
            continue

        circularity = min(1.0, 4.0 * math.pi * cnt_area / (perimeter ** 2))
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = min(1.0, cnt_area / hull_area) if hull_area > 0 else 1.0

        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        long_side, short_side = max(rw, rh), min(rw, rh)
        aspect = long_side / short_side if short_side >= 1.0 else 1.0

        moments = cv2.moments(sub, binaryImage=True)
        cx = moments["m10"] / moments["m00"] + x0
        cy = moments["m01"] / moments["m00"] + y0

        records.append(CellRecord(
            label=lab,
            centroid=(cx, cy),
            bbox=(y0, y1, x0, x1),
            area=area,
            aspect_ratio=aspect,
            circularity=circularity,
            solidity=solidity,
            touches_border=(y0 <= 5 or x0 <= 5 or y1 >= height - 5 or x1 >= width - 5),
            contour=cnt,
        ))
    return records


# ---------------------------------------------------------------------
# Step 4: clustering
# ---------------------------------------------------------------------
def _touching_pairs(sickle: List[CellRecord], labels: np.ndarray) -> set:
    """Pairs of sickle cells whose masks touch (within TOUCH_TOLERANCE_PX)."""
    tol = config.TOUCH_TOLERANCE_PX
    height, width = labels.shape
    size = 2 * tol + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    index_of = {c.label: i for i, c in enumerate(sickle)}
    edges = set()

    for i, c in enumerate(sickle):
        y0, y1, x0, x1 = c.bbox
        crop = labels[max(0, y0 - tol):min(height, y1 + tol),
                      max(0, x0 - tol):min(width, x1 + tol)]
        grown = cv2.dilate((crop == c.label).astype(np.uint8), kernel)
        neighbours = np.unique(crop[(grown > 0) & (crop > 0) & (crop != c.label)])
        for lab in neighbours:
            j = index_of.get(int(lab))
            if j is not None:
                edges.add((min(i, j), max(i, j)))
    return edges


def _contour_distance(cnt1: np.ndarray, cnt2: np.ndarray) -> float:
    """Minimum Euclidean distance between two cell contours (edge-to-edge)."""
    pts1 = cnt1.reshape(-1, 2)
    pts2 = cnt2.reshape(-1, 2)
    step1 = max(1, len(pts1) // 50)
    step2 = max(1, len(pts2) // 50)
    return float(np.min(cdist(pts1[::step1], pts2[::step2])))


def _assign_clusters(sickle: List[CellRecord], labels: np.ndarray,
                     avg_diameter: float) -> int:
    """Set cluster_id on each sickle cell using edge-to-edge contour distance; return number of clusters."""
    n = len(sickle)
    if n < 2:
        return 0

    max_edge_gap = config.CLUSTER_DISTANCE_FACTOR * avg_diameter
    edges = set()

    for i in range(n):
        for j in range(i + 1, n):
            # Fast bounding box pre-filter: if bounding boxes are farther than max_edge_gap, skip
            y0_i, y1_i, x0_i, x1_i = sickle[i].bbox
            y0_j, y1_j, x0_j, x1_j = sickle[j].bbox
            dx = max(0, max(x0_i - x1_j, x0_j - x1_i))
            dy = max(0, max(y0_i - y1_j, y0_j - y1_i))
            if dx > max_edge_gap or dy > max_edge_gap:
                continue

            # Compute true minimum edge distance between contours
            if sickle[i].contour is not None and sickle[j].contour is not None:
                dist = _contour_distance(sickle[i].contour, sickle[j].contour)
                if dist <= max_edge_gap:
                    edges.add((i, j))

    edges |= _touching_pairs(sickle, labels)  # touching pixels rule
    if not edges:
        return 0

    rows, cols = zip(*edges)
    graph = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    _, component = connected_components(graph, directed=False)

    sizes = np.bincount(component)
    cluster_components = [c for c in range(len(sizes)) if sizes[c] >= 2]
    remap = {comp: cid for cid, comp in enumerate(cluster_components)}
    for i, cell in enumerate(sickle):
        cell.cluster_id = remap.get(int(component[i]), -1)
    return len(cluster_components)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def analyze_image(image: np.ndarray) -> CellAnalysisResult:
    """Run the full cell analysis on a BGR uint8 image (as loaded by cv2.imread)."""
    image_bgr = _to_bgr_uint8(image)
    labels, mask = _segment(image_bgr)
    candidates = _extract_cells(labels)
    n_blobs = len(candidates)

    diag = dict(image_shape=image_bgr.shape[:2], n_blobs=n_blobs, mask=mask, labels=labels)

    if config.EXCLUDE_BORDER_CELLS:
        candidates = [c for c in candidates if not c.touches_border]

    # Drop specks BEFORE estimating the typical cell size. Otherwise, in images with lots
    # of debris/stain fragments the median collapses and real cells get rejected as
    # "too large". The floor scales with image area (config.MIN_BLOB_AREA_FRAC).
    speck_floor = max(
        config.ABS_MIN_BLOB_AREA_PX,
        getattr(config, "MIN_BLOB_AREA_FRAC", 0.0003) * image_bgr.shape[0] * image_bgr.shape[1],
    )
    candidates = [c for c in candidates if c.area >= speck_floor]

    if not candidates:
        return CellAnalysisResult(0, 0, 0.0, 0, 0.0, 0, 0.0, n_blobs, [], **diag)

    # Keep RBC-sized blobs only (relative to this image's median blob area)
    median_area = float(np.median([c.area for c in candidates]))
    diag["median_blob_area_px"] = median_area
    lo = config.MIN_CELL_AREA_FACTOR * median_area
    hi = config.MAX_CELL_AREA_FACTOR * median_area
    cells = [c for c in candidates if lo <= c.area <= hi]
    rejected = n_blobs - len(cells)

    if not cells:
        return CellAnalysisResult(0, 0, 0.0, 0, 0.0, 0, 0.0, rejected, [], **diag)

    avg_diameter = float(np.mean([math.sqrt(4.0 * c.area / math.pi) for c in cells]))

    for c in cells:
        c.is_sickle = (c.aspect_ratio >= config.SICKLE_MIN_ASPECT_RATIO
                       and c.circularity <= config.SICKLE_MAX_CIRCULARITY)

    sickle = [c for c in cells if c.is_sickle]
    n_clusters = _assign_clusters(sickle, labels, avg_diameter)
    clustered = sum(1 for c in sickle if c.cluster_id >= 0)

    total = len(cells)
    sci = (len(sickle) / total) * 100.0
    cd = (clustered / len(sickle)) * 100.0 if sickle else 0.0

    return CellAnalysisResult(
        total_rbc=total,
        sickle_count=len(sickle),
        sci=sci,
        clustered_sickle=clustered,
        cd=cd,
        n_clusters=n_clusters,
        avg_rbc_diameter_px=avg_diameter,
        rejected_blobs=rejected,
        cells=cells,
        **diag,
    )


def analyze_path(path: str) -> CellAnalysisResult:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return analyze_image(image)


def draw_overlay(image: np.ndarray, result: CellAnalysisResult) -> np.ndarray:
    """Annotated copy: green = normal, red = isolated sickle-like, magenta = clustered sickle-like."""
    out = _to_bgr_uint8(image).copy()
    for c in result.cells:
        if not c.is_sickle:
            colour = (0, 200, 0)
        elif c.cluster_id >= 0:
            colour = (255, 0, 255)
        else:
            colour = (0, 0, 255)
        cv2.drawContours(out, [c.contour], -1, colour, 2 if c.is_sickle else 1)
        if c.cluster_id >= 0:
            cv2.putText(out, str(c.cluster_id), (int(c.centroid[0]), int(c.centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    label = (f"RBC={result.total_rbc} sickle={result.sickle_count} "
             f"SCI={result.sci:.1f}% CD={result.cd:.1f}%")
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out
