"""
verify_cell_analysis.py - visual check of segmentation, sickle flagging and clustering.

For each sample image it saves a 3-panel picture:
    [ original | foreground mask | annotated result ]
    annotated: green = normal RBC, red = isolated sickle-like, magenta = clustered sickle-like

Table columns:
    clust = number of sickle-like cells that sit in a cluster
    nClus = number of clusters (each cluster has >= 2 sickle-like cells)

Usage (from the project root, in PowerShell):
    python scripts\\verify_cell_analysis.py
    python scripts\\verify_cell_analysis.py --cls negative --n 5
    python scripts\\verify_cell_analysis.py --image "path\\to\\one.png"
"""
import argparse
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)          # so `import config` and `from src...` work

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if not os.path.isfile(os.path.join(BASE_DIR, "config.py")):
    sys.exit(
        f"config.py not found in the project root:\n  {BASE_DIR}\n"
        f"Files/folders actually there: {sorted(os.listdir(BASE_DIR))}\n"
        "(Windows can hide extensions - look for names like 'config.py.txt'.)"
    )

import config  # noqa: E402

from src.cell_analysis import analyze_image, draw_overlay  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images(dataset: str, split: str, cls: str, n: int, seed: int):
    folder = os.path.join(BASE_DIR, dataset, split, cls)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    random.Random(seed).shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def three_panel(image, result):
    mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([image, mask_bgr, draw_overlay(image, result)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=config.DATASET_NAME,
                        help="dataset folder name inside the project root (default: config.DATASET_NAME)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cls", default="positive", choices=["positive", "negative"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", default=None, help="analyse one specific image instead")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "outputs", "cell_analysis_check"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = ([args.image] if args.image
             else collect_images(args.dataset, args.split, args.cls, args.n, args.seed))

    header = (f"{'image':<16}{'size':>11}{'blobs':>7}{'medArea':>9}{'RBC':>5}{'sickle':>8}"
              f"{'SCI%':>7}{'clust':>7}{'nClus':>7}{'CD%':>7}{'diam':>7}{'rej':>6}")
    print(header)
    print("-" * len(header))
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"{os.path.basename(path):<16}  could not read")
            continue
        result = analyze_image(image)
        s = result.summary()
        print(f"{os.path.basename(path)[:15]:<16}{s['image_size']:>11}{s['n_blobs']:>7}"
              f"{s['median_blob_area_px']:>9.0f}{s['total_rbc']:>5}{s['sickle_count']:>8}"
              f"{s['sci']:>7.1f}{s['clustered_sickle']:>7}{s['n_clusters']:>7}{s['cd']:>7.1f}"
              f"{s['avg_rbc_diameter_px']:>7.1f}{s['rejected_blobs']:>6}")
        cv2.imwrite(os.path.join(args.out, "check_" + os.path.basename(path)),
                    three_panel(image, result))

    print(f"\nAnnotated images saved to: {args.out}")


if __name__ == "__main__":
    main()