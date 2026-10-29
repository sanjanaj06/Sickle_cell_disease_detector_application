"""
rebuild_dataset_split.py

Pools the existing train/val/test folders back into one set, detects
near-duplicate images (to catch same-smear leakage now that patient IDs
aren't recoverable), and produces a fresh stratified 70/15/15 split.

Writes the new split into a SEPARATE output folder (does not touch or
overwrite your existing dataset/) so you can compare before/after safely.

External library used for the first time here: `imagehash`
    pip install imagehash
(also needs Pillow, which you already have)
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash

# -----------------------------------------------------------------------
# CONFIG — edit these paths to match your machine
# -----------------------------------------------------------------------
SOURCE_DATASET_DIR = Path(r"E:\Major project\Sickle Cell Detection\dataset")
OUTPUT_DATASET_DIR = Path(r"E:\Major project\Sickle Cell Detection\dataset_rebalanced")
SPLITS = ["train", "val", "test"]
CLASSES = ["positive", "negative"]

HASH_THRESHOLD = 5     # max hamming distance to count as "near-duplicate"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


# -----------------------------------------------------------------------
# STEP 1 — Pool every image from the existing split folders
# -----------------------------------------------------------------------
def scan_dataset(root: Path):
    """
    Walks train/val/test x positive/negative and returns a flat list of
    dicts: {"path": Path, "label": "positive"/"negative"}.
    We deliberately ignore which split each image currently sits in —
    that information is exactly what we're throwing out and rebuilding.
    """
    items = []
    for split in SPLITS:
        for label in CLASSES:
            folder = root / split / label
            if not folder.exists():
                print(f"  [warn] missing folder: {folder}")
                continue
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    items.append({"path": img_path, "label": label})
    print(f"Pooled {len(items)} images "
          f"({sum(i['label']=='positive' for i in items)} positive, "
          f"{sum(i['label']=='negative' for i in items)} negative)")
    return items


# -----------------------------------------------------------------------
# STEP 2 — Compute a perceptual hash for every image
# -----------------------------------------------------------------------
def compute_hashes(items):
    """
    imagehash.phash() produces a 64-bit perceptual hash: visually similar
    images (even after minor crop/brightness differences) end up with
    hashes that differ by only a few bits. This is our stand-in for
    "these two images probably came from the same smear", since we no
    longer have patient/smear IDs.
    """
    for item in items:
        with Image.open(item["path"]) as img:
            item["hash"] = imagehash.phash(img)
    print("Computed perceptual hashes for all images.")
    return items


# -----------------------------------------------------------------------
# STEP 3 — Cluster near-duplicates using union-find
# -----------------------------------------------------------------------
class UnionFind:
    """Standard union-find so we can merge images into duplicate clusters
    without an expensive full pairwise comparison being repeated."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def cluster_duplicates(items, threshold=HASH_THRESHOLD):
    """
    Compares every pair of images' hashes (fine at ~2000 images; would
    need a faster approach like LSH at much larger scale) and unions any
    pair within `threshold` hamming distance. Returns a cluster_id (int)
    assigned to every item, so items in the same cluster share an id.
    """
    n = len(items)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if items[i]["hash"] - items[j]["hash"] <= threshold:
                uf.union(i, j)

    for i, item in enumerate(items):
        item["cluster_id"] = uf.find(i)

    n_clusters = len(set(item["cluster_id"] for item in items))
    n_dupe_clusters = n - n_clusters  # images "absorbed" into a larger cluster
    print(f"Found {n_clusters} clusters from {n} images "
          f"({n - n_clusters} images grouped as near-duplicates of another).")
    return items


# -----------------------------------------------------------------------
# STEP 4 — Greedy stratified, group-aware split
# -----------------------------------------------------------------------
def greedy_split(items, ratios=SPLIT_RATIOS, seed=RANDOM_SEED):
    """
    Groups items by cluster_id (so near-duplicates always land in the
    SAME split — never split across train/val/test), then greedily
    assigns whole clusters to whichever split is furthest below its
    target class count. This keeps both the split-size ratios AND the
    positive/negative class ratio close to your original ~44/56 split.
    """
    clusters = defaultdict(list)
    for item in items:
        clusters[item["cluster_id"]].append(item)

    cluster_list = list(clusters.values())
    random.Random(seed).shuffle(cluster_list)

    total_pos = sum(i["label"] == "positive" for i in items)
    total_neg = sum(i["label"] == "negative" for i in items)

    targets = {
        split: {
            "positive": total_pos * ratio,
            "negative": total_neg * ratio,
        }
        for split, ratio in ratios.items()
    }
    current = {split: {"positive": 0, "negative": 0} for split in ratios}
    assignment = {}  # cluster items' paths -> split name

    for cluster in cluster_list:
        pos_count = sum(i["label"] == "positive" for i in cluster)
        neg_count = sum(i["label"] == "negative" for i in cluster)

        # Score each split by how much IT SPECIFICALLY still needs the
        # class(es) present in this cluster, weighted by how many of each
        # class the cluster contains. A pure-positive cluster is scored
        # only on positive deficit, a pure-negative cluster only on
        # negative deficit — they no longer get blended together, which
        # was causing images to be routed to the wrong split.
        def score(split):
            pos_deficit = targets[split]["positive"] - current[split]["positive"]
            neg_deficit = targets[split]["negative"] - current[split]["negative"]
            return pos_count * pos_deficit + neg_count * neg_deficit

        best_split = max(ratios.keys(), key=score)
        current[best_split]["positive"] += pos_count
        current[best_split]["negative"] += neg_count
        for item in cluster:
            assignment[item["path"]] = best_split

    for split in ratios:
        p, n = current[split]["positive"], current[split]["negative"]
        total = p + n
        pct = (p / total * 100) if total else 0
        print(f"  {split:5s}: {total:4d} images  ({p} pos / {n} neg, {pct:.1f}% positive)")

    return assignment


# -----------------------------------------------------------------------
# STEP 5 — Rebuild the folder structure with the new assignment
# -----------------------------------------------------------------------
def rebuild_folders(items, assignment, output_root=OUTPUT_DATASET_DIR):
    """Copies (not moves) every image into its new split/label folder.
    Clears any previous output first, so re-running the script doesn't
    leave stale images behind in a folder they were previously (but are
    no longer) assigned to."""
    if output_root.exists():
        print(f"  Clearing previous output at {output_root} ...")
        shutil.rmtree(output_root)

    for item in items:
        split = assignment[item["path"]]
        dest_dir = output_root / split / item["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], dest_dir / item["path"].name)
    print(f"New dataset written to: {output_root.resolve()}")


# -----------------------------------------------------------------------
def main():
    print("Step 1: pooling images...")
    items = scan_dataset(SOURCE_DATASET_DIR)

    print("\nStep 2: computing perceptual hashes...")
    items = compute_hashes(items)

    print("\nStep 3: clustering near-duplicates...")
    items = cluster_duplicates(items)

    print("\nStep 4: building new stratified split...")
    assignment = greedy_split(items)

    print("\nStep 5: writing new dataset folders...")
    rebuild_folders(items, assignment)

    print("\nDone. Original dataset/ was not modified.")
    print("Next: point verify_dataset.py at dataset_rebalanced/ to sanity-check it.")


if __name__ == "__main__":
    main()