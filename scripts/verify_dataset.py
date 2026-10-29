from pathlib import Path
SPLIT_NAMES = ["train", "val", "test"]
CLASS_NAMES = ["positive", "negative"]

# Changed from "dataset" to "dataset_rebalanced" to check the new split
# built by rebuild_dataset_split.py
DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset_rebalanced"

def count_images_in_folder(folder_path):
    if not folder_path.exists():
        return 0
    image_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
    return len(image_files)

def check_folder_structure(dataset_root):
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            folder_path = dataset_root / split / class_name
            if not folder_path.exists():
                print(f"Missing folder: {folder_path}")
                return False
    return True
def print_image_counts(dataset_root):
    print(f"{'Split':<10}{'Positive':12}{'Negative':12}{'Total':<10}")
    for split in SPLIT_NAMES:
        pos_folder = dataset_root / split / "positive"
        neg_folder = dataset_root / split / "negative"
        pos_count = count_images_in_folder(pos_folder)
        neg_count = count_images_in_folder(neg_folder)
        total_count = pos_count + neg_count
        print(f"{split:<10}{pos_count:<12}{neg_count:<12}{total_count:<10}")

def main():
    dataset_root = DATASET_ROOT
    print(f"Checking folder structure in: {dataset_root}")
    structure_ok = check_folder_structure(dataset_root)
    if structure_ok:
        print("Folder structure is correct.")
    else:
        print("Some folders are missing. Please check the dataset structure.")
    print("\nCounting images")
    print_image_counts(dataset_root)
if __name__ == "__main__":
    main()