from pathlib import Path

import yaml

from number_plate_detector import DEFAULT_MODEL_CANDIDATES, project_root


def count_files(path: Path, patterns):
    total = 0
    for pattern in patterns:
        total += len(list(path.glob(pattern)))
    return total


def main():
    root = project_root()
    data_yaml = root / "data.yaml"
    print(f"Project root: {root}")

    if data_yaml.exists():
        with data_yaml.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        dataset_root = Path(data.get("path", root / "dataset"))
        if not dataset_root.is_absolute():
            dataset_root = root / dataset_root
    else:
        dataset_root = root / "dataset"
        print("MISSING: data.yaml")

    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    train_images = count_files(dataset_root / "images" / "train", image_patterns)
    val_images = count_files(dataset_root / "images" / "val", image_patterns)
    train_labels = count_files(dataset_root / "labels" / "train", ["*.txt"])
    val_labels = count_files(dataset_root / "labels" / "val", ["*.txt"])

    print(f"Train images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Train labels: {train_labels}")
    print(f"Validation labels: {val_labels}")

    model_paths = [root / path for path in DEFAULT_MODEL_CANDIDATES]
    existing_models = [path for path in model_paths if path.exists() and path.stat().st_size > 0]
    if existing_models:
        print("YOLO weights found:")
        for path in existing_models:
            print(f"  - {path}")
    else:
        print("MISSING: trained YOLO best.pt weights")

    if train_labels == 0 or val_labels == 0:
        print("ACTION: Add YOLO annotation .txt files before running train.py.")
    if not existing_models:
        print("ACTION: Train the model, then run the Streamlit app with the generated best.pt.")


if __name__ == "__main__":
    main()
