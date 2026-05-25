from pathlib import Path
import shutil
import random

SOURCE_DIR = Path(r"C:\Users\HP\Downloads\NumberPlate Datset with annotation")

IMAGE_DIR = SOURCE_DIR / "images" / "images"
LABEL_DIR = SOURCE_DIR / "yolo_labels"

TRAIN_SPLIT = 0.8

folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# ONLY LABEL FILES
label_files = list(LABEL_DIR.glob("*.txt"))

pairs = []

for label_file in label_files:

    base_name = label_file.stem

    image_file = None

    # Match exact image
    for ext in [".jpg", ".jpeg", ".png"]:

        candidate = IMAGE_DIR / f"{base_name}{ext}"

        if candidate.exists():
            image_file = candidate
            break

    # ONLY ADD MATCHED PAIRS
    if image_file is not None:
        pairs.append((image_file, label_file))

print("Matched pairs:", len(pairs))

# Shuffle
random.shuffle(pairs)

split_index = int(len(pairs) * TRAIN_SPLIT)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

def copy_pairs(pair_list, split_name):

    for image_file, label_file in pair_list:

        image_dest = Path(f"dataset/images/{split_name}") / image_file.name.replace("ll", "")

        label_dest = Path(f"dataset/labels/{split_name}") / label_file.name.replace("ll", "")

        shutil.copy(image_file, image_dest)

        shutil.copy(label_file, label_dest)

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

print("Dataset organized successfully.")