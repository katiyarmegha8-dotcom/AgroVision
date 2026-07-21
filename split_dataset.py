import os
import random
import shutil
from pathlib import Path

# Change this path according to your computer
SOURCE_DIR = r"C:\Users\Megha Katiyar\Downloads\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color"

# This creates the dataset folder inside your project
DEST_DIR = "dataset"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

classes = [d for d in os.listdir(SOURCE_DIR)
           if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    print(f"Processing: {cls}")

    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }

    for split_name, image_list in splits.items():
        split_class_dir = os.path.join(DEST_DIR, split_name, cls)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in image_list:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(split_class_dir, img)
            )

print("\nDataset successfully split!")