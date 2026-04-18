import os
import shutil
import random

# Paths
SRC_BASE = "data/processed/split/demo"
DEST_BASE = "data/feedback/labelled"

CLASSES = ["Parasitized", "Uninfected"]
NUM_SAMPLES = 10

for cls in CLASSES:
    src_dir = os.path.join(SRC_BASE, cls)
    dest_dir = os.path.join(DEST_BASE, cls)

    os.makedirs(dest_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(images) < NUM_SAMPLES:
        raise ValueError(f"Not enough images in {cls}")

    selected = random.sample(images, NUM_SAMPLES)

    for img in selected:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(dest_dir, img)
        )

    print(f"✅ Copied {NUM_SAMPLES} images for {cls}")

print("\n🎉 Labelled dataset created at data/feedback/labelled/")