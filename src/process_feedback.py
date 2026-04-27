import os
import shutil
from PIL import Image
# Assisted by ChatGPT
SRC = "data/feedback/labelled"
DEST = "data/processed/incremental_resized"
SIZE = (224, 224)

os.makedirs(DEST, exist_ok=True)

count = 0

for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)
    dest_cls = os.path.join(DEST, cls)

    os.makedirs(dest_cls, exist_ok=True)

    for f in os.listdir(src_cls):
        src_file = os.path.join(src_cls, f)
        dest_file = os.path.join(dest_cls, f)

        try:
            img = Image.open(src_file).convert("RGB")
            img = img.resize(SIZE)
            img.save(dest_file)
            count += 1
        except:
            continue

print(f"✅ Processed {count} new images")

# clear labelled after processing
shutil.rmtree(SRC)
os.makedirs(SRC, exist_ok=True)