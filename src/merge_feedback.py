import os
import shutil

SRC = "data/feedback/labelled"
DEST = "data/raw/cell_images"

if not os.path.exists(SRC):
    print("No feedback data")
    exit(0)

total_copied = 0

import uuid

for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)
    dest_cls = os.path.join(DEST, cls)

    if not os.path.isdir(src_cls):
        continue

    os.makedirs(dest_cls, exist_ok=True)

    for f in os.listdir(src_cls):
        src_file = os.path.join(src_cls, f)

        # 🔥 always create a unique filename
        unique_name = f"{uuid.uuid4()}_{f}"
        dest_file = os.path.join(dest_cls, unique_name)

        shutil.copy(src_file, dest_file)
        total_copied += 1

print(f"✅ Copied {total_copied} new images")

# 🔥 ONLY delete if something was copied
if total_copied > 0:
    

    for cls in os.listdir(SRC):
        cls_path = os.path.join(SRC, cls)
        if os.path.isdir(cls_path):
            shutil.rmtree(cls_path)

    print("🗑️ Cleared labelled contents (folder preserved)")

else:
    print("⚠️ Nothing new copied — not deleting labelled")