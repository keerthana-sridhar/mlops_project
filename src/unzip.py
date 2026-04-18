import zipfile
import os

ZIP_PATH = "/Users/keerthanasridhar/DA5402/project_updated/data/raw/archive.zip"
DEST = "data/raw"

os.makedirs(DEST, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DEST)

print(f"✅ Extracted dataset to {DEST}")