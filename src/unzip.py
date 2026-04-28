import shutil
import tempfile
import zipfile
from pathlib import Path

# Assisted by Claude
ZIP_PATH = Path("data/raw/raw_zipped.zip")
DEST = Path("data/raw")
CELL_IMAGES_DIR = DEST / "cell_images"
CLASSES = ("Parasitized", "Uninfected")


def reset_output_dir():
    """Clear previous extracted data so repeated runs stay deterministic."""
    if CELL_IMAGES_DIR.exists():
        shutil.rmtree(CELL_IMAGES_DIR)

    for label in CLASSES:
        top_level_dir = DEST / label
        if top_level_dir.exists():
            shutil.rmtree(top_level_dir)


def find_dataset_root(extract_root: Path) -> Path:
    """Support archives with either class folders at the root or inside cell_images/."""
    candidates = [extract_root]

    for child in sorted(extract_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        candidates.append(child)
        nested = child / "cell_images"
        if nested.is_dir():
            candidates.append(nested)

    for candidate in candidates:
        if all((candidate / label).is_dir() for label in CLASSES):
            return candidate

    expected = ", ".join(CLASSES)
    raise FileNotFoundError(
        f"Could not find class folders ({expected}) inside {ZIP_PATH}"
    )


def copy_dataset(dataset_root: Path):
    CELL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for label in CLASSES:
        shutil.copytree(dataset_root / label, CELL_IMAGES_DIR / label)


if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Expected dataset archive at {ZIP_PATH}")

DEST.mkdir(parents=True, exist_ok=True)
reset_output_dir()

with tempfile.TemporaryDirectory(dir=DEST, prefix="extract_") as tmp_dir:
    tmp_path = Path(tmp_dir)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    dataset_root = find_dataset_root(tmp_path)
    copy_dataset(dataset_root)

print(f"Prepared dataset at {CELL_IMAGES_DIR}")
