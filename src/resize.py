import os
import yaml
from PIL import Image
import shutil


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def reset_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def create_dirs(output_dir, splits, classes):
    for split in splits:
        for label in classes:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)


def resize_images(input_dir, output_dir, splits, classes, img_size):
    for split in splits:
        for label in classes:
            input_path = os.path.join(input_dir, split, label)
            output_path = os.path.join(output_dir, split, label)

            for img_name in os.listdir(input_path):
                try:
                    img = Image.open(os.path.join(input_path, img_name)).convert("RGB")
                    img = img.resize(tuple(img_size))
                    img.save(os.path.join(output_path, img_name))
                except:
                    continue


def main():
    params = load_params()

    input_dir = params["data"]["split_dir"]
    output_dir = params["data"]["resized_dir"]
    classes = params["data"]["classes"]
    img_size = params["preprocess"]["img_size"]
    splits = ["train", "val", "test"]

    print("Resetting resized directory...")
    reset_dir(output_dir)

    print("Creating directories...")
    create_dirs(output_dir, splits, classes)

    print("Resizing images...")
    resize_images(input_dir, output_dir, splits, classes, img_size)

    print("Done!")


if __name__ == "__main__":
    main()