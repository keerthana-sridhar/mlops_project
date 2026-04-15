import os
import shutil
import yaml
from sklearn.model_selection import train_test_split


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def reset_processed_dir(processed_dir):
    if os.path.exists(processed_dir):
        print("Cleaning existing processed data")
        shutil.rmtree(processed_dir)


def create_dirs(processed_dir, classes):
    for split in ["train", "val", "test"]:
        for label in classes:
            path = os.path.join(processed_dir, split, label)
            os.makedirs(path, exist_ok=True)


def get_image_paths(raw_dir, classes):
    paths = []
    labels = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        for img in os.listdir(folder):
            paths.append(os.path.join(folder, img))
            labels.append(label)

    return paths, labels


def split_data(paths, labels, train_split, val_split, test_split, random_state):
    """Split data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(1 - train_split), stratify=labels, random_state=random_state
    )

    val_size = test_split / (val_split + test_split)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def copy_files(X, y, split, processed_dir):
    """Copy files to their respective processed split directories."""
    for img_path, label in zip(X, y):
        dest = os.path.join(processed_dir, split, label)
        shutil.copy(img_path, dest)


import random

def create_demo_split(processed_dir, classes, demo_per_class=100):
    demo_dir = os.path.join(processed_dir, "demo")
    
    for label in classes:
        test_class_dir = os.path.join(processed_dir, "test", label)
        demo_class_dir = os.path.join(demo_dir, label)

        os.makedirs(demo_class_dir, exist_ok=True)

        images = os.listdir(test_class_dir)

        # Ensure enough images
        if len(images) < demo_per_class:
            raise ValueError(f"Not enough images in test/{label} to create demo set")

        sampled = random.sample(images, demo_per_class)

        for img in sampled:
            src = os.path.join(test_class_dir, img)
            dst = os.path.join(demo_class_dir, img)

            shutil.move(src, dst)  # move so test remains clean

def main():
    # Load parameters
    params = load_params()
    
    # Extract data parameters
    raw_dir = params["data"]["raw_dir"]
    processed_dir = params["data"]["split_dir"]
    classes = params["data"]["classes"]
    
    # Extract split parameters
    train_split = params["split"]["train_ratio"]
    val_split = params["split"]["val_ratio"]
    test_split = params["split"]["test_ratio"]
    random_state = params["split"]["random_state"]

    print("Resetting processed directory...")
    reset_processed_dir(processed_dir)

    print("Creating directories...")
    create_dirs(processed_dir, classes)

    print("Loading data...")
    paths, labels = get_image_paths(raw_dir, classes)

    print("Splitting data...")
    train, val, test = split_data(
        paths, labels, train_split, val_split, test_split, random_state
    )

    print("Copying train data...")
    copy_files(*train, "train", processed_dir)

    print("Copying val data...")
    copy_files(*val, "val", processed_dir)

    print("Copying test data...")
    copy_files(*test, "test", processed_dir)
    print("Creating demo split...")
    create_demo_split(processed_dir, classes, demo_per_class=100)   

    print("Done!")


if __name__ == "__main__":
    main()