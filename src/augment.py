import os
import shutil
import yaml
from PIL import Image
import torchvision.transforms as transforms


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_transform(aug_params):
    transform_list = []

    if aug_params["horizontal_flip"]:
        transform_list.append(transforms.RandomHorizontalFlip())

    if aug_params["vertical_flip"]:
        transform_list.append(transforms.RandomVerticalFlip())

    if aug_params["rotation"] > 0:
        transform_list.append(transforms.RandomRotation(aug_params["rotation"]))

    transform_list.append(
        transforms.ColorJitter(
            brightness=aug_params["brightness"],
            contrast=aug_params["contrast"]
        )
    )

    return transforms.Compose(transform_list)


def reset_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def create_dirs(output_dir, splits, classes):
    for split in splits:
        for label in classes:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)


def augment_images(input_dir, output_dir, splits, classes, transform, num_augments):
    for split in splits:
        for label in classes:
            input_path = os.path.join(input_dir, split, label)
            output_path = os.path.join(output_dir, split, label)

            for img_name in os.listdir(input_path):
                img_path = os.path.join(input_path, img_name)

                try:
                    img = Image.open(img_path).convert("RGB")

                    # ✅ Always copy original (train, val, test)
                    img.save(os.path.join(output_path, img_name))

                    # 🔥 Only augment TRAIN
                    if split == "train":
                        for i in range(num_augments):
                            aug_img = transform(img)
                            aug_img.save(
                                os.path.join(output_path, f"{img_name}_aug{i}.png")
                            )

                except:
                    continue

def main():
    params = load_params()

    input_dir = params["data"]["resized_dir"]
    output_dir = params["data"]["augmented_dir"]
    classes = params["data"]["classes"]

    aug_params = params["augmentation"]
    num_augments = aug_params["num_augments"]

    splits = ["train", "val", "test"]

    transform = get_transform(aug_params)

    print("Resetting augmented directory...")
    reset_dir(output_dir)

    print("Creating directories...")
    create_dirs(output_dir, splits, classes)

    print("Augmenting images...")
    augment_images(input_dir, output_dir, splits, classes, transform, num_augments)

    print("Done!")


if __name__ == "__main__":
    main()