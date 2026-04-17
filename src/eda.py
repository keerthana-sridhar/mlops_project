import os
import yaml
import json
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def count_images(raw_dir, classes):
    counts = {}
    for label in classes:
        path = os.path.join(raw_dir, label)
        counts[label] = len(os.listdir(path))
    return counts


def image_size_stats(raw_dir, classes):
    sizes = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        for f in os.listdir(folder)[:200]:  # small cap, no param
            try:
                with Image.open(os.path.join(folder, f)) as img:
                    sizes.append(img.size)
            except:
                continue

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    return {
        "avg_width": float(np.mean(widths)),
        "avg_height": float(np.mean(heights)),
        "min_width": int(np.min(widths)),
        "max_width": int(np.max(widths)),
    }


def check_corrupt(raw_dir, classes):
    corrupt = 0

    for label in classes:
        folder = os.path.join(raw_dir, label)

        for f in os.listdir(folder):
            try:
                Image.open(os.path.join(folder, f)).verify()
            except:
                corrupt += 1

    return corrupt




def plot_class_distribution(counts, output_path):
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure()
    plt.bar(labels, values)
    plt.title("Class Distribution")
    plt.savefig(output_path)
    plt.close()


def plot_sample_images(raw_dir, classes, output_path):
    plt.figure(figsize=(10, 5))

    images = []
    labels = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        files = os.listdir(folder)
        sampled = random.sample(files, min(3, len(files)))

        for f in sampled:
            images.append(os.path.join(folder, f))
            labels.append(label)

    for i, (img_path, label) in enumerate(zip(images, labels)):
        img = Image.open(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_image_sizes(raw_dir, classes, output_path):
    widths = []
    heights = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        for f in os.listdir(folder)[:200]:
            try:
                img = Image.open(os.path.join(folder, f))
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except:
                continue

    plt.figure()
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Size Distribution")
    plt.savefig(output_path)
    plt.close()


def plot_pixel_distribution(raw_dir, classes, output_path):
    pixels = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        for f in os.listdir(folder)[:100]:
            try:
                img = Image.open(os.path.join(folder, f)).convert("L")
                pixels.extend(np.array(img).flatten())
            except:
                continue

    plt.figure()
    plt.hist(pixels, bins=50)
    plt.title("Pixel Intensity Distribution")
    plt.savefig(output_path)
    plt.close()


def compute_pixel_stats(raw_dir, classes):
    pixels = []

    for label in classes:
        folder = os.path.join(raw_dir, label)
        for f in os.listdir(folder)[:200]:
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB")
                arr = np.array(img) / 255.0
                pixels.append(arr)
            except:
                continue

    pixels = np.concatenate([p.reshape(-1, 3) for p in pixels], axis=0)

    mean = pixels.mean(axis=0).tolist()
    std = pixels.std(axis=0).tolist()

    return {
        "mean": mean,
        "std": std
    }

def main():
    params = load_params()

    raw_dir = params["data"]["raw_dir"]
    classes = params["data"]["classes"]

    report_dir = params["eda_reports"]["base_dir"]
    metrics_dir = params["eda_reports"]["metrics_dir"]
    fig_dir = params["eda_reports"]["figures_dir"]

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Running EDA...")

    counts = count_images(raw_dir, classes)
    total = sum(counts.values())
    class_distribution = {k: v / total for k, v in counts.items()}

    size_stats = image_size_stats(raw_dir, classes)
    corrupt = check_corrupt(raw_dir, classes)
    pixel_stats = compute_pixel_stats(raw_dir, classes)

    results = {
    "counts": counts,
    "class_distribution": class_distribution,
    "image_size": size_stats,
    "corrupt_images": corrupt,
    "pixel_stats": pixel_stats
}
    

    with open(os.path.join(metrics_dir, "eda_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # plots
    plot_class_distribution(counts, os.path.join(fig_dir, "class_distribution.png"))
    plot_sample_images(raw_dir, classes, os.path.join(fig_dir, "sample_images.png"))
    plot_image_sizes(raw_dir, classes, os.path.join(fig_dir, "image_size_scatter.png"))
    plot_pixel_distribution(raw_dir, classes, os.path.join(fig_dir, "pixel_intensity_hist.png"))

    print("EDA completed. Results saved.")


if __name__ == "__main__":
    main()