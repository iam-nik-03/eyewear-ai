from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATASET_PATH = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2\FaceShape Dataset"
)

CLASSES = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square",
]

SETS = [
    "training_set",
    "testing_set",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def get_images(set_name, class_name, number_of_images=3):

    class_path = DATASET_PATH / set_name / class_name

    if not class_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {class_path}"
        )

    images = [
        image
        for image in class_path.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.seed(42)

    return random.sample(
        images,
        min(number_of_images, len(images))
    )


def main():

    print("=" * 70)
    print("EYewear AI - VISUAL DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET_PATH)

    rows = len(CLASSES) * len(SETS)
    columns = 3

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 25)
    )

    row = 0

    for set_name in SETS:

        for class_name in CLASSES:

            print(f"\n{set_name} → {class_name}")

            images = get_images(
                set_name,
                class_name,
                columns
            )

            for column, image_path in enumerate(images):

                print(f"  {image_path.name}")

                try:

                    with Image.open(image_path) as image:

                        image = image.convert("RGB")

                        axes[row, column].imshow(image)

                        axes[row, column].set_title(
                            f"{set_name}\n{class_name}"
                        )

                        axes[row, column].axis("off")

                except Exception as error:

                    print(
                        f"  ERROR: {error}"
                    )

                    axes[row, column].axis("off")

            # Hide unused cells
            for column in range(len(images), columns):
                axes[row, column].axis("off")

            row += 1

    plt.tight_layout()

    output_path = Path(
        "data/dataset_sample_grid.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\n" + "=" * 70)
    print("VISUAL INSPECTION COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(output_path.resolve())

    plt.show()


if __name__ == "__main__":
    main()