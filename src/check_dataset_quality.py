from pathlib import Path
from PIL import Image
from collections import Counter


DATASET_PATH = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2\FaceShape Dataset"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def main():
    print("=" * 70)
    print("EYewear AI - DATASET QUALITY CHECK")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print("\nERROR: Dataset path does not exist.")
        print(DATASET_PATH)
        return

    image_files = [
        file
        for file in DATASET_PATH.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nTotal images: {len(image_files):,}")

    # ---------------------------------------------------------
    # Image statistics
    # ---------------------------------------------------------

    corrupted_images = []
    image_modes = Counter()
    image_formats = Counter()
    image_sizes = Counter()

    width_values = []
    height_values = []

    for index, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as image:

                # Verify image integrity
                image.verify()

            # Re-open after verify()
            with Image.open(image_path) as image:

                image_modes[image.mode] += 1
                image_formats[image.format] += 1

                width, height = image.size

                width_values.append(width)
                height_values.append(height)

                image_sizes[(width, height)] += 1

        except Exception:
            corrupted_images.append(image_path)

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("IMAGE FORMATS")
    print("-" * 70)

    for image_format, count in image_formats.most_common():
        print(f"{image_format}: {count:,}")

    print("\n" + "-" * 70)
    print("IMAGE MODES")
    print("-" * 70)

    for mode, count in image_modes.most_common():
        print(f"{mode}: {count:,}")

    print("\n" + "-" * 70)
    print("IMAGE DIMENSIONS")
    print("-" * 70)

    print(f"Unique dimensions: {len(image_sizes):,}")

    print("\nMost common dimensions:")

    for size, count in image_sizes.most_common(10):
        print(f"{size[0]} x {size[1]} : {count:,}")

    if width_values and height_values:

        print("\nMinimum width:", min(width_values))
        print("Maximum width:", max(width_values))

        print("Minimum height:", min(height_values))
        print("Maximum height:", max(height_values))

    # ---------------------------------------------------------
    # Corrupted images
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CORRUPTED IMAGES")
    print("-" * 70)

    print(f"Corrupted images: {len(corrupted_images):,}")

    if corrupted_images:

        print("\nExamples:")

        for image in corrupted_images[:10]:
            print(image)

    # ---------------------------------------------------------
    # Class statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    class_counts = Counter()

    for image_path in image_files:

        class_name = image_path.parent.name
        class_counts[class_name] += 1

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count:,}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITY CHECK COMPLETE")
    print("=" * 70)

    print(f"\nTotal images       : {len(image_files):,}")
    print(f"Valid images       : {len(image_files) - len(corrupted_images):,}")
    print(f"Corrupted images   : {len(corrupted_images):,}")
    print(f"Unique dimensions  : {len(image_sizes):,}")

    if len(corrupted_images) == 0:
        print("\nSTATUS: Dataset integrity looks good.")
    else:
        print("\nSTATUS: Dataset contains corrupted images.")

    print("\nNext step: inspect sample images visually.")


if __name__ == "__main__":
    main()