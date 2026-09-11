from pathlib import Path

# Dataset path downloaded by KaggleHub
dataset_path = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2"
)

print("=" * 60)
print("EYewear AI - DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset path:")
print(dataset_path)

if not dataset_path.exists():
    print("\nERROR: Dataset path does not exist.")
    exit()

print("\nDataset exists: YES")

# Show folders and files directly inside dataset
print("\n" + "-" * 60)
print("CONTENTS")
print("-" * 60)

for item in sorted(dataset_path.iterdir()):
    if item.is_dir():
        print(f"[FOLDER] {item.name}")
    else:
        print(f"[FILE]   {item.name}")

# Count image files
image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

image_files = [
    file
    for file in dataset_path.rglob("*")
    if file.is_file() and file.suffix.lower() in image_extensions
]

print("\n" + "-" * 60)
print("IMAGE STATISTICS")
print("-" * 60)

print(f"Total images found: {len(image_files):,}")

# Count images by immediate parent folder
folder_counts = {}

for image in image_files:
    folder_name = image.parent.name
    folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1

print("\nImages by folder:")

for folder, count in sorted(folder_counts.items()):
    print(f"  {folder}: {count:,}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)