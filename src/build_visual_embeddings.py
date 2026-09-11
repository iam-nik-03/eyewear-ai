import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# EYewear AI - VISUAL EMBEDDING DATASET
# ============================================================

DATASET_ROOT = (
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19"
    r"\face-shape-dataset\versions\2\FaceShape Dataset"
)

OUTPUT_DIR = r"data\processed\faces\visual_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("EYewear AI - VISUAL EMBEDDING EXTRACTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Find all images
# ------------------------------------------------------------

classes = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square"
]

image_records = []

for split in ["training_set", "testing_set"]:

    split_path = os.path.join(DATASET_ROOT, split)

    for label in classes:

        class_path = os.path.join(split_path, label)

        if not os.path.exists(class_path):
            print(f"WARNING: Missing folder: {class_path}")
            continue

        for filename in os.listdir(class_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif")
            ):

                image_records.append({
                    "image_path": os.path.join(
                        class_path,
                        filename
                    ),
                    "label": label,
                    "split": split
                })


df = pd.DataFrame(image_records)

print("\nImages found:")
print(f"Total: {len(df)}")

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# ------------------------------------------------------------
# 2. Load pretrained ResNet-50
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Loading ResNet-50...")
print("-" * 70)

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

# Remove the final classification layer.
# This makes ResNet return the 2048 visual features.
model.fc = torch.nn.Identity()

model.eval()

preprocess = weights.transforms()

print("ResNet-50 loaded successfully.")


# ------------------------------------------------------------
# 3. Dataset class
# ------------------------------------------------------------

class FaceDataset(Dataset):

    def __init__(self, dataframe, transform):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        try:

            image = Image.open(
                row["image_path"]
            ).convert("RGB")

            image = self.transform(image)

            return (
                image,
                row["image_path"],
                row["label"],
                row["split"]
            )

        except Exception as e:

            return (
                None,
                row["image_path"],
                row["label"],
                row["split"]
            )


# ------------------------------------------------------------
# 4. Custom collate function
# ------------------------------------------------------------

def collate_fn(batch):

    valid_items = [
        item for item in batch
        if item[0] is not None
    ]

    if not valid_items:
        return None

    images = torch.stack(
        [item[0] for item in valid_items]
    )

    paths = [
        item[1]
        for item in valid_items
    ]

    labels = [
        item[2]
        for item in valid_items
    ]

    splits = [
        item[3]
        for item in valid_items
    ]

    return images, paths, labels, splits


# ------------------------------------------------------------
# 5. Create DataLoader
# ------------------------------------------------------------

dataset = FaceDataset(
    df,
    preprocess
)

# Small batch because we're using CPU.
BATCH_SIZE = 16

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ------------------------------------------------------------
# 6. Extract embeddings
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Extracting visual embeddings...")
print("-" * 70)

all_embeddings = []
metadata = []
failures = []

processed = 0

with torch.no_grad():

    for batch_number, batch in enumerate(loader, start=1):

        if batch is None:
            continue

        images, paths, labels, splits = batch

        try:

            embeddings = model(images)

            embeddings = embeddings.cpu().numpy()

            for i in range(len(paths)):

                all_embeddings.append(
                    embeddings[i]
                )

                metadata.append({
                    "image_path": paths[i],
                    "label": labels[i],
                    "split": splits[i]
                })

                processed += 1

        except Exception as e:

            for path, label, split in zip(
                paths,
                labels,
                splits
            ):

                failures.append({
                    "image_path": path,
                    "label": label,
                    "split": split,
                    "error": str(e)
                })

        if batch_number % 20 == 0:

            print(
                f"Processed: {processed}/{len(df)}"
            )


# ------------------------------------------------------------
# 7. Convert embeddings to NumPy array
# ------------------------------------------------------------

embeddings_array = np.array(
    all_embeddings,
    dtype=np.float32
)

metadata_df = pd.DataFrame(
    metadata
)

failures_df = pd.DataFrame(
    failures
)


# ------------------------------------------------------------
# 8. Save embeddings
# ------------------------------------------------------------

embedding_path = os.path.join(
    OUTPUT_DIR,
    "embeddings.npy"
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.csv"
)

failure_path = os.path.join(
    OUTPUT_DIR,
    "failures.csv"
)

np.save(
    embedding_path,
    embeddings_array
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

failures_df.to_csv(
    failure_path,
    index=False
)


# ------------------------------------------------------------
# 9. Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VISUAL EMBEDDING EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nImages successfully processed: {len(embeddings_array)}")

print(
    f"Embedding shape: "
    f"{embeddings_array.shape}"
)

print(
    f"Failures: "
    f"{len(failures_df)}"
)

print("\nSaved:")

print(embedding_path)
print(metadata_path)
print(failure_path)

print("\nExpected embedding shape:")
print("(5000, 2048)")