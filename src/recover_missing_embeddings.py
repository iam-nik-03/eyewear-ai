import os

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageFile
from torchvision.models import resnet50, ResNet50_Weights


# Allow Pillow to read the one recoverable truncated JPEG
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# EYewear AI - RECOVER MISSING EMBEDDINGS
# ============================================================

DATASET_ROOT = (
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19"
    r"\face-shape-dataset\versions\2\FaceShape Dataset"
)

EMBEDDING_DIR = (
    r"data\processed\faces\visual_embeddings"
)

EMBEDDING_PATH = os.path.join(
    EMBEDDING_DIR,
    "embeddings.npy"
)

METADATA_PATH = os.path.join(
    EMBEDDING_DIR,
    "metadata.csv"
)


# ------------------------------------------------------------
# 1. Missing images
# ------------------------------------------------------------

missing_images = [
    {
        "path": os.path.join(
            DATASET_ROOT,
            "training_set",
            "Heart",
            "heart (633).jpg"
        ),
        "label": "Heart",
        "split": "training_set"
    },
    {
        "path": os.path.join(
            DATASET_ROOT,
            "training_set",
            "Square",
            "square (84).jpg"
        ),
        "label": "Square",
        "split": "training_set"
    }
]


print("=" * 70)
print("EYewear AI - RECOVER MISSING EMBEDDINGS")
print("=" * 70)


# ------------------------------------------------------------
# 2. Load existing data
# ------------------------------------------------------------

embeddings = np.load(
    EMBEDDING_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

print("\nExisting embeddings:")
print(embeddings.shape)

print("\nExisting metadata:")
print(len(metadata))


# ------------------------------------------------------------
# 3. Load ResNet-50
# ------------------------------------------------------------

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

model.fc = torch.nn.Identity()

model.eval()

preprocess = weights.transforms()

print("ResNet-50 loaded.")


# ------------------------------------------------------------
# 4. Process missing images
# ------------------------------------------------------------

new_embeddings = []
new_metadata = []

for item in missing_images:

    path = item["path"]

    print("\nProcessing:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = Image.open(
        path
    ).convert("RGB")

    input_tensor = preprocess(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        embedding = model(
            input_tensor
        )

    embedding = (
        embedding
        .cpu()
        .numpy()[0]
        .astype(np.float32)
    )

    print(
        "Embedding shape:",
        embedding.shape
    )

    new_embeddings.append(
        embedding
    )

    new_metadata.append({
        "image_path": path,
        "label": item["label"],
        "split": item["split"]
    })


# ------------------------------------------------------------
# 5. Add recovered embeddings
# ------------------------------------------------------------

new_embeddings = np.array(
    new_embeddings,
    dtype=np.float32
)

updated_embeddings = np.vstack([
    embeddings,
    new_embeddings
])

updated_metadata = pd.concat(
    [
        metadata,
        pd.DataFrame(new_metadata)
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------

np.save(
    EMBEDDING_PATH,
    updated_embeddings
)

updated_metadata.to_csv(
    METADATA_PATH,
    index=False
)


# ------------------------------------------------------------
# 7. Verify
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RECOVERY COMPLETE")
print("=" * 70)

print("\nFinal embeddings shape:")
print(updated_embeddings.shape)

print("\nFinal metadata rows:")
print(len(updated_metadata))

print("\nClass distribution:")

print(
    updated_metadata
    .groupby(["split", "label"])
    .size()
    .to_string()
)

print("\nExpected:")
print("Training: 4000")
print("Testing:  1000")
print("Total:    5000")

print("\nSaved successfully.")