import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# --------------------------------------------------
# Paths
# --------------------------------------------------

CATALOGUE_FILE = "data/catalogue/eyewear_catalogue.csv"
IMAGE_DIR = "data/catalogue/images"

OUTPUT_DIR = "data/processed/eyewear"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "frame_embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "frame_metadata.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("Using device:", device)

# --------------------------------------------------
# Load ResNet-50
# --------------------------------------------------

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)

# Remove classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("ResNet-50 loaded successfully.")

# --------------------------------------------------
# Load catalogue
# --------------------------------------------------

catalogue = pd.read_csv(CATALOGUE_FILE)

print("\nCatalogue:")
print(catalogue[["frame_id", "frame_image", "frame_shape"]])

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

embeddings = []
metadata = []
failures = []

print("\nGenerating frame embeddings...\n")

with torch.no_grad():

    for _, row in catalogue.iterrows():

        frame_id = row["frame_id"]
        image_name = row["frame_image"]

        image_path = os.path.join(IMAGE_DIR, image_name)

        print(f"Processing {frame_id}: {image_name}")

        try:

            image = Image.open(image_path).convert("RGB")

            image_tensor = preprocess(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)

            embedding = model(image_tensor)

            embedding = embedding.squeeze(0).numpy()

            embeddings.append(embedding)

            metadata.append(row.to_dict())

            print("  Embedding shape:", embedding.shape)

        except Exception as e:

            print("  FAILED:", e)

            failures.append({
                "frame_id": frame_id,
                "image": image_name,
                "error": str(e)
            })

# --------------------------------------------------
# Save results
# --------------------------------------------------

if embeddings:

    embeddings_array = np.stack(embeddings)

    np.save(
        EMBEDDINGS_FILE,
        embeddings_array
    )

    metadata_df = pd.DataFrame(metadata)

    metadata_df.to_csv(
        METADATA_FILE,
        index=False
    )

    print("\n========================================")
    print("FRAME EMBEDDING COMPLETE")
    print("========================================")

    print("Embeddings shape:", embeddings_array.shape)
    print("Metadata rows:", len(metadata_df))

else:

    print("\nNo embeddings were generated.")


# --------------------------------------------------
# Save failures
# --------------------------------------------------

if failures:

    failures_df = pd.DataFrame(failures)

    failures_file = os.path.join(
        OUTPUT_DIR,
        "embedding_failures.csv"
    )

    failures_df.to_csv(
        failures_file,
        index=False
    )

    print("\nFailures:", len(failures))

else:

    print("\nFailures: 0")