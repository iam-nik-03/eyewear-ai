import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

EMBEDDINGS_PATH = Path(
    "data/processed/eyewear/frame_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/eyewear/frame_metadata.csv"
)

CATALOGUE_PATH = Path(
    "data/catalogue/eyewear_catalogue.csv"
)


# ============================================================
# LOAD
# ============================================================

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

catalogue = pd.read_csv(
    CATALOGUE_PATH
)


# ============================================================
# CHECK
# ============================================================

print("\n==============================================")
print("CHECKING FRAME VISUAL EMBEDDINGS")
print("==============================================")

print(
    f"Embedding shape: {embeddings.shape}"
)

print(
    f"Embedding metadata rows: {len(metadata)}"
)

print(
    f"Catalogue rows: {len(catalogue)}"
)


# ============================================================
# FRAME ID CHECK
# ============================================================

embedding_ids = set(
    metadata["frame_id"]
)

catalogue_ids = set(
    catalogue["frame_id"]
)


missing_embeddings = (
    catalogue_ids - embedding_ids
)

extra_embeddings = (
    embedding_ids - catalogue_ids
)


print(
    f"\nMissing embeddings: "
    f"{len(missing_embeddings)}"
)

print(
    f"Extra embeddings: "
    f"{len(extra_embeddings)}"
)


# ============================================================
# ORDER CHECK
# ============================================================

merged = catalogue[
    ["frame_id"]
].merge(
    metadata[
        ["frame_id"]
    ],
    on="frame_id",
    how="left",
    indicator=True
)


print("\nCatalogue → embedding mapping:")

print(
    merged.to_string(
        index=False
    )
)


# ============================================================
# FINAL STATUS
# ============================================================

if (
    len(missing_embeddings) == 0
    and len(extra_embeddings) == 0
):

    print(
        "\nSTATUS: ✓ All catalogue frames "
        "have embeddings."
    )

else:

    print(
        "\nSTATUS: ⚠ Embedding mapping needs attention."
    )


print("\n==============================================")
print("CHECK COMPLETE")
print("==============================================")