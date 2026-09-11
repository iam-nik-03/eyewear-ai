import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = "data/processed/eyewear/frame_embeddings.npy"
METADATA_FILE = "data/processed/eyewear/frame_metadata.csv"

# Load embeddings
embeddings = np.load(EMBEDDINGS_FILE)

# Load metadata
metadata = pd.read_csv(METADATA_FILE)

print("Embeddings shape:", embeddings.shape)
print("Metadata rows:", len(metadata))

# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity matrix shape:", similarity_matrix.shape)

# Display similarity matrix
print("\nCosine Similarity Matrix:\n")

print(
    pd.DataFrame(
        similarity_matrix,
        index=metadata["frame_id"],
        columns=metadata["frame_id"]
    ).round(3)
)

# --------------------------------------------------
# Find most visually similar frame for each frame
# --------------------------------------------------

print("\nMost visually similar frame:\n")

for i, frame_id in enumerate(metadata["frame_id"]):

    similarities = similarity_matrix[i].copy()

    # Ignore the frame itself
    similarities[i] = -1

    best_index = np.argmax(similarities)

    best_frame = metadata.iloc[best_index]["frame_id"]
    score = similarities[best_index]

    print(
        f"{frame_id} → {best_frame} "
        f"(similarity = {score:.3f})"
    )