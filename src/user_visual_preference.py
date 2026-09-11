import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

EMBEDDINGS_PATH = Path(
    "data/processed/eyewear/frame_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/eyewear/frame_metadata.csv"
)

FEEDBACK_PATH = Path(
    "data/catalogue/feedback.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

feedback = pd.read_csv(
    FEEDBACK_PATH
)


print("\n==============================================")
print("USER VISUAL PREFERENCE")
print("==============================================")


# ============================================================
# CURRENT USER
# ============================================================

user_id = "USER_001"


user_feedback = feedback[
    feedback["user_id"] == user_id
]


# ============================================================
# FIND LIKED FRAMES
# ============================================================

liked_frames = user_feedback[
    user_feedback["liked"] == 1
]["frame_id"].tolist()


print(
    f"User: {user_id}"
)

print(
    f"Liked frames: {liked_frames}"
)


# ============================================================
# GET EMBEDDINGS
# ============================================================

liked_indices = []

for frame_id in liked_frames:

    matches = metadata.index[
        metadata["frame_id"] == frame_id
    ].tolist()

    if matches:

        liked_indices.append(
            matches[0]
        )


if len(liked_indices) == 0:

    raise ValueError(
        "User has no liked frames."
    )


liked_embeddings = embeddings[
    liked_indices
]


# ============================================================
# CREATE USER VISUAL VECTOR
# ============================================================

user_visual_vector = (
    liked_embeddings.mean(
        axis=0
    )
)


# Normalize the vector
norm = np.linalg.norm(
    user_visual_vector
)

if norm > 0:

    user_visual_vector = (
        user_visual_vector / norm
    )


# ============================================================
# CALCULATE SIMILARITY TO EVERY FRAME
# ============================================================

similarities = cosine_similarity(
    user_visual_vector.reshape(1, -1),
    embeddings
)[0]


results = metadata[
    ["frame_id"]
].copy()

results["visual_preference_score"] = (
    similarities
)

results = results.sort_values(
    "visual_preference_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# DISPLAY
# ============================================================

print("\nVisual preference ranking:")

print(
    results.to_string(
        index=False
    )
)


print("\n==============================================")
print("VISUAL PREFERENCE COMPLETE")
print("==============================================")