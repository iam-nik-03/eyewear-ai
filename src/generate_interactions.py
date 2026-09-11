import numpy as np
import pandas as pd
import random

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CATALOGUE_PATH = Path(
    "data/catalogue/eyewear_catalogue.csv"
)

EMBEDDINGS_PATH = Path(
    "data/processed/eyewear/frame_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/eyewear/frame_metadata.csv"
)

OUTPUT_DIR = Path(
    "data/processed/recommendations"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "synthetic_hybrid_interactions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
NUM_USERS = 100

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

catalogue = pd.read_csv(
    CATALOGUE_PATH
)

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


print("\n==============================================")
print("GENERATING SYNTHETIC HYBRID INTERACTIONS")
print("==============================================")


# ============================================================
# VERIFY EMBEDDINGS
# ============================================================

embedding_lookup = {}

for index, row in metadata.iterrows():

    embedding_lookup[
        row["frame_id"]
    ] = embeddings[index]


# ============================================================
# FACE COMPATIBILITY
# ============================================================

face_compatibility_rules = {
    "Heart": {
        "Rectangle": 0.85,
        "Cat Eye": 0.75,
        "Aviator": 0.90,
        "Geometric": 0.70,
        "Round": 0.80,
        "Square": 0.65,
        "Clubmaster": 0.85,
    },

    "Oblong": {
        "Rectangle": 0.85,
        "Cat Eye": 0.75,
        "Aviator": 0.90,
        "Geometric": 0.75,
        "Round": 0.90,
        "Square": 0.80,
        "Clubmaster": 0.85,
    },

    "Oval": {
        "Rectangle": 0.90,
        "Cat Eye": 0.85,
        "Aviator": 0.85,
        "Geometric": 0.85,
        "Round": 0.90,
        "Square": 0.85,
        "Clubmaster": 0.90,
    },

    "Round": {
        "Rectangle": 1.00,
        "Cat Eye": 0.75,
        "Aviator": 0.85,
        "Geometric": 0.90,
        "Round": 0.65,
        "Square": 0.90,
        "Clubmaster": 0.85,
    },

    "Square": {
        "Rectangle": 0.70,
        "Cat Eye": 0.85,
        "Aviator": 0.90,
        "Geometric": 0.80,
        "Round": 0.90,
        "Square": 0.70,
        "Clubmaster": 0.85,
    },
}

def get_face_compatibility(
    face_shape,
    frame_shape
):

    return face_compatibility_rules.get(
        face_shape,
        {}
    ).get(
        frame_shape,
        0.0
    )


# ============================================================
# POSSIBLE USER PREFERENCES
# ============================================================

face_shapes = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square"
]

styles = [
    "Casual",
    "Fashion",
    "Professional"
]

colors = [
    "Black",
    "Silver",
    "Purple",
    "Transparent"
]

frame_types = [
    "Optical"
]


# ============================================================
# GENERATE USERS
# ============================================================

users = []

for user_number in range(
    1,
    NUM_USERS + 1
):

    users.append({
        "user_id":
            f"SYNTH_USER_{user_number:03d}",

        "face_shape":
            random.choice(face_shapes),

        "preferred_style":
            random.choice(styles),

        "preferred_color":
            random.choice(colors),

        "preferred_frame_type":
            random.choice(frame_types)
    })


# ============================================================
# GENERATE INTERACTIONS
# ============================================================

rows = []


for user in users:

    user_id = user["user_id"]

    # --------------------------------------------------------
    # Calculate base preference score for each frame
    # --------------------------------------------------------

    frame_scores = []

    for _, frame in catalogue.iterrows():

        face_score = get_face_compatibility(
            user["face_shape"],
            frame["frame_shape"]
        )

        style_match = int(
            user["preferred_style"].lower()
            ==
            str(frame["style"]).lower()
        )

        color_match = int(
            user["preferred_color"].lower()
            in
            str(frame["frame_color"]).lower()
        )

        frame_type_match = int(
            user["preferred_frame_type"].lower()
            ==
            str(frame["frame_type"]).lower()
        )

        base_score = (
            0.50 * face_score
            + 0.25 * style_match
            + 0.15 * color_match
            + 0.10 * frame_type_match
        )

        frame_scores.append({
            "frame_id":
                frame["frame_id"],

            "base_score":
                base_score,

            "face_score":
                face_score,

            "style_match":
                style_match,

            "color_match":
                color_match,

            "frame_type_match":
                frame_type_match
        })


    # --------------------------------------------------------
    # Simulate which frames this user likes
    # --------------------------------------------------------

    liked_frame_ids = []

    for item in frame_scores:

        noise = random.uniform(
            -0.15,
            0.15
        )

        preference_score = (
            item["base_score"]
            + noise
        )

        liked = int(
            preference_score >= 0.55
        )

        if liked == 1:

            liked_frame_ids.append(
                item["frame_id"]
            )


    # --------------------------------------------------------
    # Guarantee at least one liked frame
    # --------------------------------------------------------

    if len(liked_frame_ids) == 0:

        best_frame = max(
            frame_scores,
            key=lambda x: x["base_score"]
        )

        liked_frame_ids.append(
            best_frame["frame_id"]
        )


    # --------------------------------------------------------
    # Build user visual preference vector
    # --------------------------------------------------------

    liked_embeddings = []

    for frame_id in liked_frame_ids:

        if frame_id in embedding_lookup:

            liked_embeddings.append(
                embedding_lookup[frame_id]
            )


    if len(liked_embeddings) == 0:
        continue


    user_vector = np.mean(
        liked_embeddings,
        axis=0
    )


    # Normalize
    norm = np.linalg.norm(
        user_vector
    )

    if norm > 0:

        user_vector = (
            user_vector / norm
        )


    # --------------------------------------------------------
    # Calculate visual preference
    # --------------------------------------------------------

    for item in frame_scores:

        frame_id = item["frame_id"]

        frame_vector = embedding_lookup[
            frame_id
        ]

        frame_norm = np.linalg.norm(
            frame_vector
        )

        if frame_norm > 0:

            frame_vector = (
                frame_vector / frame_norm
            )


        visual_score = float(
            np.dot(
                user_vector,
                frame_vector
            )
        )


        # Find simulated feedback
        liked = int(
            frame_id in liked_frame_ids
        )


        rows.append({

            "user_id":
                user_id,

            "face_shape":
                user["face_shape"],

            "preferred_style":
                user["preferred_style"],

            "preferred_color":
                user["preferred_color"],

            "preferred_frame_type":
                user["preferred_frame_type"],

            "frame_id":
                frame_id,

            "face_compatibility":
                item["face_score"],

            "visual_preference_score":
                visual_score,

            "style_match":
                item["style_match"],

            "color_match":
                item["color_match"],

            "frame_type_match":
                item["frame_type_match"],

            "liked":
                liked,

            "data_source":
                "synthetic"
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

hybrid_df = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

hybrid_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"Users generated: "
    f"{hybrid_df['user_id'].nunique()}"
)

print(
    f"Frames per user: "
    f"{len(catalogue)}"
)

print(
    f"Interactions: "
    f"{len(hybrid_df)}"
)

print("\nLike distribution:")

print(
    hybrid_df["liked"]
    .value_counts()
    .rename(
        index={
            0: "Dislike",
            1: "Like"
        }
    )
)


print("\nVisual score range:")

print(
    f"Minimum: "
    f"{hybrid_df['visual_preference_score'].min():.4f}"
)

print(
    f"Maximum: "
    f"{hybrid_df['visual_preference_score'].max():.4f}"
)


print("\nFirst 10 rows:")

print(
    hybrid_df.head(10).to_string(
        index=False
    )
)


print("\n==============================================")
print("SYNTHETIC HYBRID DATASET SAVED")
print("==============================================")

print(
    f"Location: {OUTPUT_PATH}"
)