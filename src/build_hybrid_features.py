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

CATALOGUE_PATH = Path(
    "data/catalogue/eyewear_catalogue.csv"
)

PREFERENCES_PATH = Path(
    "data/catalogue/user_preferences.csv"
)

FEEDBACK_PATH = Path(
    "data/catalogue/feedback.csv"
)

OUTPUT_DIR = Path(
    "data/processed/recommendations"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "hybrid_features.csv"
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

catalogue = pd.read_csv(
    CATALOGUE_PATH
)

preferences = pd.read_csv(
    PREFERENCES_PATH
)

feedback = pd.read_csv(
    FEEDBACK_PATH
)


print("\n==============================================")
print("BUILDING HYBRID RECOMMENDATION FEATURES")
print("==============================================")


# ============================================================
# FACE COMPATIBILITY
# ============================================================

face_compatibility_rules = {
    "Heart": {
        "Aviator": 0.90,
        "Cat Eye": 0.85,
        "Rectangle": 0.70,
        "Hexagon": 0.65,
        "Pentagon": 0.60,
    },
    "Oblong": {
        "Aviator": 0.90,
        "Rectangle": 0.85,
        "Cat Eye": 0.75,
        "Hexagon": 0.70,
        "Pentagon": 0.65,
    },
    "Oval": {
        "Rectangle": 0.90,
        "Aviator": 0.85,
        "Cat Eye": 0.80,
        "Hexagon": 0.80,
        "Pentagon": 0.75,
    },
    "Round": {
        "Rectangle": 1.00,
        "Hexagon": 0.90,
        "Pentagon": 0.85,
        "Aviator": 0.80,
        "Cat Eye": 0.70,
    },
    "Square": {
        "Aviator": 0.90,
        "Cat Eye": 0.85,
        "Hexagon": 0.80,
        "Rectangle": 0.70,
        "Pentagon": 0.65,
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
# PROCESS EACH USER
# ============================================================

rows = []


for _, user in preferences.iterrows():

    user_id = user["user_id"]

    user_feedback = feedback[
        feedback["user_id"] == user_id
    ]

    liked_frame_ids = user_feedback[
        user_feedback["liked"] == 1
    ]["frame_id"].tolist()


    if len(liked_frame_ids) == 0:
        continue


    # --------------------------------------------------------
    # Find embeddings for liked frames
    # --------------------------------------------------------

    liked_indices = []

    for frame_id in liked_frame_ids:

        matches = metadata.index[
            metadata["frame_id"] == frame_id
        ].tolist()

        if matches:

            liked_indices.append(
                matches[0]
            )


    if len(liked_indices) == 0:
        continue


    # --------------------------------------------------------
    # User visual preference vector
    # --------------------------------------------------------

    user_vector = embeddings[
        liked_indices
    ].mean(axis=0)


    norm = np.linalg.norm(
        user_vector
    )

    if norm > 0:

        user_vector = (
            user_vector / norm
        )


    # --------------------------------------------------------
    # Compare user preference with every frame
    # --------------------------------------------------------

    visual_scores = cosine_similarity(
        user_vector.reshape(1, -1),
        embeddings
    )[0]


    # --------------------------------------------------------
    # Build interaction features
    # --------------------------------------------------------

    for _, frame in catalogue.iterrows():

        frame_id = frame["frame_id"]

        metadata_matches = metadata.index[
            metadata["frame_id"] == frame_id
        ].tolist()


        if not metadata_matches:
            continue


        embedding_index = metadata_matches[0]


        face_score = get_face_compatibility(
            user["face_shape"],
            frame["frame_shape"]
        )


        style_match = int(
            str(user["preferred_style"]).lower()
            == str(frame["style"]).lower()
        )


        color_match = int(
            str(user["preferred_color"]).lower()
            in str(frame["frame_color"]).lower()
        )


        frame_type_match = int(
            str(user["preferred_frame_type"]).lower()
            == str(frame["frame_type"]).lower()
        )


        feedback_match = user_feedback[
            user_feedback["frame_id"] == frame_id
        ]


        if len(feedback_match) == 0:
            continue


        liked = int(
            feedback_match.iloc[0]["liked"]
        )


        rows.append({

            "user_id": user_id,

            "frame_id": frame_id,

            "face_compatibility":
                face_score,

            "visual_preference_score":
                visual_scores[embedding_index],

            "style_match":
                style_match,

            "color_match":
                color_match,

            "frame_type_match":
                frame_type_match,

            "liked":
                liked,

            "data_source":
                "real_feedback"

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
# DISPLAY
# ============================================================

print(
    f"Users processed: "
    f"{hybrid_df['user_id'].nunique()}"
)

print(
    f"Interactions: "
    f"{len(hybrid_df)}"
)

print("\nGenerated features:")

print(
    hybrid_df.to_string(
        index=False
    )
)


print("\n==============================================")
print("HYBRID DATASET SAVED")
print("==============================================")

print(
    f"Location: {OUTPUT_PATH}"
)