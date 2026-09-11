from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOGUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "eyewear_catalogue.csv"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eyewear"
    / "frame_embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eyewear"
    / "frame_metadata.csv"
)

FEEDBACK_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "feedback.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "user_visual_profile.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Embedding count does not match metadata count."
        )

    catalogue_ids = set(
        catalogue["frame_id"]
        .astype(str)
    )

    metadata_ids = set(
        metadata["frame_id"]
        .astype(str)
    )

    if catalogue_ids != metadata_ids:

        raise ValueError(
            "Catalogue and embedding frame IDs do not match."
        )

    return (
        catalogue,
        embeddings,
        metadata,
    )


# ============================================================
# EMBEDDING LOOKUP
# ============================================================

def build_embedding_lookup(
    embeddings,
    metadata,
):

    lookup = {}

    for index, row in metadata.iterrows():

        frame_id = str(
            row["frame_id"]
        )

        vector = embeddings[
            index
        ].astype(
            np.float32
        )

        norm = np.linalg.norm(
            vector
        )

        if norm == 0:

            raise ValueError(
                f"Zero-norm embedding found for {frame_id}."
            )

        lookup[
            frame_id
        ] = vector / norm

    return lookup


# ============================================================
# FEEDBACK
# ============================================================

def load_feedback():

    if FEEDBACK_PATH.exists():

        feedback = pd.read_csv(
            FEEDBACK_PATH
        )

    else:

        feedback = pd.DataFrame(
            columns=[
                "user_id",
                "frame_id",
                "liked",
            ]
        )

    required = {
        "user_id",
        "frame_id",
        "liked",
    }

    if not required.issubset(
        feedback.columns
    ):

        raise ValueError(
            "feedback.csv must contain: "
            "user_id, frame_id, liked"
        )

    feedback["user_id"] = (
        feedback["user_id"]
        .astype(str)
    )

    feedback["frame_id"] = (
        feedback["frame_id"]
        .astype(str)
    )

    feedback["liked"] = (
        feedback["liked"]
        .astype(int)
    )

    return feedback


# ============================================================
# UPDATE FEEDBACK
# ============================================================

def update_feedback(
    user_id,
    frame_id,
    liked,
):
    """
    Add or update a user's feedback for a frame.

    liked:
        1 = Like
        0 = Dislike
    """

    user_id = str(
        user_id
    )

    frame_id = str(
        frame_id
    )

    liked = int(
        liked
    )

    if liked not in {
        0,
        1,
    }:

        raise ValueError(
            "liked must be 0 or 1."
        )

    catalogue, _, _ = load_data()

    valid_frame_ids = (
        catalogue[
            "frame_id"
        ]
        .astype(str)
        .tolist()
    )

    if frame_id not in valid_frame_ids:

        raise ValueError(
            f"Invalid frame ID: {frame_id}"
        )

    feedback = load_feedback()

    # Remove previous interaction for
    # this exact user/frame pair.
    feedback = feedback[
        ~(
            (
                feedback["user_id"]
                == user_id
            )
            &
            (
                feedback["frame_id"]
                == frame_id
            )
        )
    ].copy()

    new_row = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "frame_id": frame_id,
                "liked": liked,
            }
        ]
    )

    feedback = pd.concat(
        [
            feedback,
            new_row,
        ],
        ignore_index=True,
    )

    FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feedback.to_csv(
        FEEDBACK_PATH,
        index=False,
    )

    return feedback


# ============================================================
# USER FEEDBACK
# ============================================================

def get_user_feedback(
    feedback,
    user_id,
):

    return feedback[
        feedback["user_id"]
        == str(user_id)
    ].copy()


# ============================================================
# BUILD USER VISUAL PROFILE
# ============================================================

def build_user_visual_profile(
    catalogue,
    embedding_lookup,
    user_feedback,
    user_id,
):
    """
    Build a user visual-preference profile from
    positively rated frames.

    The user vector is the normalized mean of
    embeddings belonging to liked frames.

    Frame scores are cosine similarities between
    that user vector and each frame embedding.
    """

    liked_frames = (
        user_feedback[
            user_feedback["liked"] == 1
        ]["frame_id"]
        .astype(str)
        .tolist()
    )

    disliked_frames = (
        user_feedback[
            user_feedback["liked"] == 0
        ]["frame_id"]
        .astype(str)
        .tolist()
    )

    liked_embeddings = [
        embedding_lookup[frame_id]
        for frame_id in liked_frames
        if frame_id in embedding_lookup
    ]

    # --------------------------------------------------------
    # Build user vector
    # --------------------------------------------------------

    if liked_embeddings:

        user_vector = np.mean(
            liked_embeddings,
            axis=0,
        )

        norm = np.linalg.norm(
            user_vector
        )

        if norm > 0:

            user_vector = (
                user_vector / norm
            )

        else:

            user_vector = None

    else:

        user_vector = None

    # --------------------------------------------------------
    # Score every frame
    # --------------------------------------------------------

    rows = []

    for _, frame in catalogue.iterrows():

        frame_id = str(
            frame["frame_id"]
        )

        frame_vector = (
            embedding_lookup.get(
                frame_id
            )
        )

        if (
            user_vector is not None
            and frame_vector is not None
        ):

            score = float(
                np.dot(
                    user_vector,
                    frame_vector,
                )
            )

            score = float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )

        else:

            score = 0.0

        rows.append(
            {
                "user_id": str(user_id),
                "frame_id": frame_id,
                "visual_preference_score": score,
                "previously_liked": int(
                    frame_id
                    in liked_frames
                ),
                "previously_disliked": int(
                    frame_id
                    in disliked_frames
                ),
            }
        )

    profile = pd.DataFrame(
        rows
    )

    profile = (
        profile
        .sort_values(
            "visual_preference_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        profile,
        liked_frames,
        disliked_frames,
    )


# ============================================================
# REBUILD PROFILE
# ============================================================

def rebuild_user_profile(
    user_id,
):
    """
    Rebuild the visual preference profile after
    feedback changes.
    """

    catalogue, embeddings, metadata = (
        load_data()
    )

    feedback = load_feedback()

    user_feedback = get_user_feedback(
        feedback,
        user_id,
    )

    embedding_lookup = (
        build_embedding_lookup(
            embeddings,
            metadata,
        )
    )

    (
        profile,
        liked_frames,
        disliked_frames,
    ) = build_user_visual_profile(
        catalogue,
        embedding_lookup,
        user_feedback,
        user_id,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # If this is the active user's profile,
    # preserve the existing file behaviour.
    if str(user_id) == "USER_002":

        profile.to_csv(
            OUTPUT_PATH,
            index=False,
        )

    return (
        profile,
        liked_frames,
        disliked_frames,
    )


# ============================================================
# ONE-STEP PERSONALIZATION UPDATE
# ============================================================

def record_feedback(
    user_id,
    frame_id,
    liked,
):
    """
    Public function for the Streamlit UI.

    1. Stores feedback.
    2. Rebuilds the visual profile.
    3. Returns the updated profile.
    """

    update_feedback(
        user_id=user_id,
        frame_id=frame_id,
        liked=liked,
    )

    (
        profile,
        liked_frames,
        disliked_frames,
    ) = rebuild_user_profile(
        user_id
    )

    return {
        "profile": profile,
        "liked_frames": liked_frames,
        "disliked_frames": disliked_frames,
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    user_id = "USER_002"

    print(
        "\n=============================================="
    )

    print(
        "PERSONALIZATION LOOP"
    )

    print(
        "=============================================="
    )

    print(
        f"User: {user_id}"
    )

    catalogue, embeddings, metadata = (
        load_data()
    )

    print(
        f"Catalogue frames: {len(catalogue)}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        "\n=============================================="
    )

    print(
        "AVAILABLE EYEWEAR FRAMES"
    )

    print(
        "=============================================="
    )

    print(
        catalogue[
            [
                "frame_id",
                "brand",
                "frame_shape",
                "frame_color",
                "style",
            ]
        ].to_string(
            index=False
        )
    )

    valid_frame_ids = (
        catalogue[
            "frame_id"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    while True:

        frame_id = input(
            "\nEnter Frame ID "
            "(example: FRAME_007): "
        ).strip().upper()

        if frame_id in valid_frame_ids:
            break

        print(
            "Invalid Frame ID."
        )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    while True:

        feedback_input = input(
            "Enter feedback "
            "(1 = Like, 0 = Dislike): "
        ).strip()

        if feedback_input in {
            "0",
            "1",
        }:

            liked = int(
                feedback_input
            )

            break

        print(
            "Invalid feedback."
        )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    result = record_feedback(
        user_id=user_id,
        frame_id=frame_id,
        liked=liked,
    )

    profile = result[
        "profile"
    ]

    liked_frames = result[
        "liked_frames"
    ]

    disliked_frames = result[
        "disliked_frames"
    ]

    print(
        "\n=============================================="
    )

    print(
        "FEEDBACK UPDATED"
    )

    print(
        "=============================================="
    )

    print(
        f"Frame: {frame_id}"
    )

    print(
        f"Feedback: "
        f"{'LIKE' if liked == 1 else 'DISLIKE'}"
    )

    print(
        f"\nLiked frames: {liked_frames}"
    )

    print(
        f"Disliked frames: {disliked_frames}"
    )

    print(
        "\n=============================================="
    )

    print(
        "UPDATED VISUAL PREFERENCE"
    )

    print(
        "=============================================="
    )

    print(
        profile.to_string(
            index=False
        )
    )

    minimum_score = profile[
        "visual_preference_score"
    ].min()

    maximum_score = profile[
        "visual_preference_score"
    ].max()

    print(
        "\n=============================================="
    )

    print(
        "SCORE VALIDATION"
    )

    print(
        "=============================================="
    )

    print(
        f"Minimum score: {minimum_score:.6f}"
    )

    print(
        f"Maximum score: {maximum_score:.6f}"
    )

    if (
        minimum_score >= -1.0
        and maximum_score <= 1.0
    ):

        print(
            "Cosine similarity validation: PASSED"
        )

    else:

        print(
            "Cosine similarity validation: FAILED"
        )

    print(
        "\nProfile saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()