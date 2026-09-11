import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

USER_ID = "USER_002"
TOP_N = 5

MODEL_WEIGHT = 0.70
VISUAL_WEIGHT = 0.30


# ============================================================
# PATHS
# ============================================================

CATALOGUE_PATH = Path(
    "data/catalogue/eyewear_catalogue.csv"
)

MODEL_PATH = Path(
    "models/recommendation/hybrid_model.joblib"
)

FEEDBACK_PATH = Path(
    "data/catalogue/feedback.csv"
)

VISUAL_PROFILE_PATH = Path(
    "data/processed/recommendations/user_visual_profile.csv"
)

OUTPUT_PATH = Path(
    "data/processed/recommendations/latest_recommendations.csv"
)


# ============================================================
# USER PROFILE
# ============================================================

USER_PROFILE = {
    "face_shape": "Round",
    "preferred_style": "Casual",
    "preferred_color": "Black",
    "preferred_frame_type": "Optical",
    "budget": 3000,
}


# ============================================================
# FACE COMPATIBILITY
# ============================================================

FACE_COMPATIBILITY_RULES = {
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
    return (
        FACE_COMPATIBILITY_RULES
        .get(face_shape, {})
        .get(frame_shape, 0.0)
    )


# ============================================================
# LOAD CATALOGUE
# ============================================================

def load_catalogue():

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue not found: {CATALOGUE_PATH}"
        )

    catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    required = [
        "frame_id",
        "brand",
        "frame_shape",
        "frame_color",
        "style",
        "frame_type",
        "material",
    ]

    missing = [
        column
        for column in required
        if column not in catalogue.columns
    ]

    if missing:
        raise ValueError(
            f"Catalogue missing columns: {missing}"
        )

    catalogue["frame_id"] = (
        catalogue["frame_id"]
        .astype(str)
    )

    return catalogue


# ============================================================
# LOAD FEEDBACK
# ============================================================

def load_user_history():

    if not FEEDBACK_PATH.exists():

        return set(), set()

    feedback = pd.read_csv(
        FEEDBACK_PATH
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
            "feedback.csv must contain "
            "user_id, frame_id and liked."
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
        pd.to_numeric(
            feedback["liked"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    user_feedback = feedback[
        feedback["user_id"] == USER_ID
    ]

    liked = set(
        user_feedback[
            user_feedback["liked"] == 1
        ]["frame_id"]
    )

    disliked = set(
        user_feedback[
            user_feedback["liked"] == 0
        ]["frame_id"]
    )

    return liked, disliked


# ============================================================
# LOAD VISUAL PROFILE
# ============================================================

def load_visual_profile():

    if not VISUAL_PROFILE_PATH.exists():

        return {}

    profile = pd.read_csv(
        VISUAL_PROFILE_PATH
    )

    required = {
        "frame_id",
        "visual_preference_score",
    }

    if not required.issubset(
        profile.columns
    ):

        raise ValueError(
            "Visual profile must contain "
            "frame_id and visual_preference_score."
        )

    profile["frame_id"] = (
        profile["frame_id"]
        .astype(str)
    )

    profile[
        "visual_preference_score"
    ] = pd.to_numeric(
        profile[
            "visual_preference_score"
        ],
        errors="coerce"
    ).fillna(0.0)

    return dict(
        zip(
            profile["frame_id"],
            profile[
                "visual_preference_score"
            ]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("ML HYBRID EYEWEAR RECOMMENDER")
    print("==============================================")

    print(
        f"User: {USER_ID}"
    )


    # --------------------------------------------------------
    # Catalogue
    # --------------------------------------------------------

    catalogue = load_catalogue()

    print(
        f"Catalogue frames: {len(catalogue)}"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model loaded: {MODEL_PATH}"
    )


    # --------------------------------------------------------
    # Verify exact trained feature order
    # --------------------------------------------------------

    expected_features = [
        "face_compatibility",
        "visual_preference_score",
        "style_match",
        "color_match",
        "frame_type_match",
    ]

    if hasattr(
        model,
        "feature_names_in_"
    ):

        actual_features = list(
            model.feature_names_in_
        )

        print(
            "\nModel feature order:"
        )

        print(
            actual_features
        )

        if actual_features != expected_features:

            raise ValueError(
                "\nFeature order mismatch.\n"
                f"Expected: {expected_features}\n"
                f"Model: {actual_features}"
            )


    # --------------------------------------------------------
    # User history
    # --------------------------------------------------------

    liked_frames, disliked_frames = (
        load_user_history()
    )

    interacted_frames = (
        liked_frames
        |
        disliked_frames
    )


    print("\n==============================================")
    print("USER HISTORY")
    print("==============================================")


    print(
        "Liked:",
        sorted(liked_frames)
    )

    print(
        "Disliked:",
        sorted(disliked_frames)
    )


    # --------------------------------------------------------
    # Visual profile
    # --------------------------------------------------------

    visual_scores = (
        load_visual_profile()
    )


    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    rows = []


    for _, frame in catalogue.iterrows():

        frame_id = str(
            frame["frame_id"]
        )


        face_compatibility = (
            get_face_compatibility(
                USER_PROFILE["face_shape"],
                str(
                    frame["frame_shape"]
                )
            )
        )


        style_match = int(
            str(
                frame["style"]
            ).lower()
            ==
            str(
                USER_PROFILE["preferred_style"]
            ).lower()
        )


        color_match = int(
            str(
                USER_PROFILE["preferred_color"]
            ).lower()
            in
            str(
                frame["frame_color"]
            ).lower()
        )


        frame_type_match = int(
            str(
                frame["frame_type"]
            ).lower()
            ==
            str(
                USER_PROFILE["preferred_frame_type"]
            ).lower()
        )


        visual_preference_score = float(
            visual_scores.get(
                frame_id,
                0.0
            )
        )


        rows.append(
            {
                "frame_id":
                    frame_id,

                "face_compatibility":
                    face_compatibility,

                "visual_preference_score":
                    visual_preference_score,

                "style_match":
                    style_match,

                "color_match":
                    color_match,

                "frame_type_match":
                    frame_type_match,
            }
        )


    features_df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # EXACT MODEL INPUT
    # --------------------------------------------------------

    X = features_df[
        expected_features
    ].copy()


    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    like_probability = (
        model.predict_proba(X)[:, 1]
    )


    features_df[
        "model_probability"
    ] = like_probability


    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    features_df[
        "hybrid_score"
    ] = (

        MODEL_WEIGHT
        *
        features_df[
            "model_probability"
        ]

        +

        VISUAL_WEIGHT
        *
        features_df[
            "visual_preference_score"
        ]
    )


    # --------------------------------------------------------
    # ONLY NEW FRAMES
    # --------------------------------------------------------

    recommendations = features_df[
        ~features_df[
            "frame_id"
        ].isin(
            interacted_frames
        )
    ].copy()


    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    recommendations = (
        recommendations
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Add catalogue details
    # --------------------------------------------------------

    recommendations = (
        recommendations.merge(
            catalogue[
                [
                    "frame_id",
                    "brand",
                    "frame_shape",
                    "frame_color",
                    "style",
                    "frame_type",
                    "material",
                ]
            ],
            on="frame_id",
            how="left"
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n==============================================")
    print("TOP PERSONALIZED RECOMMENDATIONS")
    print("==============================================")


    if recommendations.empty:

        print(
            "No unseen frames available."
        )

    else:

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            print(
                f"\n#{rank} {row['frame_id']}"
            )

            print(
                f"Brand: {row['brand']}"
            )

            print(
                f"Shape: {row['frame_shape']}"
            )

            print(
                f"Color: {row['frame_color']}"
            )

            print(
                f"Style: {row['style']}"
            )

            print(
                f"Face compatibility: "
                f"{row['face_compatibility']:.4f}"
            )

            print(
                f"Visual preference: "
                f"{row['visual_preference_score']:.4f}"
            )

            print(
                f"ML probability: "
                f"{row['model_probability']:.4f}"
            )

            print(
                f"Hybrid score: "
                f"{row['hybrid_score']:.4f}"
            )


    print("\n==============================================")
    print("RECOMMENDATION COMPLETE")
    print("==============================================")


    print(
        f"Candidates evaluated: "
        f"{len(features_df)}"
    )

    print(
        f"Previously interacted: "
        f"{len(interacted_frames)}"
    )

    print(
        f"New recommendations: "
        f"{len(recommendations)}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
