from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


CATALOGUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "eyewear_catalogue.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "hybrid_model.joblib"
)


VISUAL_PROFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "user_visual_profile.csv"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "real_face_recommendations.csv"
)


# ============================================================
# FACE SHAPE → FRAME SHAPE COMPATIBILITY
# ============================================================

COMPATIBILITY = {
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


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "face_compatibility",
    "visual_preference_score",
    "style_match",
    "color_match",
    "frame_type_match",
]


MODEL_WEIGHT = 0.70
VISUAL_WEIGHT = 0.30


# ============================================================
# LOAD CATALOGUE
# ============================================================

def load_catalogue():
    """Load the eyewear catalogue."""

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Eyewear catalogue not found:\n{CATALOGUE_PATH}"
        )

    catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    if catalogue.empty:
        raise ValueError(
            "Eyewear catalogue is empty."
        )

    return catalogue


# ============================================================
# LOAD MODEL
# ============================================================

def load_recommendation_model():
    """Load the trained hybrid recommendation model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Recommendation model not found:\n{MODEL_PATH}"
        )

    saved_model = joblib.load(
        MODEL_PATH
    )

    return saved_model


# ============================================================
# FACE COMPATIBILITY
# ============================================================

def calculate_weighted_face_compatibility(
    frame_shape,
    probabilities,
):
    """
    Calculate expected face/frame compatibility.

    Instead of assuming one face shape with 100% certainty,
    the score integrates the complete face-shape probability
    distribution.
    """

    score = 0.0

    for (
        face_shape,
        probability,
    ) in probabilities.items():

        frame_scores = COMPATIBILITY.get(
            face_shape,
            {},
        )

        compatibility = frame_scores.get(
            frame_shape,
            0.0,
        )

        score += (
            probability
            * compatibility
        )

    return float(score)


# ============================================================
# NORMALIZE PROBABILITIES
# ============================================================

def normalize_probabilities(
    probabilities,
):
    """Normalize face-shape probabilities defensively."""

    cleaned = {
        str(shape): float(probability)
        for shape, probability
        in probabilities.items()
    }

    total = sum(
        cleaned.values()
    )

    if total <= 0:
        raise ValueError(
            "Invalid face-shape probabilities."
        )

    return {
        shape: probability / total
        for shape, probability
        in cleaned.items()
    }


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    catalogue,
    probabilities,
    user_preferences,
):
    """
    Build recommendation features for every frame.
    """

    preferred_style = str(
        user_preferences.get(
            "preferred_style",
            "",
        )
    )

    preferred_color = str(
        user_preferences.get(
            "preferred_color",
            "",
        )
    )

    preferred_type = str(
        user_preferences.get(
            "preferred_frame_type",
            "",
        )
    )

    rows = []

    for _, frame in catalogue.iterrows():

        frame_shape = str(
            frame["frame_shape"]
        )

        face_compatibility = (
            calculate_weighted_face_compatibility(
                frame_shape,
                probabilities,
            )
        )

        style_match = int(
            str(
                frame["style"]
            ).lower()
            == preferred_style.lower()
        )

        color_match = int(
            preferred_color.lower()
            in str(
                frame["frame_color"]
            ).lower()
        )

        frame_type_match = int(
            str(
                frame["frame_type"]
            ).lower()
            == preferred_type.lower()
        )

        rows.append(
            {
                "frame_id": frame["frame_id"],
                "face_compatibility": (
                    face_compatibility
                ),
                "style_match": style_match,
                "color_match": color_match,
                "frame_type_match": (
                    frame_type_match
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VISUAL PREFERENCE
# ============================================================

def add_visual_preference(
    features,
    visual_profile_path=VISUAL_PROFILE_PATH,
):
    """
    Add the existing user visual-preference signal.

    If no visual profile exists, use zero rather than
    inventing a preference.
    """

    features = features.copy()

    if visual_profile_path.exists():

        visual_profile = pd.read_csv(
            visual_profile_path
        )

        if (
            "frame_id" in visual_profile.columns
            and
            "visual_preference_score"
            in visual_profile.columns
        ):

            visual_map = dict(
                zip(
                    visual_profile[
                        "frame_id"
                    ],
                    visual_profile[
                        "visual_preference_score"
                    ],
                )
            )

            features[
                "visual_preference_score"
            ] = (
                features["frame_id"]
                .map(visual_map)
                .fillna(0.0)
            )

        else:

            features[
                "visual_preference_score"
            ] = 0.0

    else:

        features[
            "visual_preference_score"
        ] = 0.0

    return features


# ============================================================
# RECOMMEND
# ============================================================

def recommend_from_face_analysis(
    face_analysis,
    user_preferences,
):
    """
    Generate eyewear recommendations from a real
    face-analysis result.

    Parameters
    ----------
    face_analysis:
        Result returned by analyze_image().

    user_preferences:
        Dictionary containing:
            preferred_style
            preferred_color
            preferred_frame_type
            budget

    Returns
    -------
    pandas.DataFrame
        Ranked eyewear recommendations.
    """

    # --------------------------------------------------------
    # Validate face result
    # --------------------------------------------------------

    if not face_analysis:
        raise ValueError(
            "Face analysis result is empty."
        )

    if "probabilities" not in face_analysis:
        raise ValueError(
            "Face analysis does not contain probabilities."
        )

    probabilities = normalize_probabilities(
        face_analysis["probabilities"]
    )

    # --------------------------------------------------------
    # Load assets
    # --------------------------------------------------------

    catalogue = load_catalogue()

    model = load_recommendation_model()

    # --------------------------------------------------------
    # Build metadata features
    # --------------------------------------------------------

    features = build_features(
        catalogue,
        probabilities,
        user_preferences,
    )

    # --------------------------------------------------------
    # Add visual preference
    # --------------------------------------------------------

    features = add_visual_preference(
        features
    )

    # --------------------------------------------------------
    # Exact model schema
    # --------------------------------------------------------

    X = features[
        MODEL_FEATURES
    ]

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    features[
        "model_probability"
    ] = model.predict_proba(
        X
    )[:, 1]

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    features[
        "hybrid_score"
    ] = (
        MODEL_WEIGHT
        * features[
            "model_probability"
        ]
        +
        VISUAL_WEIGHT
        * features[
            "visual_preference_score"
        ]
    )

    # --------------------------------------------------------
    # Attach catalogue
    # --------------------------------------------------------

    result = features.merge(
        catalogue,
        on="frame_id",
        how="left",
    )

    # --------------------------------------------------------
    # Budget filter
    # --------------------------------------------------------

    budget = user_preferences.get(
        "budget"
    )

    if (
        budget is not None
        and "price" in result.columns
    ):

        numeric_price = pd.to_numeric(
            result["price"],
            errors="coerce",
        )

        # Only filter when catalogue prices
        # actually contain usable values.
        if numeric_price.notna().any():

            result = result[
                (
                    numeric_price
                    <= float(budget)
                )
                |
                numeric_price.isna()
            ].copy()

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    result = result.sort_values(
        "hybrid_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_recommendations(
    recommendations,
    output_path=OUTPUT_PATH,
):
    """Save recommendations to CSV."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "AURA — REAL FACE RECOMMENDATION ENGINE"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Load previously generated face analysis
    # --------------------------------------------------------

    face_analysis_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "faces"
        / "user_face_analysis.csv"
    )

    if not face_analysis_path.exists():

        raise FileNotFoundError(
            "Run src/analyze_user_face.py first."
        )

    face_df = pd.read_csv(
        face_analysis_path
    )

    if face_df.empty:

        raise ValueError(
            "Face analysis file is empty."
        )

    row = face_df.iloc[0]

    # --------------------------------------------------------
    # Reconstruct probability dictionary
    # --------------------------------------------------------

    probabilities = {}

    for shape in COMPATIBILITY.keys():

        column = (
            "probability_"
            + shape.lower()
        )

        if column in face_df.columns:

            probabilities[
                shape
            ] = float(
                row[column]
            )

    # --------------------------------------------------------
    # Face-analysis structure
    # --------------------------------------------------------

    face_analysis = {
        "image": row["image"],
        "predicted_face_shape": (
            row["predicted_face_shape"]
        ),
        "confidence": float(
            row["confidence"]
        ),
        "probabilities": probabilities,
    }

    # --------------------------------------------------------
    # Default USER_002 profile
    # --------------------------------------------------------

    user_preferences = {
        "preferred_style": "Casual",
        "preferred_color": "Black",
        "preferred_frame_type": "Optical",
        "budget": 3000,
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = recommend_from_face_analysis(
        face_analysis,
        user_preferences,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"\nPredicted face shape: "
        f"{face_analysis['predicted_face_shape']}"
    )

    print(
        f"Confidence: "
        f"{face_analysis['confidence']:.4f}"
    )

    print(
        "\nRecommendations:"
    )

    for _, row in result.iterrows():

        print(
            "\n"
            + "-" * 65
        )

        print(
            f"#{int(row['rank'])} "
            f"{row['frame_id']}"
        )

        print(
            f"Brand: "
            f"{row['brand']}"
        )

        print(
            f"Shape: "
            f"{row['frame_shape']}"
        )

        print(
            f"Colour: "
            f"{row['frame_color']}"
        )

        print(
            f"Style: "
            f"{row['style']}"
        )

        print(
            f"Face compatibility: "
            f"{row['face_compatibility']:.4f}"
        )

        print(
            f"ML preference: "
            f"{row['model_probability']:.4f}"
        )

        print(
            f"Visual preference: "
            f"{row['visual_preference_score']:.4f}"
        )

        print(
            f"Hybrid score: "
            f"{row['hybrid_score']:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saved_path = save_recommendations(
        result
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "Recommendation generation completed."
    )

    print(
        f"Saved: {saved_path.resolve()}"
    )

    print(
        "=" * 72
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()