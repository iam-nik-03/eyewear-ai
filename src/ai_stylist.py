from pathlib import Path

import pandas as pd


# ============================================================
# AI STYLIST - PERSONALIZED EXPLANATION ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOGUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "eyewear_catalogue.csv"
)

RECOMMENDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "latest_recommendations.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "stylist_recommendations.csv"
)

USER_PREFERENCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "user_preferences.csv"
)

USER_ID = "USER_002"


# ============================================================
# DATA LOADING
# ============================================================

def load_data(user_id=USER_ID):
    catalogue = pd.read_csv(CATALOGUE_PATH)
    recommendations = pd.read_csv(RECOMMENDATION_PATH)
    preferences = pd.read_csv(USER_PREFERENCES_PATH)

    user = preferences[
        preferences["user_id"].astype(str) == str(user_id)
    ]

    if user.empty:
        raise ValueError(
            f"No preferences found for {user_id}"
        )

    return (
        catalogue,
        recommendations,
        user.iloc[0].to_dict(),
    )


# ============================================================
# COMPATIBILITY
# ============================================================

def compatibility_level(score):

    score = float(score)

    if score >= 0.90:
        return "Excellent match"

    elif score >= 0.80:
        return "Strong match"

    elif score >= 0.70:
        return "Good alternative"

    else:
        return "Experimental option"


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def create_explanation(frame, user):

    face_shape = str(
        user["face_shape"]
    )

    preferred_style = str(
        user["preferred_style"]
    )

    preferred_color = str(
        user["preferred_color"]
    )

    preferred_type = str(
        user["preferred_frame_type"]
    )

    frame_shape = str(
        frame["frame_shape"]
    )

    frame_color = str(
        frame["frame_color"]
    )

    frame_style = str(
        frame["style"]
    )

    frame_type = str(
        frame["frame_type"]
    )

    face_score = float(
        frame.get(
            "face_compatibility",
            0,
        )
    )

    visual_score = float(
        frame.get(
            "visual_preference_score",
            0,
        )
    )

    model_probability = float(
        frame.get(
            "model_probability",
            0,
        )
    )

    hybrid_score = float(
        frame.get(
            "hybrid_score",
            0,
        )
    )

    level = compatibility_level(
        face_score
    )

    reasons = []

    # ========================================================
    # FACE COMPATIBILITY
    # ========================================================

    if face_score >= 0.90:

        reasons.append(
            f"The {frame_shape.lower()} shape is highly "
            f"compatible with your {face_shape.lower()} "
            f"face profile."
        )

    elif face_score >= 0.80:

        reasons.append(
            f"The {frame_shape.lower()} shape is a strong "
            f"match for your {face_shape.lower()} "
            f"face profile."
        )

    elif face_score >= 0.70:

        reasons.append(
            f"The {frame_shape.lower()} shape offers a good "
            f"alternative for your {face_shape.lower()} "
            f"face profile."
        )

    else:

        reasons.append(
            f"The {frame_shape.lower()} shape gives you a "
            f"more experimental styling option for your "
            f"{face_shape.lower()} face profile."
        )

    # ========================================================
    # COLOUR
    # ========================================================

    if preferred_color.lower() in frame_color.lower():

        reasons.append(
            f"The {frame_color.lower()} colour also fits "
            f"your preferred colour."
        )

    # ========================================================
    # FRAME TYPE
    # ========================================================

    if frame_type.lower() == preferred_type.lower():

        reasons.append(
            f"It matches your preferred "
            f"{frame_type.lower()} frame type."
        )

    # ========================================================
    # STYLE
    # ========================================================

    if frame_style.lower() == preferred_style.lower():

        reasons.append(
            f"Its {frame_style.lower()} styling matches "
            f"your usual style preference."
        )

    else:

        reasons.append(
            f"Its {frame_style.lower()} styling gives you "
            f"an alternative to your usual "
            f"{preferred_style.lower()} look."
        )

    # ========================================================
    # PERSONALIZATION
    # ========================================================

    if visual_score >= 0.90:

        reasons.append(
            "Your previous positive feedback indicates "
            "a strong visual similarity preference for "
            "this frame."
        )

    elif visual_score >= 0.80:

        reasons.append(
            "Its visual characteristics are similar to "
            "frames you have previously liked."
        )

    elif visual_score >= 0.70:

        reasons.append(
            "Its visual characteristics show some "
            "similarity to your previous preferences."
        )

    # ========================================================
    # ML MODEL
    # ========================================================

    if model_probability >= 0.90:

        reasons.append(
            "The recommendation model also gives it a "
            "very strong predicted preference score."
        )

    elif model_probability >= 0.75:

        reasons.append(
            "The recommendation model gives it a strong "
            "predicted preference score."
        )

    elif model_probability >= 0.50:

        reasons.append(
            "The recommendation model gives it a moderate "
            "predicted preference score."
        )

    return (
        level,
        " ".join(reasons),
        hybrid_score,
    )


# ============================================================
# GENERATE STYLIST RECOMMENDATIONS
# ============================================================

def generate_stylist_recommendations(
    recommendations,
    user,
    catalogue,
    user_id=USER_ID,
):
    output_rows = []

    for index, recommendation in (
        recommendations.iterrows()
    ):

        frame_id = str(
            recommendation["frame_id"]
        )

        matches = catalogue[
            catalogue["frame_id"].astype(str)
            == frame_id
        ]

        if matches.empty:
            continue

        frame = matches.iloc[
            0
        ].to_dict()

        # Combine catalogue information
        # with recommendation outputs.
        frame.update(
            recommendation.to_dict()
        )

        (
            level,
            explanation,
            hybrid_score,
        ) = create_explanation(
            frame,
            user,
        )

        output_rows.append(
            {
                "user_id": user_id,
                "rank": index + 1,
                "frame_id": frame_id,
                "brand": frame["brand"],
                "frame_shape": frame[
                    "frame_shape"
                ],
                "frame_color": frame[
                    "frame_color"
                ],
                "style": frame["style"],
                "frame_type": frame[
                    "frame_type"
                ],
                "hybrid_score": hybrid_score,
                "model_probability": float(
                    frame.get(
                        "model_probability",
                        0,
                    )
                ),
                "visual_preference_score": float(
                    frame.get(
                        "visual_preference_score",
                        0,
                    )
                ),
                "face_compatibility": float(
                    frame.get(
                        "face_compatibility",
                        0,
                    )
                ),
                "stylist_rating": level,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(
        output_rows
    )


# ============================================================
# GENERATE FROM SAVED RECOMMENDATIONS
# ============================================================

def generate_from_saved_recommendations(
    user_id=USER_ID,
):

    catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    recommendations = pd.read_csv(
        RECOMMENDATION_PATH
    )

    preferences = pd.read_csv(
        USER_PREFERENCES_PATH
    )

    user_rows = preferences[
        preferences["user_id"].astype(str)
        == str(user_id)
    ]

    if user_rows.empty:
        raise ValueError(
            f"No preferences found for {user_id}"
        )

    user = user_rows.iloc[
        0
    ].to_dict()

    output_df = (
        generate_stylist_recommendations(
            recommendations,
            user,
            catalogue,
            user_id,
        )
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    return output_df


# ============================================================
# CLI
# ============================================================

def main():

    print("=" * 72)
    print("AI STYLIST")
    print("=" * 72)

    (
        catalogue,
        recommendations,
        user,
    ) = load_data()

    print(
        f"\nUser: {USER_ID}"
    )

    print(
        f"Face profile: "
        f"{user['face_shape']}"
    )

    print(
        f"Preferred style: "
        f"{user['preferred_style']}"
    )

    print(
        f"Preferred colour: "
        f"{user['preferred_color']}"
    )

    print(
        f"Preferred frame type: "
        f"{user['preferred_frame_type']}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PERSONALIZED STYLE RECOMMENDATIONS"
    )

    print(
        "=" * 72
    )

    output_df = (
        generate_stylist_recommendations(
            recommendations,
            user,
            catalogue,
            USER_ID,
        )
    )

    for _, row in output_df.iterrows():

        print(
            "\n"
            + "-" * 72
        )

        print(
            f"#{int(row['rank'])} "
            f"{row['frame_id']}"
        )

        print(
            f"Brand          : "
            f"{row['brand']}"
        )

        print(
            f"Frame          : "
            f"{row['frame_shape']}"
        )

        print(
            f"Colour         : "
            f"{row['frame_color']}"
        )

        print(
            f"Style          : "
            f"{row['style']}"
        )

        print(
            f"Type           : "
            f"{row['frame_type']}"
        )

        print(
            f"Stylist rating : "
            f"{row['stylist_rating']}"
        )

        print(
            f"Hybrid score   : "
            f"{float(row['hybrid_score']):.4f}"
        )

        print(
            "\nWhy I recommend it:"
        )

        print(
            row["explanation"]
        )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "AI Stylist completed."
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()