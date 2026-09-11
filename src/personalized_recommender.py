import pandas as pd

# --------------------------------------------------
# Files
# --------------------------------------------------

CATALOGUE_FILE = "data/catalogue/eyewear_catalogue.csv"
PREFERENCES_FILE = "data/catalogue/user_preferences.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

catalogue = pd.read_csv(CATALOGUE_FILE)
preferences = pd.read_csv(PREFERENCES_FILE)


# --------------------------------------------------
# Face → Frame compatibility baseline
# --------------------------------------------------

COMPATIBILITY_RULES = {

    "Heart": {
        "Aviator": 1.00,
        "Round": 0.90,
        "Cat Eye": 0.80,
        "Rectangle": 0.70,
        "Hexagon": 0.60,
        "Pentagon": 0.60,
    },

    "Oblong": {
        "Round": 1.00,
        "Aviator": 0.90,
        "Cat Eye": 0.80,
        "Rectangle": 0.70,
        "Hexagon": 0.70,
        "Pentagon": 0.70,
    },

    "Oval": {
        "Rectangle": 1.00,
        "Aviator": 0.90,
        "Round": 0.80,
        "Hexagon": 0.80,
        "Pentagon": 0.80,
        "Cat Eye": 0.80,
    },

    "Round": {
        "Rectangle": 1.00,
        "Hexagon": 0.90,
        "Pentagon": 0.85,
        "Aviator": 0.80,
        "Cat Eye": 0.70,
        "Round": 0.60,
    },

    "Square": {
        "Round": 1.00,
        "Aviator": 0.90,
        "Cat Eye": 0.80,
        "Hexagon": 0.80,
        "Rectangle": 0.70,
        "Pentagon": 0.70,
    }
}


# --------------------------------------------------
# Style preference
# --------------------------------------------------

def style_score(frame, user):

    preferred_style = str(
        user["preferred_style"]
    ).strip().lower()

    frame_style = str(
        frame["style"]
    ).strip().lower()

    return 1.0 if preferred_style == frame_style else 0.0


# --------------------------------------------------
# Colour preference
# --------------------------------------------------

def color_score(frame, user):

    preferred_color = str(
        user["preferred_color"]
    ).strip().lower()

    frame_color = str(
        frame["frame_color"]
    ).strip().lower()

    if preferred_color in frame_color:
        return 1.0

    return 0.0


# --------------------------------------------------
# Frame type preference
# --------------------------------------------------

def frame_type_score(frame, user):

    preferred_type = str(
        user["preferred_frame_type"]
    ).strip().lower()

    frame_type = str(
        frame["frame_type"]
    ).strip().lower()

    return 1.0 if preferred_type == frame_type else 0.0


# --------------------------------------------------
# Budget score
# --------------------------------------------------

def budget_score(frame, user):

    budget = float(user["budget"])

    price = frame["price"]

    # Price is currently unavailable
    if pd.isna(price):
        return 0.5

    price = float(price)

    if price <= budget:
        return 1.0

    return 0.0


# --------------------------------------------------
# Recommendation
# --------------------------------------------------

def recommend_frames(
    user_id,
    face_shape,
    top_k=5
):

    face_shape = face_shape.strip().title()

    if face_shape not in COMPATIBILITY_RULES:
        raise ValueError(
            f"Unknown face shape: {face_shape}"
        )

    # Find user
    user_rows = preferences[
        preferences["user_id"] == user_id
    ]

    if user_rows.empty:
        raise ValueError(
            f"User not found: {user_id}"
        )

    user = user_rows.iloc[0]

    results = []

    for _, frame in catalogue.iterrows():

        # -----------------------------
        # 1. Face compatibility
        # -----------------------------

        face_score = COMPATIBILITY_RULES[
            face_shape
        ].get(
            frame["frame_shape"],
            0.50
        )

        # -----------------------------
        # 2. Style
        # -----------------------------

        style = style_score(
            frame,
            user
        )

        # -----------------------------
        # 3. Colour
        # -----------------------------

        color = color_score(
            frame,
            user
        )

        # -----------------------------
        # 4. Frame type
        # -----------------------------

        frame_type = frame_type_score(
            frame,
            user
        )

        # -----------------------------
        # 5. Budget
        # -----------------------------

        budget = budget_score(
            frame,
            user
        )

        # -----------------------------
        # Final score
        # -----------------------------

        final_score = (

            0.50 * face_score

            + 0.20 * style

            + 0.15 * color

            + 0.10 * frame_type

            + 0.05 * budget
        )

        results.append({

            "frame_id": frame["frame_id"],

            "brand": frame["brand"],

            "frame_shape": frame["frame_shape"],

            "frame_color": frame["frame_color"],

            "style": frame["style"],

            "face_score": round(
                face_score, 3
            ),

            "style_score": round(
                style, 3
            ),

            "color_score": round(
                color, 3
            ),

            "frame_type_score": round(
                frame_type, 3
            ),

            "budget_score": round(
                budget, 3
            ),

            "final_score": round(
                final_score, 3
            )
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "final_score",
        ascending=False
    )

    return results.head(top_k)


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("\n==============================================")
    print("PERSONALIZED EYEWEAR RECOMMENDER")
    print("==============================================")

    print("\nUser:")
    print("  Face shape: Round")
    print("  Style: Casual")
    print("  Colour: Black")
    print("  Frame type: Optical")
    print("  Budget: ₹3000")

    recommendations = recommend_frames(
        user_id="USER_001",
        face_shape="Round",
        top_k=5
    )

    print("\nRecommendations:")

    print(
        recommendations.to_string(
            index=False
        )
    )