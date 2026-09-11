import pandas as pd

# --------------------------------------------------
# Load eyewear catalogue
# --------------------------------------------------

CATALOGUE_FILE = "data/catalogue/eyewear_catalogue.csv"

catalogue = pd.read_csv(CATALOGUE_FILE)

# --------------------------------------------------
# Baseline compatibility rules
# --------------------------------------------------
#
# IMPORTANT:
# This is a baseline only.
# Later we will replace these hand-written rules
# with a learned ranking model.
#

COMPATIBILITY_RULES = {
    "Heart": [
        "Aviator",
        "Round",
        "Cat Eye",
        "Rectangle"
    ],

    "Oblong": [
        "Round",
        "Aviator",
        "Cat Eye",
        "Rectangle"
    ],

    "Oval": [
        "Rectangle",
        "Square",
        "Aviator",
        "Round"
    ],

    "Round": [
        "Rectangle",
        "Hexagon",
        "Pentagon",
        "Aviator"
    ],

    "Square": [
        "Round",
        "Aviator",
        "Cat Eye",
        "Oval"
    ]
}


def recommend_frames(face_shape, top_k=5):

    face_shape = face_shape.strip().title()

    if face_shape not in COMPATIBILITY_RULES:
        raise ValueError(
            f"Unknown face shape: {face_shape}"
        )

    preferred_shapes = COMPATIBILITY_RULES[face_shape]

    recommendations = []

    for priority, frame_shape in enumerate(preferred_shapes):

        matches = catalogue[
            catalogue["frame_shape"].str.lower()
            == frame_shape.lower()
        ]

        for _, frame in matches.iterrows():

            recommendations.append({
                "frame_id": frame["frame_id"],
                "brand": frame["brand"],
                "frame_shape": frame["frame_shape"],
                "frame_color": frame["frame_color"],
                "compatibility_priority": priority + 1
            })

    return pd.DataFrame(recommendations).head(top_k)


# --------------------------------------------------
# Test the recommender
# --------------------------------------------------

if __name__ == "__main__":

    test_face_shapes = [
        "Heart",
        "Oblong",
        "Oval",
        "Round",
        "Square"
    ]

    for face_shape in test_face_shapes:

        print("\n" + "=" * 50)
        print(f"Recommendations for {face_shape} face")
        print("=" * 50)

        results = recommend_frames(face_shape)

        if results.empty:
            print("No matching frames found.")

        else:
            print(results.to_string(index=False))