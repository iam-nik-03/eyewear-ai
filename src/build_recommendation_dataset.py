import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CATALOGUE_PATH = Path("data/catalogue/eyewear_catalogue.csv")
PREFERENCES_PATH = Path("data/catalogue/user_preferences.csv")
FEEDBACK_PATH = Path("data/catalogue/feedback.csv")

OUTPUT_DIR = Path("data/processed/recommendations")
OUTPUT_PATH = OUTPUT_DIR / "recommendation_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

catalogue = pd.read_csv(CATALOGUE_PATH)
preferences = pd.read_csv(PREFERENCES_PATH)
feedback = pd.read_csv(FEEDBACK_PATH)


print("\n==============================================")
print("BUILDING RECOMMENDATION DATASET")
print("==============================================")

print(f"Catalogue rows: {len(catalogue)}")
print(f"Preference rows: {len(preferences)}")
print(f"Feedback rows: {len(feedback)}")


# ============================================================
# MERGE USER PREFERENCES WITH FEEDBACK
# ============================================================

dataset = feedback.merge(
    preferences,
    on="user_id",
    how="left"
)

# Add frame information
dataset = dataset.merge(
    catalogue,
    on="frame_id",
    how="left"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Face-shape compatibility baseline
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


def get_face_compatibility(face_shape, frame_shape):
    return face_compatibility_rules.get(
        face_shape,
        {}
    ).get(
        frame_shape,
        0.0
    )


dataset["face_compatibility"] = dataset.apply(
    lambda row: get_face_compatibility(
        row["face_shape"],
        row["frame_shape"]
    ),
    axis=1
)
# Style preference match
dataset["style_match"] = (
    dataset["preferred_style"].str.lower()
    == dataset["style"].str.lower()
).astype(int)


# Colour preference match
dataset["color_match"] = (
    dataset["preferred_color"].str.lower()
    .apply(
        lambda preferred: False
    )
)

# More flexible colour matching
dataset["color_match"] = dataset.apply(
    lambda row:
        1
        if str(row["preferred_color"]).lower()
        in str(row["frame_color"]).lower()
        else 0,
    axis=1
)


# Frame type match
dataset["frame_type_match"] = (
    dataset["preferred_frame_type"].str.lower()
    == dataset["frame_type"].str.lower()
).astype(int)


# Budget match
dataset["budget_match"] = dataset.apply(
    lambda row:
        1
        if pd.notna(row["price"])
        and row["price"] <= row["budget"]
        else 0.5
        if pd.isna(row["price"])
        else 0,
    axis=1
)


# ============================================================
# SELECT ML FEATURES
# ============================================================

final_columns = [
    "user_id",
    "frame_id",
    "preferred_style",
    "preferred_color",
    "preferred_frame_type",
    "budget",
    "face_shape",
    "frame_shape",
    "face_compatibility",
    "style_match",
    "color_match",
    "frame_type_match",
    "budget_match",
    "liked"
]




dataset = dataset[final_columns]


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

dataset.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\nGenerated dataset:")
print(dataset.to_string(index=False))

print("\n==============================================")
print("DATASET SAVED")
print("==============================================")
print(f"Location: {OUTPUT_PATH}")
print(f"Rows: {len(dataset)}")
print(f"Columns: {len(dataset.columns)}")