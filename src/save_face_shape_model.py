from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# PATHS
# =========================================================

FEATURES_PATH = Path(
    "data/processed/faces/facial_features.csv"
)

MODEL_DIR = Path(
    "models/face_shape"
)

MODEL_PATH = MODEL_DIR / "geometry_svm.joblib"


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "face_width",
    "face_height",
    "face_aspect_ratio",
    "forehead_width",
    "cheek_width",
    "jaw_width",
    "forehead_to_face_ratio",
    "cheek_to_face_ratio",
    "jaw_to_face_ratio",
]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SAVING FACE SHAPE GEOMETRY MODEL")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    df = pd.read_csv(FEATURES_PATH)

    train_df = df[
        df["split"] == "training_set"
    ].copy()

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "label"
    ]

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Classes: {sorted(y_train.unique())}"
    )

    # Same SVM configuration as the evaluated baseline.
    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        ),
    ])

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": sorted(y_train.unique().tolist()),
            "training_samples": len(X_train),
            "description": (
                "Geometry-based face shape classifier "
                "trained on the training split only."
            ),
        },
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH.resolve())

    # -----------------------------------------------------
    # Verify saved model
    # -----------------------------------------------------

    loaded = joblib.load(
        MODEL_PATH
    )

    print("\nVerification:")

    print(
        f"Model type: "
        f"{type(loaded['model']).__name__}"
    )

    print(
        f"Feature count: "
        f"{len(loaded['feature_columns'])}"
    )

    print(
        f"Classes: "
        f"{loaded['classes']}"
    )

    print("\nFace shape model is ready.")


if __name__ == "__main__":
    main()
