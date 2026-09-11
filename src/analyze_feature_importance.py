from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PATHS
# =========================================================

FEATURES_PATH = Path(
    "data/processed/faces/facial_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/faces/baseline"
)


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
    print("EYewear AI - FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        FEATURES_PATH
    )

    train_df = df[
        df["split"] == "training_set"
    ]

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "label"
    ]

    # -----------------------------------------------------
    # Train Random Forest
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance = pd.DataFrame({

        "feature":
            FEATURE_COLUMNS,

        "importance":
            model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE")
    print("-" * 70)

    print(
        importance.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Facial Geometry Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nSaved:")

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()