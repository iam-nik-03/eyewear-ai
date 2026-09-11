from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "synthetic_hybrid_interactions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "calibrated_hybrid_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
)


FEATURE_COLUMNS = [
    "face_compatibility",
    "visual_preference_score",
    "style_match",
    "color_match",
    "frame_type_match",
]


def main():

    print("=" * 72)
    print("RECOMMENDATION PROBABILITY CALIBRATION EVALUATION")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)

    df["user_id"] = df["user_id"].astype(str)

    unique_users = (
        df["user_id"]
        .drop_duplicates()
        .tolist()
    )

    train_users, test_users = train_test_split(
        unique_users,
        test_size=0.20,
        random_state=42,
    )

    test_df = df[
        df["user_id"].isin(test_users)
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "liked"
    ].astype(int)

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    print(
        f"\nTest users: {len(test_users)}"
    )

    print(
        f"Test interactions: {len(test_df)}"
    )

    print(
        f"Brier score: {brier:.4f}"
    )

    # --------------------------------------------------------
    # Reliability curve
    # --------------------------------------------------------

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_test,
            probabilities,
            n_bins=5,
            strategy="quantile",
        )
    )

    print("\nCalibration bins:")

    for predicted, actual in zip(
        mean_predicted,
        fraction_positive,
    ):

        print(
            f"Predicted: {predicted:.4f} "
            f"Actual: {actual:.4f}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Calibrated model"
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed positive rate"
    )

    plt.title(
        "Recommendation Model Calibration"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "recommendation_calibration_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"\nCalibration curve saved:"
    )

    print(
        output_path.resolve()
    )

    print("\nCalibration evaluation completed.")


if __name__ == "__main__":
    main()
