from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CALIBRATED HYBRID RECOMMENDATION MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "synthetic_hybrid_interactions.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
)

MODEL_PATH = (
    MODEL_DIR
    / "calibrated_hybrid_model.joblib"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
    / "calibrated_model_comparison.csv"
)


FEATURE_COLUMNS = [
    "face_compatibility",
    "visual_preference_score",
    "style_match",
    "color_match",
    "frame_type_match",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

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

    train_df = df[
        df["user_id"].isin(train_users)
    ].copy()

    test_df = df[
        df["user_id"].isin(test_users)
    ].copy()

    return (
        train_df,
        test_df,
        train_users,
        test_users,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "log_loss": log_loss(
            y_test,
            probabilities
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities
        ),
    }

    return metrics, probabilities, predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CALIBRATED HYBRID RECOMMENDATION MODEL")
    print("=" * 72)

    train_df, test_df, train_users, test_users = (
        load_data()
    )

    print(
        f"\nTotal interactions: "
        f"{len(train_df) + len(test_df)}"
    )

    print(
        f"Training users: "
        f"{len(train_users)}"
    )

    print(
        f"Test users: "
        f"{len(test_users)}"
    )

    print(
        f"Training interactions: "
        f"{len(train_df)}"
    )

    print(
        f"Test interactions: "
        f"{len(test_df)}"
    )

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "liked"
    ].astype(int)

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "liked"
    ].astype(int)

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    base_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    # --------------------------------------------------------
    # Calibration
    #
    # ensemble=False:
    #   CV predictions are generated without using the sample
    #   to fit the corresponding base classifier.
    #
    # This avoids fitting the calibrator directly on the same
    # predictions that were generated from training data.
    # --------------------------------------------------------

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
        ensemble=False,
    )

    print(
        "\nTraining calibrated model..."
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate on locked test users
    # --------------------------------------------------------

    metrics, probabilities, predictions = (
        evaluate_model(
            calibrated_model,
            X_test,
            y_test,
        )
    )

    print("\n" + "=" * 72)
    print("HELD-OUT TEST RESULTS")
    print("=" * 72)

    for name, value in metrics.items():

        print(
            f"{name:15s}: {value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # Probability diagnostics
    # --------------------------------------------------------

    print("\nProbability range:")

    print(
        f"Minimum: {probabilities.min():.4f}"
    )

    print(
        f"Maximum: {probabilities.max():.4f}"
    )

    print(
        f"Mean   : {probabilities.mean():.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": calibrated_model,
            "feature_columns": FEATURE_COLUMNS,
            "description": (
                "Calibrated hybrid recommendation model "
                "trained on synthetic development interactions. "
                "Evaluation uses user-disjoint held-out users."
            ),
            "training_users": train_users,
            "test_users": test_users,
        },
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    comparison = pd.DataFrame([
        {
            "model": "Calibrated Hybrid",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
            "brier_score": metrics["brier_score"],
            "probability_min": probabilities.min(),
            "probability_max": probabilities.max(),
            "probability_mean": probabilities.mean(),
        }
    ])

    comparison.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\n" + "=" * 72)

    print(
        f"Model saved: {MODEL_PATH.resolve()}"
    )

    print(
        f"Metrics saved: {RESULT_PATH.resolve()}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
