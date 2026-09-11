from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


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
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(
        FEATURES_PATH
    )

    train_df = df[
        df["split"] == "training_set"
    ].copy()

    test_df = df[
        df["split"] == "testing_set"
    ].copy()

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "label"
    ]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = model.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # Confusion matrix
    labels = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\nConfusion Matrix:")

    print(matrix)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "labels": labels,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("EYewear AI - GEOMETRY BASELINE EXPERIMENT")
    print("=" * 70)

    if not FEATURES_PATH.exists():

        print(
            "\nERROR: Feature dataset not found:"
        )

        print(
            FEATURES_PATH
        )

        return

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_data()

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )

    print(
        f"Features        : {len(FEATURE_COLUMNS)}"
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": Pipeline([
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
        ]),
    }

    # -----------------------------------------------------
    # Train and evaluate
    # -----------------------------------------------------

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(result)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    summary = pd.DataFrame([
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
        for result in results
    ])

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # -----------------------------------------------------
    # Confusion matrices
    # -----------------------------------------------------

    for result in results:

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            result["confusion_matrix"]
        )

        plt.title(
            f"Confusion Matrix - "
            f"{result['model']}"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.xticks(
            range(len(result["labels"])),
            result["labels"],
            rotation=45
        )

        plt.yticks(
            range(len(result["labels"])),
            result["labels"]
        )

        for row in range(
            len(result["labels"])
        ):

            for column in range(
                len(result["labels"])
            ):

                plt.text(
                    column,
                    row,
                    result["confusion_matrix"][
                        row,
                        column
                    ],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        filename = (
            result["model"]
            .lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=150
        )

        plt.close()

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()