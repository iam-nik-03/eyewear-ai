import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# EYewear AI - VISUAL EMBEDDING BASELINE
# ============================================================

EMBEDDING_PATH = (
    "data/processed/faces/visual_embeddings/embeddings.npy"
)

METADATA_PATH = (
    "data/processed/faces/visual_embeddings/metadata.csv"
)

OUTPUT_DIR = "data/processed/faces/baseline"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("EYewear AI - VISUAL EMBEDDING BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load embeddings and metadata
# ------------------------------------------------------------

X_all = np.load(EMBEDDING_PATH)

metadata = pd.read_csv(METADATA_PATH)

print("\nEmbedding shape:")
print(X_all.shape)

print("\nMetadata rows:")
print(len(metadata))


# ------------------------------------------------------------
# 2. Split training and testing data
# ------------------------------------------------------------

train_mask = metadata["split"] == "training_set"
test_mask = metadata["split"] == "testing_set"

X_train = X_all[train_mask.values]
X_test = X_all[test_mask.values]

y_train = metadata.loc[train_mask, "label"].values
y_test = metadata.loc[test_mask, "label"].values


print("\nTraining:")
print(X_train.shape)

print("\nTesting:")
print(X_test.shape)


# ------------------------------------------------------------
# 3. Logistic Regression
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training Logistic Regression...")
print("-" * 70)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)


# ------------------------------------------------------------
# 4. SVM
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training SVM...")
print("-" * 70)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42
        )
    )
])

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)


# ------------------------------------------------------------
# 5. Evaluation function
# ------------------------------------------------------------

def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ------------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------------

results = []

results.append(
    evaluate_model(
        "Visual Logistic Regression",
        y_test,
        logistic_predictions
    )
)

results.append(
    evaluate_model(
        "Visual SVM",
        y_test,
        svm_predictions
    )
)


# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "visual_baseline_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 70)
print("VISUAL BASELINE COMPLETE")
print("=" * 70)

print("\nModel comparison:")
print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format
        }
    )
)

print(f"\nResults saved to:")
print(results_path)