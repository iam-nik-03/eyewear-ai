import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ============================================================
# EYewear AI - IMPROVED GEOMETRY BASELINE
# ============================================================

DATA_PATH = "data/processed/faces/facial_features.csv"

FEATURES = [
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

print("=" * 70)
print("EYewear AI - IMPROVED GEOMETRY BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded:")
print(f"Total images: {len(df)}")


# ------------------------------------------------------------
# 2. Use TRAINING SET only
# ------------------------------------------------------------

train_df = df[df["split"] == "training_set"].copy()

X = train_df[FEATURES]
y = train_df["label"]

print(f"Training images: {len(train_df)}")

print("\nClass distribution:")
print(y.value_counts().sort_index())


# ------------------------------------------------------------
# 3. Define models
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),

    "SVM RBF": Pipeline([
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
}


# ------------------------------------------------------------
# 4. 5-Fold Cross Validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "-" * 70)
print("5-FOLD CROSS-VALIDATION")
print("-" * 70)

results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append({
        "model": name,
        "mean_accuracy": mean_score,
        "std": std_score
    })

    print(f"\n{name}")
    print("-" * 40)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print(f"Mean Accuracy: {mean_score:.4f}")
    print(f"Std Dev:       {std_score:.4f}")


# ------------------------------------------------------------
# 5. Compare models
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "mean_accuracy": "{:.4f}".format,
            "std": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------

output_path = (
    "data/processed/faces/baseline/"
    "geometry_cross_validation.csv"
)

import os

os.makedirs(
    "data/processed/faces/baseline",
    exist_ok=True
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)

print(f"\nResults saved to:")
print(output_path)

print("\nIMPORTANT:")
print("The testing set was NOT used.")
print("It remains completely untouched for final evaluation.")