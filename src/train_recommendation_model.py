import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/recommendations/synthetic_interactions.csv"
)

MODEL_DIR = Path("models/recommendation")
MODEL_PATH = MODEL_DIR / "logistic_recommendation_model.joblib"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n==============================================")
print("TRAINING RECOMMENDATION MODEL")
print("==============================================")

print(f"Dataset size: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

features = [
    "face_compatibility",
    "style_match",
    "color_match",
    "frame_type_match"
]

target = "liked"


X = df[features]
y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n==============================================")
print("MODEL EVALUATION")
print("==============================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Dislike",
            "Like"
        ]
    )
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

classifier = model.named_steps["classifier"]

print("\nFeature coefficients:")

for feature, coefficient in zip(
    features,
    classifier.coef_[0]
):
    print(
        f"{feature:20s}: {coefficient:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


print("\n==============================================")
print("MODEL SAVED")
print("==============================================")

print(f"Location: {MODEL_PATH}")