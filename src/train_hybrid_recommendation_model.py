import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/recommendations/"
    "synthetic_hybrid_interactions.csv"
)

MODEL_DIR = Path(
    "models/recommendation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Make user IDs consistent strings
df["user_id"] = df["user_id"].astype(str)


print("\n==============================================")
print("HYBRID RECOMMENDATION MODEL TRAINING")
print("==============================================")

print(
    f"Total interactions: {len(df)}"
)

print(
    f"Total users: {df['user_id'].nunique()}"
)


# ============================================================
# USER-DISJOINT TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# The same user must NOT appear in both
# training and testing data.
#
# This prevents user-level leakage.
# ============================================================

unique_users = (
    df["user_id"]
    .drop_duplicates()
    .tolist()
)

print(
    f"Unique users detected: {len(unique_users)}"
)


train_users, test_users = train_test_split(
    unique_users,
    test_size=0.20,
    random_state=42
)


# Create training dataset
train_df = df[
    df["user_id"].isin(train_users)
].copy()


# Create testing dataset
test_df = df[
    df["user_id"].isin(test_users)
].copy()


print(
    f"Training users: {len(train_users)}"
)

print(
    f"Test users: {len(test_users)}"
)

print(
    f"Training interactions: {len(train_df)}"
)

print(
    f"Test interactions: {len(test_df)}"
)


# ============================================================
# FEATURES
# ============================================================

# Model A:
# Only traditional recommendation features

metadata_features = [
    "face_compatibility",
    "style_match",
    "color_match",
    "frame_type_match",
]


# Model B:
# Traditional features + visual preference

hybrid_features = [
    "face_compatibility",
    "visual_preference_score",
    "style_match",
    "color_match",
    "frame_type_match",
]


# Target
target = "liked"


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    features,
    model_name
):

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = train_df[features]
    y_train = train_df[target]


    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    X_test = test_df[features]
    y_test = test_df[target]


    # --------------------------------------------------------
    # Model pipeline
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # Probability of "Like"
    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n----------------------------------------------")
    print(model_name)
    print("----------------------------------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
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

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )


    model_path = (
        MODEL_DIR /
        model_filename
    )


    joblib.dump(
        model,
        model_path
    )


    print(
        f"\nSaved model: {model_path}"
    )


    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# MODEL A — METADATA ONLY
# ============================================================

metadata_result = train_model(
    metadata_features,
    "Metadata Model"
)


# ============================================================
# MODEL B — HYBRID
# ============================================================

hybrid_result = train_model(
    hybrid_features,
    "Hybrid Model"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    metadata_result,
    hybrid_result
])


comparison_path = (
    Path(
        "data/processed/recommendations"
    )
    / "hybrid_model_comparison.csv"
)


comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")


print(
    comparison.to_string(
        index=False
    )
)


print(
    f"\nComparison saved to:"
    f"\n{comparison_path}"
)


print("\n==============================================")
print("TRAINING COMPLETE")
print("==============================================")