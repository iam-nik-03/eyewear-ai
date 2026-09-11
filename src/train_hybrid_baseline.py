import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# EYewear AI - HYBRID GEOMETRY + VISUAL BASELINE
# ============================================================

EMBEDDING_PATH = (
    "data/processed/faces/visual_embeddings/embeddings.npy"
)

METADATA_PATH = (
    "data/processed/faces/visual_embeddings/metadata.csv"
)

GEOMETRY_PATH = (
    "data/processed/faces/facial_features.csv"
)

OUTPUT_DIR = "data/processed/faces/baseline"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


GEOMETRY_FEATURES = [
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
print("EYewear AI - HYBRID GEOMETRY + VISUAL BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load visual embeddings
# ------------------------------------------------------------

visual_embeddings = np.load(
    EMBEDDING_PATH
)

visual_metadata = pd.read_csv(
    METADATA_PATH
)

print("\nVisual embeddings:")
print(visual_embeddings.shape)

print("\nVisual metadata:")
print(len(visual_metadata))


# ------------------------------------------------------------
# 2. Load geometry features
# ------------------------------------------------------------

geometry_df = pd.read_csv(
    GEOMETRY_PATH
)

print("\nGeometry dataset:")
print(geometry_df.shape)


# ------------------------------------------------------------
# 3. Create matching key
# ------------------------------------------------------------

# Geometry dataset:
# image = filename
#
# Visual dataset:
# image_path = full path
#
# We create the same key in both datasets:
#
# split + label + filename
#
# This prevents accidental matching of similarly named
# files from different folders.


visual_metadata["image"] = (
    visual_metadata["image_path"]
    .apply(os.path.basename)
)


visual_metadata["match_key"] = (
    visual_metadata["split"]
    + "|"
    + visual_metadata["label"]
    + "|"
    + visual_metadata["image"]
)


geometry_df["match_key"] = (
    geometry_df["split"]
    + "|"
    + geometry_df["label"]
    + "|"
    + geometry_df["image"]
)


# ------------------------------------------------------------
# 4. Check for duplicate keys
# ------------------------------------------------------------

visual_duplicates = (
    visual_metadata["match_key"]
    .duplicated()
    .sum()
)

geometry_duplicates = (
    geometry_df["match_key"]
    .duplicated()
    .sum()
)


print("\nDuplicate visual keys:")
print(visual_duplicates)

print("\nDuplicate geometry keys:")
print(geometry_duplicates)


if visual_duplicates > 0:

    raise ValueError(
        "Duplicate keys found in visual metadata."
    )


if geometry_duplicates > 0:

    raise ValueError(
        "Duplicate keys found in geometry dataset."
    )


# ------------------------------------------------------------
# 5. Match geometry features
# ------------------------------------------------------------

geometry_lookup = geometry_df.set_index(
    "match_key"
)


geometry_rows = []

for key in visual_metadata["match_key"]:

    if key not in geometry_lookup.index:

        raise ValueError(
            f"Geometry features missing for: {key}"
        )

    geometry_rows.append(
        geometry_lookup.loc[
            key,
            GEOMETRY_FEATURES
        ].values
    )


geometry_features = np.array(
    geometry_rows,
    dtype=np.float32
)


print("\nGeometry features matched:")
print(geometry_features.shape)


# ------------------------------------------------------------
# 6. Verify labels and splits
# ------------------------------------------------------------

visual_labels = (
    visual_metadata["label"]
    .values
)

geometry_labels = np.array([
    geometry_lookup.loc[
        key,
        "label"
    ]
    for key in visual_metadata["match_key"]
])


if not np.array_equal(
    visual_labels,
    geometry_labels
):

    raise ValueError(
        "Label mismatch between visual and geometry datasets."
    )


visual_splits = (
    visual_metadata["split"]
    .values
)

geometry_splits = np.array([
    geometry_lookup.loc[
        key,
        "split"
    ]
    for key in visual_metadata["match_key"]
])


if not np.array_equal(
    visual_splits,
    geometry_splits
):

    raise ValueError(
        "Split mismatch between visual and geometry datasets."
    )


print("\nLabel alignment: VERIFIED")
print("Split alignment: VERIFIED")


# ------------------------------------------------------------
# 7. Combine features
# ------------------------------------------------------------

hybrid_features = np.hstack([
    geometry_features,
    visual_embeddings
])


print("\nHybrid feature shape:")
print(hybrid_features.shape)

print("\nExpected:")
print("5,000 images x 2,057 features")


# ------------------------------------------------------------
# 8. Train/test split
# ------------------------------------------------------------

train_mask = (
    visual_metadata["split"]
    == "training_set"
)

test_mask = (
    visual_metadata["split"]
    == "testing_set"
)


X_train = hybrid_features[
    train_mask.values
]

X_test = hybrid_features[
    test_mask.values
]

y_train = visual_labels[
    train_mask.values
]

y_test = visual_labels[
    test_mask.values
]


print("\nTraining:")
print(X_train.shape)

print("\nTesting:")
print(X_test.shape)


# ------------------------------------------------------------
# 9. Hybrid SVM
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training Hybrid SVM...")
print("-" * 70)


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42
        )
    )
])


model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 10. Predictions
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)


# ------------------------------------------------------------
# 11. Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("HYBRID MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ------------------------------------------------------------
# 12. Compare models
# ------------------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": "Geometry SVM",
        "accuracy": 0.4500,
        "weighted_f1": 0.4331
    },
    {
        "model": "Visual SVM",
        "accuracy": 0.4350,
        "weighted_f1": 0.4346
    },
    {
        "model": "Hybrid SVM",
        "accuracy": accuracy,
        "weighted_f1": f1
    }
])


comparison_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_comparison.csv"
)


comparison.to_csv(
    comparison_path,
    index=False
)


# ------------------------------------------------------------
# 13. Complete
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HYBRID EXPERIMENT COMPLETE")
print("=" * 70)

print("\nModel comparison:")

print(
    comparison.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
        }
    )
)

print("\nSaved:")
print(comparison_path)