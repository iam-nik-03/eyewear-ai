import pandas as pd

from pathlib import Path


# ============================================================
# PATH
# ============================================================

INPUT_PATH = Path(
    "data/processed/recommendations/"
    "hybrid_model_comparison.csv"
)

OUTPUT_PATH = Path(
    "data/processed/recommendations/"
    "hybrid_model_comparison_with_improvement.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)


metadata = df[
    df["model"] == "Metadata Model"
].iloc[0]


hybrid = df[
    df["model"] == "Hybrid Model"
].iloc[0]


# ============================================================
# METRICS
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

results = []


for metric in metrics:

    metadata_value = float(
        metadata[metric]
    )

    hybrid_value = float(
        hybrid[metric]
    )

    absolute_improvement = (
        hybrid_value
        - metadata_value
    )

    relative_improvement = (
        absolute_improvement
        / metadata_value
    ) * 100


    results.append({

        "metric":
            metric.upper(),

        "metadata_model":
            metadata_value,

        "hybrid_model":
            hybrid_value,

        "absolute_improvement":
            absolute_improvement,

        "relative_improvement_percent":
            relative_improvement
    })


comparison = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("HYBRID MODEL IMPROVEMENT ANALYSIS")
print("==============================================")

print(
    comparison.to_string(
        index=False
    )
)


print(
    f"\nSaved to:"
    f"\n{OUTPUT_PATH}"
)


# ============================================================
# KEY RESULT
# ============================================================

f1_improvement = (
    hybrid["f1"]
    - metadata["f1"]
)


print("\n==============================================")
print("KEY RESULT")
print("==============================================")


print(
    f"F1 improvement: "
    f"{f1_improvement:.4f}"
)


print(
    f"F1 improvement in percentage points: "
    f"{f1_improvement * 100:.2f}"
)


print("\n==============================================")
print("COMPARISON COMPLETE")
print("==============================================")