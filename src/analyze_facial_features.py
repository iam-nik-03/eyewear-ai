from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATH
# =========================================================

FEATURES_PATH = Path(
    "data/processed/faces/facial_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/faces/analysis"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("EYewear AI - FACIAL FEATURE ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not FEATURES_PATH.exists():

        print("\nERROR: Feature dataset not found:")
        print(FEATURES_PATH)

        return

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = pd.read_csv(
        FEATURES_PATH
    )

    print("\nDataset loaded successfully.")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # -----------------------------------------------------
    # Basic information
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("COLUMNS")
    print("-" * 70)

    for column in df.columns:
        print(f"  {column}")

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isnull().sum()

    for column, count in missing.items():

        print(
            f"{column:30s}: {count}"
        )

    # -----------------------------------------------------
    # Dataset split
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAIN / TEST DISTRIBUTION")
    print("-" * 70)

    print(
        df["split"].value_counts()
    )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        df["label"].value_counts()
    )

    print("\nBy split:")

    print(
        pd.crosstab(
            df["split"],
            df["label"]
        )
    )

    # -----------------------------------------------------
    # Feature statistics
    # -----------------------------------------------------

    feature_columns = [
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

    print("\n" + "-" * 70)
    print("OVERALL FEATURE STATISTICS")
    print("-" * 70)

    print(
        df[feature_columns].describe().round(4)
    )

    # -----------------------------------------------------
    # Class means
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("FEATURE MEANS BY FACE SHAPE")
    print("-" * 70)

    class_means = (
        df.groupby("label")[feature_columns]
        .mean()
        .round(4)
    )

    print(
        class_means.to_string()
    )

    # -----------------------------------------------------
    # Class standard deviation
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("FEATURE STANDARD DEVIATION BY FACE SHAPE")
    print("-" * 70)

    class_std = (
        df.groupby("label")[feature_columns]
        .std()
        .round(4)
    )

    print(
        class_std.to_string()
    )

    # -----------------------------------------------------
    # Train/test comparison
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAIN / TEST FEATURE MEANS")
    print("-" * 70)

    train_means = (
        df[df["split"] == "training_set"]
        [feature_columns]
        .mean()
    )

    test_means = (
        df[df["split"] == "testing_set"]
        [feature_columns]
        .mean()
    )

    comparison = pd.DataFrame({
        "train_mean": train_means,
        "test_mean": test_means,
        "difference": test_means - train_means,
    })

    print(
        comparison.round(4).to_string()
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("FEATURE CORRELATION")
    print("-" * 70)

    correlation = (
        df[feature_columns]
        .corr()
        .round(3)
    )

    print(
        correlation.to_string()
    )

    # -----------------------------------------------------
    # Save analysis
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    class_means.to_csv(
        OUTPUT_DIR / "class_feature_means.csv"
    )

    class_std.to_csv(
        OUTPUT_DIR / "class_feature_std.csv"
    )

    comparison.to_csv(
        OUTPUT_DIR / "train_test_feature_comparison.csv"
    )

    correlation.to_csv(
        OUTPUT_DIR / "feature_correlation.csv"
    )

    # -----------------------------------------------------
    # Create boxplots
    # -----------------------------------------------------

    print("\nCreating feature plots...")

    for feature in feature_columns:

        plt.figure(
            figsize=(9, 6)
        )

        df.boxplot(
            column=feature,
            by="label"
        )

        plt.title(
            f"{feature} by Face Shape"
        )

        plt.suptitle("")

        plt.xlabel(
            "Face Shape"
        )

        plt.ylabel(
            feature
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{feature}_distribution.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FACIAL FEATURE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nAnalysis files saved to:")

    print(
        OUTPUT_DIR.resolve()
    )

    print("\nGenerated:")
    print("  class_feature_means.csv")
    print("  class_feature_std.csv")
    print("  train_test_feature_comparison.csv")
    print("  feature_correlation.csv")
    print("  feature distribution plots")


if __name__ == "__main__":
    main()