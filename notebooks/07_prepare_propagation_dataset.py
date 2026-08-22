from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_propagation_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fibvid_propagation_model_data.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIBVID PROPAGATION MODEL DATA PREPARATION")
print("=" * 80)

df = pd.read_csv(
    INPUT_FILE,
    index_col=0
)

print(f"Input rows: {len(df)}")


# ============================================================
# REMOVE FUTURE / TARGET-LEAKING INFORMATION
# ============================================================

columns_to_remove = [
    "cascade_start",
    "cascade_end",
    "duration_hours",

    # Final outcomes — these are not allowed as inputs
    "final_cascade_size",
    "final_unique_users",
    "final_max_depth",
    "final_total_likes",
    "final_total_retweets",

    # Derived using the final target
    "growth_ratio_1h",
    "growth_ratio_6h",
    "growth_ratio_24h"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "log_final_cascade_size"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# IDENTIFY FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column != TARGET
]


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nFeature columns:")

for column in feature_columns:
    print(f"  - {column}")

print(f"\nNumber of input features: {len(feature_columns)}")

print(f"\nTarget: {TARGET}")

print("\nMissing values:")

missing = df.isna().sum()

print(
    missing[missing > 0]
)


# ============================================================
# TARGET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TARGET SUMMARY")
print("=" * 80)

print(
    df[TARGET].describe()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=True
)

print("\n" + "=" * 80)
print("MODEL DATASET SAVED")
print("=" * 80)

print(OUTPUT_FILE)

print("=" * 80)