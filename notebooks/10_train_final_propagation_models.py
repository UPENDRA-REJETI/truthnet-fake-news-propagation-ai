from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_propagation_model_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "propagation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_FILE = (
    MODEL_DIR
    / "metadata.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

WINDOWS = [1, 6, 24]

TARGET = "log_final_cascade_size"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL PROPAGATION MODEL TRAINING")
print("=" * 80)

print("\nLoading propagation dataset...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0
)

print(
    f"Training cascades: {len(df)}"
)

print(
    f"Target: {TARGET}"
)


# ============================================================
# 4. TARGET
# ============================================================

y = df[TARGET].values


# ============================================================
# 5. METADATA
# ============================================================

metadata = {
    "dataset": "FibVID",
    "sample_count": len(df),
    "target": TARGET,
    "target_transformation": "log1p(final_cascade_size)",
    "model": "RandomForestRegressor",
    "random_state": RANDOM_STATE,
    "windows": {}
}


# ============================================================
# 6. TRAIN THREE FINAL MODELS
# ============================================================

for window in WINDOWS:

    print("\n" + "=" * 80)
    print(f"TRAINING {window}-HOUR MODEL")
    print("=" * 80)

    feature_columns = [
        f"early_tweets_{window}h",
        f"early_users_{window}h",
        f"early_max_depth_{window}h",
        f"early_likes_{window}h",
        f"early_retweets_{window}h",
        f"engagement_{window}h"
    ]

    X = df[
        feature_columns
    ].values

    print("\nFeatures:")

    for feature in feature_columns:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Train final Random Forest
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining...")

    model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR
        / f"propagation_rf_{window}h.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nModel saved:"
    )

    print(model_file)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = dict(
        zip(
            feature_columns,
            model.feature_importances_
        )
    )

    importance = dict(
        sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    print("\nFeature importance:")

    for feature, value in importance.items():

        print(
            f"  {feature}: {value:.4f}"
        )

    metadata["windows"][
        str(window)
    ] = {
        "model_file": str(
            model_file.relative_to(
                PROJECT_ROOT
            )
        ),
        "features": feature_columns,
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 3,
        "feature_importance": importance
    }


# ============================================================
# 7. SAVE METADATA
# ============================================================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL PROPAGATION MODELS READY")
print("=" * 80)

print("\nModels:")

for window in WINDOWS:

    print(
        f"  {window}h → "
        f"models/propagation/"
        f"propagation_rf_{window}h.joblib"
    )

print("\nMetadata:")

print(
    "  models/propagation/metadata.json"
)

print("\nTraining dataset:")
print(
    f"  {len(df)} cascades"
)

print("=" * 80)