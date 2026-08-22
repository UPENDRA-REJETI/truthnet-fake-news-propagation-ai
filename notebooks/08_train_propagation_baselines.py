from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

WINDOWS = [1, 6, 24]

TEST_SIZE = 0.20


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_propagation_model_data.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    RESULTS_DIR
    / "experiment_02_results.csv"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("EXPERIMENT 02 — PROPAGATION PREDICTION BASELINES")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0
)

print(f"Total cascades: {len(df)}")


# ============================================================
# 4. TARGET
# ============================================================

TARGET = "log_final_cascade_size"

y = df[TARGET].values


# ============================================================
# 5. RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


print("\nDataset split:")
print(f"Training cascades: {len(train_indices)}")
print(f"Testing cascades:  {len(test_indices)}")


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    window,
    y_true_log,
    y_pred_log
):

    # Metrics in logarithmic target space
    log_mae = mean_absolute_error(
        y_true_log,
        y_pred_log
    )

    log_rmse = np.sqrt(
        mean_squared_error(
            y_true_log,
            y_pred_log
        )
    )

    log_r2 = r2_score(
        y_true_log,
        y_pred_log
    )

    # Convert back to actual cascade size
    y_true_original = np.expm1(
        y_true_log
    )

    y_pred_original = np.maximum(
        np.expm1(y_pred_log),
        0
    )

    original_mae = mean_absolute_error(
        y_true_original,
        y_pred_original
    )

    original_rmse = np.sqrt(
        mean_squared_error(
            y_true_original,
            y_pred_original
        )
    )

    original_r2 = r2_score(
        y_true_original,
        y_pred_original
    )

    results.append(
        {
            "window_hours": window,
            "model": model_name,
            "log_mae": log_mae,
            "log_rmse": log_rmse,
            "log_r2": log_r2,
            "original_mae": original_mae,
            "original_rmse": original_rmse,
            "original_r2": original_r2
        }
    )

    print(
        f"\n{model_name} — {window}-hour window"
    )

    print(
        f"Log MAE       : {log_mae:.4f}"
    )

    print(
        f"Log RMSE      : {log_rmse:.4f}"
    )

    print(
        f"Log R²        : {log_r2:.4f}"
    )

    print(
        f"Original MAE  : {original_mae:.2f}"
    )

    print(
        f"Original RMSE : {original_rmse:.2f}"
    )

    print(
        f"Original R²   : {original_r2:.4f}"
    )


# ============================================================
# 8. RUN EACH TIME WINDOW
# ============================================================

for window in WINDOWS:

    print("\n" + "=" * 80)
    print(f"{window}-HOUR PROPAGATION PREDICTION")
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

    X_train = X[
        train_indices
    ]

    X_test = X[
        test_indices
    ]

    y_train = y[
        train_indices
    ]

    y_test = y[
        test_indices
    ]


    # ========================================================
    # MODEL 0 — MEDIAN BASELINE
    # ========================================================

    median_prediction = np.full(
        len(y_test),
        np.median(y_train)
    )

    evaluate_model(
        "Median Baseline",
        window,
        y_test,
        median_prediction
    )


    # ========================================================
    # MODEL 1 — RIDGE REGRESSION
    # ========================================================

    ridge = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=1.0)
            )
        ]
    )

    ridge.fit(
        X_train,
        y_train
    )

    ridge_predictions = ridge.predict(
        X_test
    )

    evaluate_model(
        "Ridge Regression",
        window,
        y_test,
        ridge_predictions
    )


    # ========================================================
    # MODEL 2 — RANDOM FOREST
    # ========================================================

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = random_forest.predict(
        X_test
    )

    evaluate_model(
        "Random Forest",
        window,
        y_test,
        rf_predictions
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 10. DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 02 — FINAL COMPARISON")
print("=" * 80)

display_columns = [
    "window_hours",
    "model",
    "log_mae",
    "log_rmse",
    "log_r2",
    "original_mae",
    "original_rmse",
    "original_r2"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)

print("\nResults saved to:")
print(RESULTS_FILE)

print("=" * 80)