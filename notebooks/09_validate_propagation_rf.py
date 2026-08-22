from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_propagation_model_data.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_02_rf_cv_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("EXPERIMENT 02 — RANDOM FOREST ROBUSTNESS CHECK")
print("=" * 80)

df = pd.read_csv(
    DATA_FILE,
    index_col=0
)

TARGET = "log_final_cascade_size"

y = df[TARGET].values


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def rmse(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": make_scorer(
        rmse,
        greater_is_better=False
    )
}


results = []


# ============================================================
# EACH WINDOW
# ============================================================

for window in [1, 6, 24]:

    print("\n" + "=" * 80)
    print(f"{window}-HOUR RANDOM FOREST")
    print("=" * 80)

    features = [
        f"early_tweets_{window}h",
        f"early_users_{window}h",
        f"early_max_depth_{window}h",
        f"early_likes_{window}h",
        f"early_retweets_{window}h",
        f"engagement_{window}h"
    ]

    X = df[features].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    r2_scores = scores["test_r2"]

    mae_scores = -scores["test_mae"]

    rmse_scores = -scores["test_rmse"]

    print(
        f"R² scores: "
        f"{np.round(r2_scores, 4)}"
    )

    print(
        f"R² mean: "
        f"{r2_scores.mean():.4f}"
    )

    print(
        f"R² std: "
        f"{r2_scores.std():.4f}"
    )

    print(
        f"MAE mean: "
        f"{mae_scores.mean():.4f}"
    )

    print(
        f"RMSE mean: "
        f"{rmse_scores.mean():.4f}"
    )

    results.append(
        {
            "window_hours": window,
            "r2_mean": r2_scores.mean(),
            "r2_std": r2_scores.std(),
            "mae_mean": mae_scores.mean(),
            "rmse_mean": rmse_scores.mean()
        }
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS SUMMARY")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(RESULTS_FILE)

print("=" * 80)