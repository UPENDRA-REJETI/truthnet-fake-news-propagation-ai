from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIBVID_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fibvid"
    / "extracted"
    / "merry555-FibVID-14b95c3"
)

PROPAGATION_FILE = (
    FIBVID_ROOT
    / "claim_propagation"
    / "claim_propagation.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fibvid_propagation_features.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

EARLY_WINDOWS_HOURS = [1, 6, 24]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("FIBVID PROPAGATION FEATURE AUDIT")
print("=" * 80)

print(f"\nPropagation file:")
print(PROPAGATION_FILE)

if not PROPAGATION_FILE.exists():
    raise FileNotFoundError(
        f"Propagation file not found:\n{PROPAGATION_FILE}"
    )

print("\nLoading propagation data...")

df = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# 4. TIMESTAMP PROCESSING
# ============================================================

print("\nProcessing timestamps...")

df["create_date"] = pd.to_datetime(
    df["create_date"],
    utc=True,
    errors="coerce"
)

invalid_dates = df["create_date"].isna().sum()

print(f"Invalid timestamps: {invalid_dates}")


# ============================================================
# 5. BASIC CLAIM INFORMATION
# ============================================================

print("\nCalculating claim-level information...")

claim_first_time = (
    df.groupby("claim_number")["create_date"]
    .min()
    .rename("cascade_start")
)

claim_last_time = (
    df.groupby("claim_number")["create_date"]
    .max()
    .rename("cascade_end")
)

claim_table = pd.concat(
    [
        claim_first_time,
        claim_last_time
    ],
    axis=1
)

claim_table["duration_hours"] = (
    (
        claim_table["cascade_end"]
        - claim_table["cascade_start"]
    )
    .dt.total_seconds()
    / 3600
)


# ============================================================
# 6. FINAL CASCADE FEATURES
# ============================================================

final_features = (
    df.groupby("claim_number")
    .agg(
        final_cascade_size=(
            "tweet_id",
            "nunique"
        ),

        final_unique_users=(
            "tweet_user",
            "nunique"
        ),

        final_max_depth=(
            "depth",
            "max"
        ),

        final_total_likes=(
            "like_count",
            "sum"
        ),

        final_total_retweets=(
            "retweet_count",
            "sum"
        )
    )
)

claim_table = claim_table.join(
    final_features
)


# ============================================================
# 7. EARLY PROPAGATION FEATURES
# ============================================================

for hours in EARLY_WINDOWS_HOURS:

    print(
        f"\nCalculating {hours}-hour features..."
    )

    window_end = (
        claim_table["cascade_start"]
        + pd.to_timedelta(
            hours,
            unit="h"
        )
    )

    start_map = claim_table[
        "cascade_start"
    ].to_dict()

    end_map = window_end.to_dict()

    # Map each record to its claim start time
    df["cascade_start"] = (
        df["claim_number"]
        .map(start_map)
    )

    df["window_end"] = (
        df["claim_number"]
        .map(end_map)
    )

    early_df = df[
        (df["create_date"] >= df["cascade_start"])
        &
        (df["create_date"] <= df["window_end"])
    ]

    grouped = (
        early_df.groupby("claim_number")
        .agg(
            early_tweets=(
                "tweet_id",
                "nunique"
            ),

            early_users=(
                "tweet_user",
                "nunique"
            ),

            early_max_depth=(
                "depth",
                "max"
            ),

            early_likes=(
                "like_count",
                "sum"
            ),

            early_retweets=(
                "retweet_count",
                "sum"
            )
        )
    )

    grouped = grouped.rename(
        columns={
            column: f"{column}_{hours}h"
            for column in grouped.columns
        }
    )

    claim_table = claim_table.join(
        grouped
    )


# ============================================================
# 8. CLEAN MISSING EARLY VALUES
# ============================================================

early_columns = [
    column
    for column in claim_table.columns
    if any(
        column.endswith(f"_{hours}h")
        for hours in EARLY_WINDOWS_HOURS
    )
]

claim_table[early_columns] = (
    claim_table[early_columns]
    .fillna(0)
)


# ============================================================
# 9. DERIVED FEATURES
# ============================================================

for hours in EARLY_WINDOWS_HOURS:

    claim_table[
        f"growth_ratio_{hours}h"
    ] = (
        claim_table[
            f"early_tweets_{hours}h"
        ]
        /
        claim_table["final_cascade_size"]
    )

    claim_table[
        f"engagement_{hours}h"
    ] = (
        claim_table[
            f"early_likes_{hours}h"
        ]
        +
        claim_table[
            f"early_retweets_{hours}h"
        ]
    )


# ============================================================
# 10. LOG TRANSFORM TARGET
# ============================================================

claim_table[
    "log_final_cascade_size"
] = np.log1p(
    claim_table["final_cascade_size"]
)


# ============================================================
# 11. BASIC AUDIT
# ============================================================

print("\n" + "=" * 80)
print("FEATURE TABLE SUMMARY")
print("=" * 80)

print(
    f"\nNumber of propagation claims: "
    f"{len(claim_table)}"
)

print(
    f"Number of features: "
    f"{len(claim_table.columns)}"
)

print("\nColumns:")

for column in claim_table.columns:
    print(f"  - {column}")


print("\nMissing values:")

missing = claim_table.isna().sum()

print(
    missing[
        missing > 0
    ]
)


# ============================================================
# 12. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(
    claim_table[
        "final_cascade_size"
    ].describe()
)


# ============================================================
# 13. EARLY WINDOW COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("EARLY PROPAGATION COVERAGE")
print("=" * 80)

for hours in EARLY_WINDOWS_HOURS:

    tweets_column = (
        f"early_tweets_{hours}h"
    )

    users_column = (
        f"early_users_{hours}h"
    )

    claims_with_activity = (
        claim_table[tweets_column] > 0
    ).sum()

    print(
        f"\n{hours}-hour window:"
    )

    print(
        f"  Claims with activity: "
        f"{claims_with_activity}/{len(claim_table)}"
    )

    print(
        f"  Mean tweets: "
        f"{claim_table[tweets_column].mean():.2f}"
    )

    print(
        f"  Median tweets: "
        f"{claim_table[tweets_column].median():.2f}"
    )

    print(
        f"  Mean users: "
        f"{claim_table[users_column].mean():.2f}"
    )


# ============================================================
# 14. SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE FEATURE ROWS")
print("=" * 80)

print(
    claim_table.head(10).to_string()
)


# ============================================================
# 15. SAVE
# ============================================================

claim_table.to_csv(
    OUTPUT_FILE,
    index=True
)

print("\n" + "=" * 80)
print("FEATURE TABLE SAVED")
print("=" * 80)

print(OUTPUT_FILE)
print("=" * 80)