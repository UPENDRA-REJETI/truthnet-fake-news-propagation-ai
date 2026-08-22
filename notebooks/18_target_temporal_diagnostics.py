from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
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

INFLUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_all_cascade_influence.csv"
)


CLAIM_ID = 281
TARGET_COUNT = 10


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("TARGET TEMPORAL DIAGNOSTICS")
print("=" * 80)

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

influence = pd.read_csv(
    INFLUENCE_FILE
)


# ============================================================
# CASCADE
# ============================================================

cascade = propagation[
    propagation["claim_number"] == CLAIM_ID
].copy()

cascade["create_date"] = pd.to_datetime(
    cascade["create_date"],
    utc=True
)

cascade = cascade.sort_values(
    "create_date"
)

start = cascade[
    "create_date"
].min()

end = cascade[
    "create_date"
].max()

duration_seconds = (
    end - start
).total_seconds()


# ============================================================
# TARGETS
# ============================================================

claim_influence = influence[
    influence["claim_number"] == CLAIM_ID
].copy()

claim_influence = (
    claim_influence
    .sort_values(
        "influence_score",
        ascending=False
    )
    .head(TARGET_COUNT)
)


targets = (
    claim_influence["user_id"]
    .astype(int)
    .tolist()
)


# ============================================================
# INTERVENTION TIMES
# ============================================================

timing = {
    "early": 0.10,
    "mid": 0.50,
    "late": 0.80
}

print("\nCascade timeline:")
print(f"Start: {start}")
print(f"End:   {end}")
print(
    f"Duration: "
    f"{duration_seconds / 86400:.2f} days"
)

print("\nIntervention points:")

intervention_times = {}

for name, fraction in timing.items():

    intervention_time = (
        start
        +
        pd.Timedelta(
            seconds=
            duration_seconds * fraction
        )
    )

    intervention_times[name] = (
        intervention_time
    )

    print(
        f"{name:>6}: "
        f"{intervention_time}"
    )


# ============================================================
# TARGET ACTIVITY
# ============================================================

print("\n" + "=" * 80)
print("TARGET ACTIVITY")
print("=" * 80)

rows = []

for user in targets:

    user_events = cascade[
        cascade["tweet_user"]
        == user
    ]

    if len(user_events) == 0:

        continue

    first_activity = (
        user_events[
            "create_date"
        ].min()
    )

    last_activity = (
        user_events[
            "create_date"
        ].max()
    )

    event_count = len(
        user_events
    )

    rows.append(
        {
            "user_id": user,
            "influence_score":
                claim_influence.loc[
                    claim_influence["user_id"]
                    == user,
                    "influence_score"
                ].iloc[0],
            "first_activity":
                first_activity,
            "last_activity":
                last_activity,
            "propagation_events":
                event_count
        }
    )


target_activity = pd.DataFrame(
    rows
)


# ============================================================
# CHECK WHETHER TARGET WAS ACTIVE
# ============================================================

for timing_name, intervention_time in (
    intervention_times.items()
):

    target_activity[
        f"active_before_{timing_name}"
    ] = (
        target_activity[
            "first_activity"
        ]
        <= intervention_time
    )


# ============================================================
# DISPLAY
# ============================================================

columns = [
    "user_id",
    "influence_score",
    "first_activity",
    "last_activity",
    "propagation_events",
    "active_before_early",
    "active_before_mid",
    "active_before_late"
]

print(
    target_activity[
        columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TARGET COVERAGE")
print("=" * 80)

for timing_name in timing:

    column = (
        f"active_before_{timing_name}"
    )

    active_count = int(
        target_activity[column]
        .sum()
    )

    total = len(
        target_activity
    )

    print(
        f"{timing_name.capitalize():<6}: "
        f"{active_count}/{total} "
        f"targets active before intervention"
    )


print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)