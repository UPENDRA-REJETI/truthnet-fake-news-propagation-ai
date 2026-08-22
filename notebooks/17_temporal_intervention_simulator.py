from pathlib import Path
import random

import pandas as pd
import networkx as nx


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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

CLAIM_ID = 281

TARGET_COUNT = 10

# Intervention starts as a fraction of the
# observed cascade duration.
TIMING = {
    "early": 0.10,
    "mid": 0.50,
    "late": 0.80
}

# Probability reduction for affected propagation.
GLOBAL_REDUCTION = {
    "fact_checking": 0.25,
    "content_moderation": 0.40
}

TARGETED_REDUCTION = 0.60


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TEMPORAL INTERVENTION SIMULATOR")
print("=" * 80)

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

influence = pd.read_csv(
    INFLUENCE_FILE
)

print(
    f"Propagation records: {len(propagation):,}"
)


# ============================================================
# SELECT CASCADE
# ============================================================

cascade = propagation[
    propagation["claim_number"]
    == CLAIM_ID
].copy()

cascade["create_date"] = pd.to_datetime(
    cascade["create_date"],
    utc=True
)

cascade = cascade.sort_values(
    "create_date"
).reset_index(
    drop=True
)

start_time = cascade[
    "create_date"
].min()

end_time = cascade[
    "create_date"
].max()

duration = (
    end_time - start_time
).total_seconds()

print("\nCascade:")
print(f"Claim: {CLAIM_ID}")
print(f"Records: {len(cascade):,}")
print(f"Start: {start_time}")
print(f"End: {end_time}")
print(
    f"Duration: "
    f"{duration / 3600:.2f} hours"
)


# ============================================================
# GET TARGET USERS
# ============================================================

claim_influence = influence[
    influence["claim_number"]
    == CLAIM_ID
].copy()

claim_influence = (
    claim_influence
    .sort_values(
        "influence_score",
        ascending=False
    )
)

target_users = set(
    claim_influence
    .head(TARGET_COUNT)
    ["user_id"]
    .astype(int)
)

print("\nSelected intervention targets:")

for user in sorted(target_users):
    print(f"  {user}")


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    cascade,
    strategy,
    timing_name=None
):

    random.seed(
        RANDOM_SEED
    )

    if timing_name is None:

        intervention_time = None

    else:

        fraction = TIMING[
            timing_name
        ]

        intervention_time = (
            start_time
            +
            pd.Timedelta(
                seconds=
                duration * fraction
            )
        )

    surviving = []

    for row in cascade.itertuples():

        parent = int(
            row.parent_user
        )

        user = int(
            row.tweet_user
        )

        event_time = row.create_date

        # Root event always survives.
        if parent == 0:

            surviving.append(
                row.Index
            )

            continue

        probability = 1.0

        intervention_active = (
            intervention_time is not None
            and event_time >= intervention_time
        )

        # ----------------------------------------------------
        # Fact checking
        # ----------------------------------------------------

        if (
            strategy == "fact_checking"
            and intervention_active
        ):

            probability *= (
                1
                -
                GLOBAL_REDUCTION[
                    "fact_checking"
                ]
            )

        # ----------------------------------------------------
        # Content moderation
        # ----------------------------------------------------

        elif (
            strategy == "content_moderation"
            and intervention_active
        ):

            probability *= (
                1
                -
                GLOBAL_REDUCTION[
                    "content_moderation"
                ]
            )

        # ----------------------------------------------------
        # Targeted intervention
        # ----------------------------------------------------

        elif (
            strategy == "targeted_intervention"
            and intervention_active
            and parent in target_users
        ):

            probability *= (
                1
                -
                TARGETED_REDUCTION
            )

        # ----------------------------------------------------
        # Propagation decision
        # ----------------------------------------------------

        if random.random() < probability:

            surviving.append(
                row.Index
            )

    simulated = cascade.loc[
        surviving
    ].copy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_reach = (
        simulated[
            "tweet_user"
        ]
        .nunique()
    )

    remaining_edges = len(
        simulated[
            simulated["parent_user"] != 0
        ]
    )

    max_depth = (
        simulated["depth"].max()
        if len(simulated)
        else 0
    )

    return {
        "strategy": strategy,
        "timing": timing_name or "none",
        "final_reach": int(
            final_reach
        ),
        "remaining_edges": int(
            remaining_edges
        ),
        "max_depth": int(
            max_depth
        )
    }


# ============================================================
# RUN BASELINE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)

baseline = simulate(
    cascade,
    "none"
)

print(baseline)


baseline_reach = (
    baseline["final_reach"]
)


# ============================================================
# RUN TEMPORAL INTERVENTIONS
# ============================================================

results = [
    baseline
]

strategies = [
    "fact_checking",
    "content_moderation",
    "targeted_intervention"
]

for strategy in strategies:

    for timing_name in [
        "early",
        "mid",
        "late"
    ]:

        result = simulate(
            cascade,
            strategy,
            timing_name
        )

        results.append(
            result
        )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df[
    "reach_reduction_percent"
] = (
    (
        baseline_reach
        -
        results_df[
            "final_reach"
        ]
    )
    /
    baseline_reach
    *
    100
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL INTERVENTION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_temporal_intervention.csv"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 80)
print("EXPERIMENT 03 COMPLETE")
print("=" * 80)

print(
    f"Results saved to:\n{output_file}"
)

print("=" * 80)