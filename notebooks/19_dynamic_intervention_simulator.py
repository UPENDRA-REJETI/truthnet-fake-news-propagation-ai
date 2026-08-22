from pathlib import Path
import random

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_dynamic_intervention.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

CLAIM_ID = 281

TARGET_COUNT = 10

CHECKPOINTS = {
    "early": 0.20,
    "mid": 0.50,
    "late": 0.80
}

FACT_CHECK_REDUCTION = 0.25

MODERATION_REDUCTION = 0.40

TARGETED_REDUCTION = 0.60


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DYNAMIC INTERVENTION SIMULATOR")
print("=" * 80)

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

influence = pd.read_csv(
    INFLUENCE_FILE
)

propagation["create_date"] = pd.to_datetime(
    propagation["create_date"],
    utc=True
)


# ============================================================
# SELECT CASCADE
# ============================================================

cascade = propagation[
    propagation["claim_number"] == CLAIM_ID
].copy()

cascade = cascade.sort_values(
    "create_date"
).reset_index(
    drop=True
)

print(
    f"Propagation records: "
    f"{len(cascade):,}"
)

print(
    f"Cascade: {CLAIM_ID}"
)


# ============================================================
# CHECKPOINTS
# ============================================================

checkpoint_data = {}

for name, fraction in CHECKPOINTS.items():

    index = int(
        len(cascade) * fraction
    )

    index = max(
        1,
        min(
            index,
            len(cascade) - 1
        )
    )

    checkpoint_time = (
        cascade.loc[
            index,
            "create_date"
        ]
    )

    checkpoint_data[name] = {
        "index": index,
        "time": checkpoint_time
    }


print("\nPropagation checkpoints:")

for name, data in checkpoint_data.items():

    print(
        f"{name:>6}: "
        f"event {data['index']:,} "
        f"/ {len(cascade):,} "
        f"at {data['time']}"
    )


# ============================================================
# CASCADE INFLUENCE
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
)


# ============================================================
# DYNAMIC TARGET SELECTION
# ============================================================

def select_targets(
    cascade,
    influence_table,
    checkpoint_index
):

    observed = cascade.iloc[
        :checkpoint_index + 1
    ]

    active_users = set(
        observed[
            "tweet_user"
        ]
        .astype(int)
        .unique()
    )

    candidates = influence_table[
        influence_table["user_id"]
        .astype(int)
        .isin(active_users)
    ].copy()

    candidates = candidates.sort_values(
        "influence_score",
        ascending=False
    )

    selected = (
        candidates
        .head(TARGET_COUNT)
    )

    return selected


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    cascade,
    strategy,
    checkpoint_index=None,
    targets=None
):

    random.seed(
        RANDOM_SEED
    )

    if checkpoint_index is None:

        intervention_active = False

    else:

        intervention_active = True

    targets = set(
        targets or []
    )

    # --------------------------------------------------------
    # We preserve the historical propagation.
    # Only future events are subject to intervention.
    # --------------------------------------------------------

    surviving = []

    for idx, row in cascade.iterrows():

        parent = int(
            row["parent_user"]
        )

        user = int(
            row["tweet_user"]
        )

        # Root always survives.
        if parent == 0:

            surviving.append(
                idx
            )

            continue

        # Before intervention:
        if (
            not intervention_active
            or idx <= checkpoint_index
        ):

            surviving.append(
                idx
            )

            continue

        probability = 1.0

        # ----------------------------------------------------
        # Global fact-checking
        # ----------------------------------------------------

        if strategy == "fact_checking":

            probability *= (
                1
                -
                FACT_CHECK_REDUCTION
            )

        # ----------------------------------------------------
        # Global moderation
        # ----------------------------------------------------

        elif strategy == "content_moderation":

            probability *= (
                1
                -
                MODERATION_REDUCTION
            )

        # ----------------------------------------------------
        # Targeted intervention
        # ----------------------------------------------------

        elif (
            strategy
            == "targeted_intervention"
        ):

            if parent in targets:

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
                idx
            )

    simulated = cascade.loc[
        surviving
    ].copy()

    # ========================================================
    # METRICS
    # ========================================================

    final_reach = (
        simulated[
            "tweet_user"
        ]
        .nunique()
    )

    remaining_edges = len(
        simulated[
            simulated[
                "parent_user"
            ] != 0
        ]
    )

    max_depth = (
        simulated[
            "depth"
        ].max()
        if len(simulated)
        else 0
    )

    return {
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
# BASELINE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)

baseline = simulate(
    cascade,
    "none"
)

baseline_reach = (
    baseline["final_reach"]
)

print(
    f"Final reach: "
    f"{baseline_reach}"
)


# ============================================================
# EXPERIMENTS
# ============================================================

results = [
    {
        "strategy": "none",
        "timing": "none",
        "targets": 0,
        **baseline
    }
]


for timing_name, checkpoint in (
    checkpoint_data.items()
):

    checkpoint_index = (
        checkpoint["index"]
    )

    # --------------------------------------------------------
    # Dynamic target selection
    # --------------------------------------------------------

    selected = select_targets(
        cascade,
        claim_influence,
        checkpoint_index
    )

    targets = (
        selected["user_id"]
        .astype(int)
        .tolist()
    )

    print("\n" + "=" * 80)
    print(
        f"{timing_name.upper()} INTERVENTION"
    )
    print("=" * 80)

    print(
        f"Active candidates: "
        f"{len(cascade.iloc[:checkpoint_index + 1]['tweet_user'].unique()):,}"
    )

    print(
        "Selected targets:"
    )

    for user in targets:

        row = selected[
            selected["user_id"]
            == user
        ].iloc[0]

        print(
            f"  {user} | "
            f"{row['network_role']} | "
            f"score={row['influence_score']:.4f}"
        )

    # --------------------------------------------------------
    # Fact checking
    # --------------------------------------------------------

    result = simulate(
        cascade,
        "fact_checking",
        checkpoint_index
    )

    results.append(
        {
            "strategy": "fact_checking",
            "timing": timing_name,
            "targets": 0,
            **result
        }
    )

    # --------------------------------------------------------
    # Content moderation
    # --------------------------------------------------------

    result = simulate(
        cascade,
        "content_moderation",
        checkpoint_index
    )

    results.append(
        {
            "strategy": "content_moderation",
            "timing": timing_name,
            "targets": 0,
            **result
        }
    )

    # --------------------------------------------------------
    # Targeted intervention
    # --------------------------------------------------------

    result = simulate(
        cascade,
        "targeted_intervention",
        checkpoint_index,
        targets
    )

    results.append(
        {
            "strategy":
                "targeted_intervention",
            "timing":
                timing_name,
            "targets":
                len(targets),
            **result
        }
    )


# ============================================================
# RESULTS
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
print("DYNAMIC INTERVENTION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("EXPERIMENT 03 COMPLETE")
print("=" * 80)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("=" * 80)