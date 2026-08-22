from pathlib import Path
import random

import pandas as pd
import networkx as nx


# ============================================================
# PROJECT PATHS
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_causal_intervention_v2.csv"
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
    "late": 0.80,
}

FACT_CHECK_BLOCK = 0.25
MODERATION_BLOCK = 0.40
TARGETED_BLOCK = 0.60


# ============================================================
# LOAD PROPAGATION DATA
# ============================================================

print("=" * 80)
print("EXPERIMENT 03 — CAUSAL DYNAMIC INTERVENTION V2")
print("=" * 80)

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False,
)

propagation["create_date"] = pd.to_datetime(
    propagation["create_date"],
    utc=True,
)

cascade = propagation[
    propagation["claim_number"] == CLAIM_ID
].copy()

cascade = cascade.sort_values(
    "create_date"
).reset_index(drop=True)

cascade["event_index"] = range(
    len(cascade)
)

print(
    f"Propagation records: {len(cascade):,}"
)

print(
    f"Cascade claim: {CLAIM_ID}"
)


# ============================================================
# DATA INTEGRITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("PROPAGATION LINKAGE VALIDATION")
print("=" * 80)

duplicate_tweets = cascade[
    "tweet_id"
].duplicated().sum()

missing_parents = (
    (cascade["parent_id"] != 0)
    &
    (~cascade["parent_id"].isin(
        cascade["tweet_id"]
    ))
).sum()

self_parenting = (
    cascade["tweet_id"]
    ==
    cascade["parent_id"]
).sum()

root_events = (
    (cascade["parent_id"] == 0)
    &
    (cascade["parent_user"] == 0)
).sum()

print(
    f"Duplicate tweet IDs: {duplicate_tweets}"
)

print(
    f"Non-root parent IDs not found: "
    f"{missing_parents}"
)

print(
    f"Self-parenting events: "
    f"{self_parenting}"
)

print(
    f"Root events: "
    f"{root_events}"
)

if duplicate_tweets != 0:
    raise ValueError(
        "Duplicate tweet IDs detected."
    )

if missing_parents != 0:
    raise ValueError(
        "Some parent IDs do not exist."
    )

if self_parenting != 0:
    raise ValueError(
        "Self-parenting events detected."
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
            len(cascade) - 1,
        ),
    )

    checkpoint_data[name] = {
        "index": index,
        "time": cascade.loc[
            index,
            "create_date",
        ],
    }


print("\n" + "=" * 80)
print("PROPAGATION CHECKPOINTS")
print("=" * 80)

for name, data in checkpoint_data.items():

    print(
        f"{name.upper():<6} "
        f"event={data['index']:,} "
        f"time={data['time']}"
    )


# ============================================================
# DYNAMIC INFLUENCE CALCULATION
# ============================================================

def calculate_influence(
    observed_events,
):
    """
    Calculate influence using only the propagation
    visible before the intervention checkpoint.

    Graph is USER -> USER.
    """

    graph = nx.DiGraph()

    for row in observed_events.itertuples():

        user = int(
            row.tweet_user
        )

        parent_user = int(
            row.parent_user
        )

        if parent_user == 0:

            graph.add_node(
                user
            )

        else:

            graph.add_edge(
                parent_user,
                user
            )

    if len(graph) == 0:

        return pd.DataFrame()

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    pagerank = nx.pagerank(
        graph,
        alpha=0.85,
    )

    # --------------------------------------------------------
    # Weighted out-degree
    # --------------------------------------------------------

    weighted_out_degree = dict(
        graph.out_degree()
    )

    # --------------------------------------------------------
    # Betweenness
    # --------------------------------------------------------

    if len(graph) > 5000:

        betweenness = nx.betweenness_centrality(
            graph,
            k=min(
                500,
                len(graph),
            ),
            normalized=True,
            seed=RANDOM_SEED,
        )

    else:

        betweenness = nx.betweenness_centrality(
            graph,
            normalized=True,
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    def normalize(values):

        if not values:

            return {}

        maximum = max(
            values.values()
        )

        if maximum == 0:

            return {
                key: 0.0
                for key in values
            }

        return {
            key: value / maximum
            for key, value in values.items()
        }

    norm_pagerank = normalize(
        pagerank
    )

    norm_degree = normalize(
        weighted_out_degree
    )

    norm_betweenness = normalize(
        betweenness
    )

    # --------------------------------------------------------
    # Influence score
    # --------------------------------------------------------

    rows = []

    for user in graph.nodes():

        score = (
            0.40
            * norm_pagerank.get(
                user,
                0.0,
            )
            +
            0.40
            * norm_degree.get(
                user,
                0.0,
            )
            +
            0.20
            * norm_betweenness.get(
                user,
                0.0,
            )
        )

        out_degree = (
            weighted_out_degree.get(
                user,
                0,
            )
        )

        if out_degree >= 10:

            role = "Amplifier"

        elif (
            norm_betweenness.get(
                user,
                0.0,
            )
            >= 0.10
        ):

            role = "Bridge"

        elif (
            norm_pagerank.get(
                user,
                0.0,
            )
            >= 0.10
        ):

            role = "High-Reach Node"

        else:

            role = "Participant"

        rows.append(
            {
                "user_id": int(user),
                "influence_score": score,
                "pagerank":
                    pagerank.get(
                        user,
                        0.0,
                    ),
                "weighted_out_degree":
                    out_degree,
                "betweenness":
                    betweenness.get(
                        user,
                        0.0,
                    ),
                "network_role":
                    role,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "influence_score",
            ascending=False,
        )
    )


# ============================================================
# DYNAMIC TARGET SELECTION
# ============================================================

def select_targets(
    observed_events,
):
    """
    Select top influential users using only
    information available at the checkpoint.
    """

    influence = calculate_influence(
        observed_events
    )

    if influence.empty:

        return []

    return (
        influence
        .head(TARGET_COUNT)
        ["user_id"]
        .astype(int)
        .tolist()
    )


# ============================================================
# CAUSAL CASCADE SIMULATION
# ============================================================

def simulate_cascade(
    cascade,
    strategy,
    checkpoint_index=None,
    targets=None,
):
    """
    True causal propagation simulation.

    IMPORTANT:
    - parent_id -> tweet_id controls propagation.
    - parent_user determines whether the parent user
      is an intervention target.
    """

    random.seed(
        RANDOM_SEED
    )

    targets = set(
        targets or []
    )

    # --------------------------------------------------------
    # Intervention boundary
    # --------------------------------------------------------

    if checkpoint_index is None:

        intervention_index = (
            len(cascade) + 1
        )

    else:

        intervention_index = (
            checkpoint_index
        )

    # --------------------------------------------------------
    # Track SURVIVING TWEETS
    # --------------------------------------------------------

    surviving_tweet_ids = set()

    surviving_events = []

    # --------------------------------------------------------
    # Process events chronologically
    # --------------------------------------------------------

    for row in cascade.itertuples():

        event_index = int(
            row.event_index
        )

        tweet_id = int(
            row.tweet_id
        )

        parent_id = int(
            row.parent_id
        )

        parent_user = int(
            row.parent_user
        )

        # ====================================================
        # ROOT EVENT
        # ====================================================

        if parent_id == 0:

            surviving_tweet_ids.add(
                tweet_id
            )

            surviving_events.append(
                event_index
            )

            continue

        # ====================================================
        # CAUSAL PARENT CHECK
        # ====================================================

        if parent_id not in surviving_tweet_ids:

            # Parent tweet did not survive.
            # Therefore this event cannot propagate.
            continue

        # ====================================================
        # BEFORE INTERVENTION
        # ====================================================

        if event_index <= intervention_index:

            surviving_tweet_ids.add(
                tweet_id
            )

            surviving_events.append(
                event_index
            )

            continue

        # ====================================================
        # INTERVENTION
        # ====================================================

        block_probability = 0.0

        if strategy == "fact_checking":

            block_probability = (
                FACT_CHECK_BLOCK
            )

        elif strategy == "content_moderation":

            block_probability = (
                MODERATION_BLOCK
            )

        elif (
            strategy
            == "targeted_intervention"
        ):

            if parent_user in targets:

                block_probability = (
                    TARGETED_BLOCK
                )

        # ====================================================
        # STOCHASTIC BLOCK
        # ====================================================

        if (
            random.random()
            < block_probability
        ):

            continue

        # ====================================================
        # EVENT SURVIVES
        # ====================================================

        surviving_tweet_ids.add(
            tweet_id
        )

        surviving_events.append(
            event_index
        )

    # ========================================================
    # METRICS
    # ========================================================

    surviving_df = cascade.loc[
        surviving_events
    ]

    final_reach = (
        surviving_df[
            "tweet_user"
        ]
        .nunique()
    )

    remaining_edges = len(
        surviving_df[
            surviving_df[
                "parent_id"
            ] != 0
        ]
    )

    max_depth = (
        int(
            surviving_df[
                "depth"
            ].max()
        )
        if len(surviving_df)
        else 0
    )

    return {
        "final_reach":
            int(final_reach),

        "remaining_edges":
            int(remaining_edges),

        "max_depth":
            max_depth,
    }


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE — NO INTERVENTION")
print("=" * 80)

baseline = simulate_cascade(
    cascade,
    "none",
)

baseline_reach = (
    baseline["final_reach"]
)

baseline_edges = (
    baseline["remaining_edges"]
)

print(
    f"Final reach: "
    f"{baseline_reach:,}"
)

print(
    f"Remaining edges: "
    f"{baseline_edges:,}"
)

print(
    f"Maximum depth: "
    f"{baseline['max_depth']}"
)

# ------------------------------------------------------------
# Important validation
# ------------------------------------------------------------

if baseline_reach == 0:

    raise RuntimeError(
        "Baseline reach is zero."
    )

if baseline_edges == 0:

    raise RuntimeError(
        "Baseline contains no propagation edges."
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = [
    {
        "strategy": "none",
        "timing": "none",
        "targets": 0,
        **baseline,
    }
]


for timing_name, checkpoint in (
    checkpoint_data.items()
):

    checkpoint_index = (
        checkpoint["index"]
    )

    # --------------------------------------------------------
    # Information available at this point
    # --------------------------------------------------------

    observed_events = cascade.iloc[
        :checkpoint_index + 1
    ]

    # --------------------------------------------------------
    # Dynamic influence
    # --------------------------------------------------------

    targets = select_targets(
        observed_events
    )

    print("\n" + "=" * 80)
    print(
        f"{timing_name.upper()} "
        f"INTERVENTION"
    )
    print("=" * 80)

    print(
        f"Observed events: "
        f"{len(observed_events):,}"
    )

    print(
        f"Dynamic targets: "
        f"{len(targets)}"
    )

    for user in targets:

        print(
            f"  {user}"
        )

    # --------------------------------------------------------
    # Fact checking
    # --------------------------------------------------------

    result = simulate_cascade(
        cascade,
        "fact_checking",
        checkpoint_index,
    )

    results.append(
        {
            "strategy":
                "fact_checking",
            "timing":
                timing_name,
            "targets": 0,
            **result,
        }
    )

    # --------------------------------------------------------
    # Content moderation
    # --------------------------------------------------------

    result = simulate_cascade(
        cascade,
        "content_moderation",
        checkpoint_index,
    )

    results.append(
        {
            "strategy":
                "content_moderation",
            "timing":
                timing_name,
            "targets": 0,
            **result,
        }
    )

    # --------------------------------------------------------
    # Targeted intervention
    # --------------------------------------------------------

    result = simulate_cascade(
        cascade,
        "targeted_intervention",
        checkpoint_index,
        targets,
    )

    results.append(
        {
            "strategy":
                "targeted_intervention",
            "timing":
                timing_name,
            "targets":
                len(targets),
            **result,
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

results_df[
    "edge_reduction_percent"
] = (
    (
        baseline_edges
        -
        results_df[
            "remaining_edges"
        ]
    )
    /
    baseline_edges
    *
    100
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("CAUSAL INTERVENTION RESULTS V2")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

all_passed = True

for row in results_df.itertuples():

    if row.strategy == "none":

        continue

    if row.final_reach > baseline_reach:

        print(
            f"FAIL: "
            f"{row.strategy} "
            f"{row.timing} "
            f"increased reach."
        )

        all_passed = False

    else:

        print(
            f"PASS: "
            f"{row.strategy} "
            f"{row.timing}"
        )


# ============================================================
# TEMPORAL MONOTONICITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL EFFECT CHECK")
print("=" * 80)

for strategy in [
    "fact_checking",
    "content_moderation",
    "targeted_intervention",
]:

    subset = results_df[
        results_df["strategy"]
        == strategy
    ]

    early = subset[
        subset["timing"] == "early"
    ]["reach_reduction_percent"].iloc[0]

    mid = subset[
        subset["timing"] == "mid"
    ]["reach_reduction_percent"].iloc[0]

    late = subset[
        subset["timing"] == "late"
    ]["reach_reduction_percent"].iloc[0]

    if early >= mid >= late:

        print(
            f"PASS: {strategy} "
            f"(early >= mid >= late)"
        )

    else:

        print(
            f"INFO: {strategy} "
            f"did not show strict temporal monotonicity."
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("EXPERIMENT 03 V2 COMPLETE")
print("=" * 80)

print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"All basic sanity checks: "
    f"{'PASSED' if all_passed else 'FAILED'}"
)

print("=" * 80)