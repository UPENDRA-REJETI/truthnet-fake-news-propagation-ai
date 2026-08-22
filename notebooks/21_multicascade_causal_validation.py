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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_multicascade_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TOP_N_CASCADES = 15
TARGET_COUNT = 10

CHECKPOINTS = {
    "early": 0.20,
    "mid": 0.50,
    "late": 0.80,
}

FACT_CHECK_BLOCK = 0.25
MODERATION_BLOCK = 0.40
TARGETED_BLOCK = 0.60


random.seed(RANDOM_SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("EXPERIMENT 03 — MULTI-CASCADE CAUSAL VALIDATION")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading FibVID propagation data...")

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False,
)

propagation["create_date"] = pd.to_datetime(
    propagation["create_date"],
    utc=True,
)

print(
    f"Propagation records: "
    f"{len(propagation):,}"
)


# ============================================================
# CASCADE SIZES
# ============================================================

cascade_sizes = (
    propagation
    .groupby("claim_number")
    .size()
    .sort_values(
        ascending=False
    )
)

selected_claims = (
    cascade_sizes
    .head(TOP_N_CASCADES)
    .index
    .tolist()
)


print("\n" + "=" * 80)
print("SELECTED CASCADES")
print("=" * 80)

for rank, claim_id in enumerate(
    selected_claims,
    start=1,
):

    print(
        f"{rank:>2}. "
        f"Claim {claim_id:<5} "
        f"records = "
        f"{cascade_sizes[claim_id]:,}"
    )


# ============================================================
# INFLUENCE CALCULATION
# ============================================================

def calculate_influence(
    observed_events,
):
    """
    Calculate dynamic user influence using only
    events observed before intervention.
    """

    graph = nx.DiGraph()

    for row in observed_events.itertuples():

        user = int(row.tweet_user)
        parent_user = int(row.parent_user)

        if parent_user == 0:

            graph.add_node(user)

        else:

            graph.add_edge(
                parent_user,
                user,
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

    weighted_degree = dict(
        graph.out_degree()
    )

    # --------------------------------------------------------
    # Betweenness
    # --------------------------------------------------------

    if len(graph) > 5000:

        betweenness = nx.betweenness_centrality(
            graph,
            k=min(
                300,
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

    norm_page = normalize(
        pagerank
    )

    norm_degree = normalize(
        weighted_degree
    )

    norm_between = normalize(
        betweenness
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    rows = []

    for user in graph.nodes():

        score = (
            0.40
            * norm_page.get(
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
            * norm_between.get(
                user,
                0.0,
            )
        )

        rows.append(
            {
                "user_id": int(user),
                "influence_score": score,
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
# TARGET SELECTION
# ============================================================

def select_targets(
    observed_events,
):
    """
    Select the top influential users based only on
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
# CAUSAL SIMULATION
# ============================================================

def simulate(
    cascade,
    strategy,
    checkpoint_index=None,
    targets=None,
):
    """
    Causal propagation simulation.

    parent_id -> tweet_id determines whether an event
    can propagate.

    parent_user determines whether an intervention target
    is responsible for the propagation event.
    """

    targets = set(
        targets or []
    )

    if checkpoint_index is None:

        intervention_index = (
            len(cascade) + 1
        )

    else:

        intervention_index = (
            checkpoint_index
        )

    surviving_tweets = set()

    surviving_indices = []

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

        # ----------------------------------------------------
        # ROOT EVENT
        # ----------------------------------------------------

        if parent_id == 0:

            surviving_tweets.add(
                tweet_id
            )

            surviving_indices.append(
                event_index
            )

            continue

        # ----------------------------------------------------
        # CAUSAL PARENT CHECK
        # ----------------------------------------------------

        if parent_id not in surviving_tweets:

            continue

        # ----------------------------------------------------
        # HISTORICAL EVENTS
        # ----------------------------------------------------

        if event_index <= intervention_index:

            surviving_tweets.add(
                tweet_id
            )

            surviving_indices.append(
                event_index
            )

            continue

        # ----------------------------------------------------
        # INTERVENTION PROBABILITY
        # ----------------------------------------------------

        probability = 0.0

        if strategy == "fact_checking":

            probability = (
                FACT_CHECK_BLOCK
            )

        elif strategy == "content_moderation":

            probability = (
                MODERATION_BLOCK
            )

        elif strategy == "targeted_intervention":

            if parent_user in targets:

                probability = (
                    TARGETED_BLOCK
                )

        # ----------------------------------------------------
        # BLOCK
        # ----------------------------------------------------

        if random.random() < probability:

            continue

        # ----------------------------------------------------
        # SURVIVE
        # ----------------------------------------------------

        surviving_tweets.add(
            tweet_id
        )

        surviving_indices.append(
            event_index
        )

    # ========================================================
    # METRICS
    # ========================================================

    surviving = cascade.loc[
        surviving_indices
    ]

    final_reach = (
        surviving[
            "tweet_user"
        ]
        .nunique()
    )

    remaining_edges = int(
        (
            surviving["parent_id"]
            != 0
        ).sum()
    )

    max_depth = (
        int(
            surviving["depth"].max()
        )
        if len(surviving)
        else 0
    )

    return {
        "final_reach":
            int(final_reach),

        "remaining_edges":
            remaining_edges,

        "max_depth":
            max_depth,
    }


# ============================================================
# PROCESS CASCADES
# ============================================================

all_results = []

print("\n" + "=" * 80)
print("PROCESSING CASCADES")
print("=" * 80)


for cascade_number, claim_id in enumerate(
    selected_claims,
    start=1,
):

    print(
        f"\n[{cascade_number}/{TOP_N_CASCADES}] "
        f"Claim {claim_id}"
    )

    cascade = propagation[
        propagation["claim_number"]
        == claim_id
    ].copy()

    cascade = (
        cascade
        .sort_values("create_date")
        .reset_index(drop=True)
    )

    cascade["event_index"] = range(
        len(cascade)
    )

    print(
        f"Records: {len(cascade):,}"
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    duplicate_tweets = (
        cascade["tweet_id"]
        .duplicated()
        .sum()
    )

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

    if (
        duplicate_tweets
        or missing_parents
        or self_parenting
    ):

        print(
            "WARNING: cascade failed "
            "integrity checks."
        )

        continue

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = simulate(
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
        f"Baseline reach: "
        f"{baseline_reach:,}"
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    for timing, fraction in CHECKPOINTS.items():

        checkpoint_index = int(
            len(cascade) * fraction
        )

        checkpoint_index = max(
            1,
            min(
                checkpoint_index,
                len(cascade) - 1,
            ),
        )

        observed = cascade.iloc[
            :checkpoint_index + 1
        ]

        # ----------------------------------------------------
        # Dynamic targets
        # ----------------------------------------------------

        targets = select_targets(
            observed
        )

        # ----------------------------------------------------
        # Strategies
        # ----------------------------------------------------

        strategies = [
            (
                "fact_checking",
                [],
            ),
            (
                "content_moderation",
                [],
            ),
            (
                "targeted_intervention",
                targets,
            ),
        ]

        for strategy, strategy_targets in strategies:

            result = simulate(
                cascade,
                strategy,
                checkpoint_index,
                strategy_targets,
            )

            reach_reduction = (
                (
                    baseline_reach
                    -
                    result["final_reach"]
                )
                /
                baseline_reach
                *
                100
            )

            edge_reduction = (
                (
                    baseline_edges
                    -
                    result["remaining_edges"]
                )
                /
                baseline_edges
                *
                100
                if baseline_edges > 0
                else 0
            )

            all_results.append(
                {
                    "claim_number":
                        claim_id,

                    "cascade_size":
                        len(cascade),

                    "timing":
                        timing,

                    "strategy":
                        strategy,

                    "targets":
                        len(
                            strategy_targets
                        ),

                    "baseline_reach":
                        baseline_reach,

                    "final_reach":
                        result[
                            "final_reach"
                        ],

                    "reach_reduction_percent":
                        reach_reduction,

                    "baseline_edges":
                        baseline_edges,

                    "remaining_edges":
                        result[
                            "remaining_edges"
                        ],

                    "edge_reduction_percent":
                        edge_reduction,

                    "max_depth":
                        result[
                            "max_depth"
                        ],
                }
            )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# CASCADE-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MULTI-CASCADE RESULTS")
print("=" * 80)

print(
    f"Valid cascades processed: "
    f"{results_df['claim_number'].nunique()}"
)

print(
    f"Experiment rows: "
    f"{len(results_df):,}"
)


# ============================================================
# AGGREGATED SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        ["strategy", "timing"]
    )[
        "reach_reduction_percent"
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("REACH REDUCTION SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# EDGE REDUCTION SUMMARY
# ============================================================

edge_summary = (
    results_df
    .groupby(
        ["strategy", "timing"]
    )[
        "edge_reduction_percent"
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
        ]
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("EDGE REDUCTION SUMMARY")
print("=" * 80)

print(
    edge_summary.to_string(
        index=False
    )
)


# ============================================================
# STRATEGY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("STRATEGY COMPARISON")
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

    print(
        f"\n{strategy.upper()}"
    )

    for timing in [
        "early",
        "mid",
        "late",
    ]:

        values = subset[
            subset["timing"]
            == timing
        ][
            "reach_reduction_percent"
        ]

        if len(values) == 0:

            continue

        print(
            f"  {timing:<5} "
            f"mean={values.mean():6.2f}% "
            f"median={values.median():6.2f}% "
            f"std={values.std():6.2f}%"
        )


# ============================================================
# TEMPORAL ROBUSTNESS
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL ROBUSTNESS CHECK")
print("=" * 80)

for strategy in [
    "fact_checking",
    "content_moderation",
    "targeted_intervention",
]:

    subset = (
        results_df[
            results_df["strategy"]
            == strategy
        ]
        .groupby("timing")
        [
            "reach_reduction_percent"
        ]
        .mean()
    )

    early = subset.get(
        "early",
        0,
    )

    mid = subset.get(
        "mid",
        0,
    )

    late = subset.get(
        "late",
        0,
    )

    print(
        f"{strategy:<25} "
        f"early={early:6.2f}% "
        f"mid={mid:6.2f}% "
        f"late={late:6.2f}%"
    )

    if early >= mid >= late:

        print(
            "  PASS: early >= mid >= late"
        )

    else:

        print(
            "  INFO: not strictly monotonic"
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

summary_file = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_multicascade_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 03 MULTI-CASCADE VALIDATION COMPLETE")
print("=" * 80)

print(
    "Detailed results:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSummary:"
)

print(
    summary_file
)

print("=" * 80)