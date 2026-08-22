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

random.seed(
    RANDOM_SEED
)

# Number of intervention targets
TARGET_COUNT = 10

# Probability that a propagation edge survives
# under each strategy.
STRATEGY_REDUCTION = {
    "none": 0.00,
    "fact_checking": 0.25,
    "content_moderation": 0.40,
    "targeted_intervention": 0.60
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("INTERVENTION SIMULATION")
print("=" * 80)

propagation = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

influence = pd.read_csv(
    INFLUENCE_FILE
)

print(
    f"Propagation records: "
    f"{len(propagation):,}"
)

print(
    f"Influence records: "
    f"{len(influence):,}"
)


# ============================================================
# SELECT LARGEST CASCADE
# ============================================================

cascade_sizes = (
    propagation
    .groupby("claim_number")
    .size()
    .sort_values(
        ascending=False
    )
)

claim_id = cascade_sizes.index[0]

cascade = propagation[
    propagation["claim_number"]
    == claim_id
].copy()

print("\nSelected cascade:")
print(f"Claim: {claim_id}")
print(f"Records: {len(cascade):,}")


# ============================================================
# BUILD PROPAGATION GRAPH
# ============================================================

print("\nBuilding propagation graph...")

G = nx.DiGraph()

for row in cascade.itertuples():

    user = int(
        row.tweet_user
    )

    parent = int(
        row.parent_user
    )

    if parent == 0:

        G.add_node(
            user
        )

    else:

        G.add_edge(
            parent,
            user
        )


print(
    f"Nodes: {G.number_of_nodes():,}"
)

print(
    f"Edges: {G.number_of_edges():,}"
)


# ============================================================
# GET INFLUENTIAL USERS
# ============================================================

cascade_influence = influence[
    influence["claim_number"]
    == claim_id
].copy()

cascade_influence = (
    cascade_influence
    .sort_values(
        "influence_score",
        ascending=False
    )
)


target_users = (
    cascade_influence
    .head(TARGET_COUNT)
    ["user_id"]
    .astype(int)
    .tolist()
)

print("\nTarget users:")

for user in target_users:

    print(
        f"  {user}"
    )


# ============================================================
# SIMULATION FUNCTION
# ============================================================

def simulate(
    graph,
    strategy,
    targets=None
):

    reduction = (
        STRATEGY_REDUCTION[
            strategy
        ]
    )

    targets = set(
        targets or []
    )

    surviving_edges = []

    for source, target in graph.edges():

        probability = 1.0

        # ----------------------------------------------------
        # Global strategies
        # ----------------------------------------------------

        if strategy != "none":

            probability *= (
                1 - reduction
            )

        # ----------------------------------------------------
        # Targeted intervention
        # ----------------------------------------------------

        if (
            strategy
            == "targeted_intervention"
            and source in targets
        ):

            probability *= (
                1 - reduction
            )

        # ----------------------------------------------------
        # Propagation decision
        # ----------------------------------------------------

        if random.random() < probability:

            surviving_edges.append(
                (
                    source,
                    target
                )
            )

    simulated_graph = nx.DiGraph()

    simulated_graph.add_edges_from(
        surviving_edges
    )

    # Add isolated original nodes
    simulated_graph.add_nodes_from(
        graph.nodes()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    active_nodes = [
        node
        for node in simulated_graph.nodes()
        if (
            simulated_graph.in_degree(node)
            > 0
            or graph.in_degree(node)
            == 0
        )
    ]

    final_reach = len(
        active_nodes
    )

    if simulated_graph.number_of_edges():

        max_depth = 0

        roots = [
            node
            for node in simulated_graph.nodes()
            if simulated_graph.in_degree(node) == 0
        ]

        for root in roots:

            try:

                lengths = nx.single_source_shortest_path_length(
                    simulated_graph,
                    root
                )

                if lengths:

                    max_depth = max(
                        max_depth,
                        max(
                            lengths.values()
                        )
                    )

            except nx.NetworkXError:

                continue

    else:

        max_depth = 0

    return {
        "strategy": strategy,
        "final_reach": final_reach,
        "remaining_edges":
            simulated_graph.number_of_edges(),
        "max_depth": max_depth
    }


# ============================================================
# RUN SIMULATIONS
# ============================================================

strategies = [
    "none",
    "fact_checking",
    "content_moderation",
    "targeted_intervention"
]

results = []

print("\n" + "=" * 80)
print("SIMULATION RESULTS")
print("=" * 80)

for strategy in strategies:

    result = simulate(
        G,
        strategy,
        target_users
    )

    results.append(
        result
    )

    print(
        f"\n{strategy}"
    )

    print(
        f"Final reach: "
        f"{result['final_reach']}"
    )

    print(
        f"Remaining edges: "
        f"{result['remaining_edges']}"
    )

    print(
        f"Maximum depth: "
        f"{result['max_depth']}"
    )


# ============================================================
# BASELINE COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

baseline_reach = (
    results_df
    .loc[
        results_df["strategy"]
        == "none",
        "final_reach"
    ]
    .iloc[0]
)

results_df[
    "reach_reduction_percent"
] = (
    (
        baseline_reach
        - results_df["final_reach"]
    )
    /
    baseline_reach
    * 100
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("INTERVENTION COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)