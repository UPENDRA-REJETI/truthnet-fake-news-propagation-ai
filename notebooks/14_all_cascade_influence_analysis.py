from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fibvid_all_cascade_influence.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FIBVID ALL-CASCADE INFLUENCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

print(
    f"Propagation records: {len(df):,}"
)


# ============================================================
# GROUP BY CASCADE
# ============================================================

grouped = df.groupby(
    "claim_number"
)

claim_ids = list(
    grouped.groups.keys()
)

print(
    f"Propagation cascades: {len(claim_ids)}"
)


# ============================================================
# PROCESS EACH CASCADE
# ============================================================

all_results = []

for counter, claim_id in enumerate(
    claim_ids,
    start=1
):

    cascade = grouped.get_group(
        claim_id
    ).copy()

    # --------------------------------------------------------
    # Build weighted user graph
    # --------------------------------------------------------

    edges = cascade[
        cascade["parent_user"] != 0
    ]

    edge_data = (
        edges
        .groupby(
            [
                "parent_user",
                "tweet_user"
            ]
        )
        .size()
        .reset_index(
            name="interaction_count"
        )
    )

    G = nx.DiGraph()

    for row in edge_data.itertuples():

        source = int(
            row.parent_user
        )

        target = int(
            row.tweet_user
        )

        weight = int(
            row.interaction_count
        )

        G.add_edge(
            source,
            target,
            weight=weight
        )

    # --------------------------------------------------------
    # Handle tiny graphs
    # --------------------------------------------------------

    if G.number_of_nodes() == 0:

        continue

    # --------------------------------------------------------
    # Network metrics
    # --------------------------------------------------------

    pagerank = nx.pagerank(
        G,
        alpha=0.85,
        weight="weight"
    )

    weighted_out_degree = dict(
        G.out_degree(
            weight="weight"
        )
    )

    # Betweenness can become expensive.
    # For very large cascades we use sampling.
    node_count = G.number_of_nodes()

    if node_count > 5000:

        k = min(
            500,
            node_count
        )

        betweenness = (
            nx.betweenness_centrality(
                G,
                k=k,
                normalized=True,
                seed=42
            )
        )

    else:

        betweenness = (
            nx.betweenness_centrality(
                G,
                normalized=True
            )
        )

    propagation_posts = (
        cascade
        .groupby(
            "tweet_user"
        )
        .size()
        .to_dict()
    )

    # --------------------------------------------------------
    # Build metric table
    # --------------------------------------------------------

    records = []

    for user in G.nodes():

        records.append(
            {
                "claim_number": int(
                    claim_id
                ),

                "user_id": int(
                    user
                ),

                "pagerank": pagerank.get(
                    user,
                    0
                ),

                "weighted_out_degree":
                    weighted_out_degree.get(
                        user,
                        0
                    ),

                "betweenness_centrality":
                    betweenness.get(
                        user,
                        0
                    ),

                "propagation_posts":
                    propagation_posts.get(
                        user,
                        0
                    )
            }
        )

    result = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Normalize within this cascade
    # --------------------------------------------------------

    metric_columns = [
        "pagerank",
        "weighted_out_degree",
        "betweenness_centrality",
        "propagation_posts"
    ]

    for column in metric_columns:

        min_value = result[
            column
        ].min()

        max_value = result[
            column
        ].max()

        if max_value > min_value:

            result[
                f"{column}_normalized"
            ] = (
                result[column]
                - min_value
            ) / (
                max_value
                - min_value
            )

        else:

            result[
                f"{column}_normalized"
            ] = 0.0

    # --------------------------------------------------------
    # Influence score
    # --------------------------------------------------------

    result["influence_score"] = (

        0.35
        * result[
            "pagerank_normalized"
        ]

        +

        0.30
        * result[
            "weighted_out_degree_normalized"
        ]

        +

        0.25
        * result[
            "betweenness_centrality_normalized"
        ]

        +

        0.10
        * result[
            "propagation_posts_normalized"
        ]
    )

    # --------------------------------------------------------
    # Rank within cascade
    # --------------------------------------------------------

    result = result.sort_values(
        "influence_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    result[
        "influence_rank"
    ] = (
        result.index + 1
    )

    result[
        "influence_percentile"
    ] = (
        1
        -
        result.index
        /
        max(
            len(result) - 1,
            1
        )
    )

    # --------------------------------------------------------
    # Role classification
    # --------------------------------------------------------

    def classify_role(row):

        if (
            row["weighted_out_degree_normalized"]
            >= 0.90
        ):
            return "Amplifier"

        if (
            row["betweenness_centrality_normalized"]
            >= 0.90
            and row[
                "betweenness_centrality"
            ] > 0
        ):
            return "Bridge"

        if (
            row["pagerank_normalized"]
            >= 0.90
        ):
            return "High-Reach Node"

        return "Participant"

    result[
        "network_role"
    ] = result.apply(
        classify_role,
        axis=1
    )

    # --------------------------------------------------------
    # Intervention priority
    # --------------------------------------------------------

    def intervention_priority(row):

        percentile = (
            row[
                "influence_percentile"
            ]
        )

        if percentile >= 0.99:
            return "Critical"

        if percentile >= 0.95:
            return "High"

        if percentile >= 0.90:
            return "Moderate"

        return "Low"

    result[
        "intervention_priority"
    ] = result.apply(
        intervention_priority,
        axis=1
    )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    final_columns = [
        "claim_number",
        "user_id",
        "influence_rank",
        "influence_percentile",
        "influence_score",
        "network_role",
        "intervention_priority",
        "pagerank",
        "weighted_out_degree",
        "betweenness_centrality",
        "propagation_posts"
    ]

    all_results.append(
        result[
            final_columns
        ]
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        counter % 25 == 0
        or counter == len(claim_ids)
    ):

        print(
            f"Processed "
            f"{counter}/{len(claim_ids)} cascades"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

print("\nCombining cascade results...")

final_df = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ALL-CASCADE INFLUENCE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Rows: {len(final_df):,}"
)

print(
    f"Unique cascades: "
    f"{final_df['claim_number'].nunique()}"
)

print(
    f"Unique users: "
    f"{final_df['user_id'].nunique():,}"
)

print("\nRole distribution:")

print(
    final_df[
        "network_role"
    ]
    .value_counts()
    .to_string()
)

print("\nIntervention priority:")

print(
    final_df[
        "intervention_priority"
    ]
    .value_counts()
    .to_string()
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 80)