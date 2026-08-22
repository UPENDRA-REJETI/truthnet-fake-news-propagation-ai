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
    / "fibvid_refined_influence_users.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("REFINED FIBVID INFLUENCE ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

print(
    f"Propagation records: {len(df):,}"
)


# ============================================================
# SELECT LARGEST CASCADE
# ============================================================

cascade_sizes = (
    df.groupby("claim_number")
    .size()
    .sort_values(
        ascending=False
    )
)

claim_id = cascade_sizes.index[0]

cascade = df[
    df["claim_number"] == claim_id
].copy()

print("\nCascade:")
print(f"Claim: {claim_id}")
print(f"Records: {len(cascade):,}")


# ============================================================
# BUILD WEIGHTED USER GRAPH
# ============================================================

print("\nBuilding weighted influence graph...")

edges = cascade[
    cascade["parent_user"] != 0
].copy()

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

    source = int(row.parent_user)
    target = int(row.tweet_user)
    weight = int(row.interaction_count)

    G.add_edge(
        source,
        target,
        weight=weight
    )

print(
    f"Nodes: {G.number_of_nodes():,}"
)

print(
    f"Edges: {G.number_of_edges():,}"
)


# ============================================================
# CENTRALITY
# ============================================================

print("\nCalculating PageRank...")

pagerank = nx.pagerank(
    G,
    alpha=0.85,
    weight="weight"
)

print("Calculating betweenness...")

betweenness = nx.betweenness_centrality(
    G,
    normalized=True
)

print("Calculating weighted out-degree...")

weighted_out_degree = dict(
    G.out_degree(
        weight="weight"
    )
)

print("Calculating propagation activity...")

propagation_posts = (
    cascade
    .groupby("tweet_user")
    .size()
    .to_dict()
)


# ============================================================
# BUILD TABLE
# ============================================================

records = []

for user in G.nodes():

    records.append(
        {
            "user_id": int(user),

            "pagerank":
                pagerank.get(
                    user,
                    0
                ),

            "betweenness_centrality":
                betweenness.get(
                    user,
                    0
                ),

            "weighted_out_degree":
                weighted_out_degree.get(
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


# ============================================================
# NORMALIZATION
# ============================================================

score_columns = [
    "pagerank",
    "betweenness_centrality",
    "weighted_out_degree",
    "propagation_posts"
]

for column in score_columns:

    min_value = result[column].min()
    max_value = result[column].max()

    if max_value > min_value:

        result[
            f"{column}_normalized"
        ] = (
            result[column] - min_value
        ) / (
            max_value - min_value
        )

    else:

        result[
            f"{column}_normalized"
        ] = 0.0


# ============================================================
# REFINED INFLUENCE SCORE
# ============================================================

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


# ============================================================
# ROLE CLASSIFICATION
# ============================================================

def percentile(series, value):

    return (
        series <= value
    ).mean()


def classify_role(row):

    pagerank_percentile = percentile(
        result["pagerank"],
        row["pagerank"]
    )

    out_degree_percentile = percentile(
        result["weighted_out_degree"],
        row["weighted_out_degree"]
    )

    betweenness_percentile = percentile(
        result["betweenness_centrality"],
        row["betweenness_centrality"]
    )

    influence_percentile = percentile(
        result["influence_score"],
        row["influence_score"]
    )

    if influence_percentile >= 0.95:

        return "High-Influence Node"

    if out_degree_percentile >= 0.95:

        return "Amplifier"

    if betweenness_percentile >= 0.95:

        return "Bridge"

    if pagerank_percentile >= 0.95:

        return "High-Reach Node"

    return "Participant"


result["network_role"] = result.apply(
    classify_role,
    axis=1
)


# ============================================================
# RANK
# ============================================================

result = result.sort_values(
    "influence_score",
    ascending=False
).reset_index(
    drop=True
)

result[
    "influence_rank"
] = result.index + 1


# ============================================================
# DISPLAY
# ============================================================

display_columns = [
    "influence_rank",
    "user_id",
    "network_role",
    "weighted_out_degree",
    "propagation_posts",
    "pagerank",
    "betweenness_centrality",
    "influence_score"
]

print("\n" + "=" * 80)
print(f"TOP {TOP_N} INFLUENTIAL USERS")
print("=" * 80)

print(
    result[
        display_columns
    ]
    .head(TOP_N)
    .to_string(
        index=False
    )
)


# ============================================================
# ROLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("NETWORK ROLE DISTRIBUTION")
print("=" * 80)

print(
    result["network_role"]
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

result[
    display_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("REFINED INFLUENCE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("=" * 80)