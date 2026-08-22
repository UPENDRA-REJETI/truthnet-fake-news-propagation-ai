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

USER_FILE = (
    FIBVID_ROOT
    / "user_information"
    / "user_information.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fibvid_weighted_influence_users.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FIBVID WEIGHTED INFLUENCE ANALYSIS")
print("=" * 80)

print("\nLoading propagation data...")

df = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

print(
    f"Propagation records: {len(df):,}"
)


print("\nLoading user profiles...")

users = pd.read_csv(
    USER_FILE,
    low_memory=False
)

print(
    f"User profiles: {len(users):,}"
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

largest_claim = cascade_sizes.index[0]

cascade = df[
    df["claim_number"] == largest_claim
].copy()

print("\nLargest cascade:")
print(f"Claim: {largest_claim}")
print(f"Records: {len(cascade):,}")


# ============================================================
# BUILD WEIGHTED USER GRAPH
# ============================================================

print("\nBuilding weighted user graph...")

# Remove root records
edges = cascade[
    cascade["parent_user"] != 0
].copy()

# Aggregate repeated user-to-user interactions
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
    f"Graph nodes: {G.number_of_nodes():,}"
)

print(
    f"Graph edges: {G.number_of_edges():,}"
)


# ============================================================
# WEIGHTED DEGREE
# ============================================================

print("\nCalculating weighted degree...")

weighted_out_degree = dict(
    G.out_degree(
        weight="weight"
    )
)

weighted_in_degree = dict(
    G.in_degree(
        weight="weight"
    )
)


# ============================================================
# PAGERANK
# ============================================================

print("Calculating weighted PageRank...")

pagerank = nx.pagerank(
    G,
    alpha=0.85,
    weight="weight"
)


# ============================================================
# BETWEENNESS
# ============================================================

print(
    "Calculating betweenness centrality..."
)

betweenness = nx.betweenness_centrality(
    G,
    normalized=True
)


# ============================================================
# USER ACTIVITY
# ============================================================

print("\nCalculating propagation activity...")

tweet_counts = (
    cascade
    .groupby("tweet_user")
    .size()
    .to_dict()
)


# ============================================================
# BUILD RESULT TABLE
# ============================================================

records = []

for user_id in G.nodes():

    records.append(
        {
            "user_id": int(user_id),

            "weighted_out_degree":
                weighted_out_degree.get(
                    user_id,
                    0
                ),

            "weighted_in_degree":
                weighted_in_degree.get(
                    user_id,
                    0
                ),

            "pagerank":
                pagerank.get(
                    user_id,
                    0
                ),

            "betweenness_centrality":
                betweenness.get(
                    user_id,
                    0
                ),

            "propagation_posts":
                tweet_counts.get(
                    user_id,
                    0
                )
        }
    )


result = pd.DataFrame(
    records
)


# ============================================================
# MERGE USER PROFILE INFORMATION
# ============================================================

profile_columns = [
    "user_id",
    "follower_count",
    "following_count"
]

result = result.merge(
    users[profile_columns],
    on="user_id",
    how="left"
)


# ============================================================
# NORMALIZATION
# ============================================================

score_columns = [
    "weighted_out_degree",
    "pagerank",
    "betweenness_centrality",
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


# Follower normalization
result["follower_count"] = (
    result["follower_count"]
    .fillna(0)
)

follower_min = result[
    "follower_count"
].min()

follower_max = result[
    "follower_count"
].max()

if follower_max > follower_min:

    result[
        "follower_count_normalized"
    ] = (
        result["follower_count"]
        - follower_min
    ) / (
        follower_max
        - follower_min
    )

else:

    result[
        "follower_count_normalized"
    ] = 0.0


# ============================================================
# COMPOSITE INFLUENCE SCORE
# ============================================================

result["influence_score"] = (

    0.30
    * result[
        "pagerank_normalized"
    ]

    +

    0.25
    * result[
        "weighted_out_degree_normalized"
    ]

    +

    0.20
    * result[
        "betweenness_centrality_normalized"
    ]

    +

    0.15
    * result[
        "propagation_posts_normalized"
    ]

    +

    0.10
    * result[
        "follower_count_normalized"
    ]
)


# ============================================================
# SORT AND RANK
# ============================================================

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


# ============================================================
# ROLE CLASSIFICATION
# ============================================================

def classify_role(row):

    if (
        row["weighted_out_degree_normalized"] > 0.7
        and row["propagation_posts_normalized"] > 0.5
    ):
        return "Amplifier"

    if (
        row["betweenness_centrality_normalized"] > 0.5
    ):
        return "Bridge"

    if (
        row["follower_count_normalized"] > 0.7
        and row["pagerank_normalized"] > 0.5
    ):
        return "High-Reach User"

    return "Participant"


result["network_role"] = result.apply(
    classify_role,
    axis=1
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print(f"TOP {TOP_N} INFLUENTIAL USERS")
print("=" * 80)

display_columns = [
    "influence_rank",
    "user_id",
    "network_role",
    "weighted_out_degree",
    "propagation_posts",
    "follower_count",
    "pagerank",
    "betweenness_centrality",
    "influence_score"
]

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
# SAVE
# ============================================================

result[
    display_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("WEIGHTED INFLUENCE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print("=" * 80)