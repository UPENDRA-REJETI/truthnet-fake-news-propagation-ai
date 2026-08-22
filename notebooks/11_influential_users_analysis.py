from pathlib import Path

import pandas as pd
import networkx as nx


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fibvid_influential_users.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TOP_N = 20


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("FIBVID INFLUENTIAL USER ANALYSIS")
print("=" * 80)

print("\nLoading propagation data...")

df = pd.read_csv(
    PROPAGATION_FILE,
    low_memory=False
)

print(
    f"Propagation records: {len(df):,}"
)


# ============================================================
# 4. SELECT A REPRESENTATIVE CASCADE
# ============================================================

cascade_sizes = (
    df.groupby("claim_number")
    .size()
    .sort_values(
        ascending=False
    )
)

largest_claim = cascade_sizes.index[0]

largest_size = cascade_sizes.iloc[0]

print("\nLargest cascade:")
print(f"Claim: {largest_claim}")
print(f"Records: {largest_size:,}")


cascade = df[
    df["claim_number"] == largest_claim
].copy()


# ============================================================
# 5. BUILD DIRECTED GRAPH
# ============================================================

print("\nBuilding propagation graph...")

G = nx.DiGraph()


for row in cascade.itertuples():

    user = int(row.tweet_user)

    parent = int(row.parent_user)

    # Root/origin node
    if parent == 0:
        G.add_node(user)
        continue

    G.add_edge(
        parent,
        user
    )


print(
    f"Graph nodes: {G.number_of_nodes():,}"
)

print(
    f"Graph edges: {G.number_of_edges():,}"
)


# ============================================================
# 6. BASIC GRAPH STATISTICS
# ============================================================

print("\nGraph statistics:")

print(
    f"Density: "
    f"{nx.density(G):.6f}"
)

print(
    f"Connected components: "
    f"{nx.number_weakly_connected_components(G)}"
)


# ============================================================
# 7. DEGREE CENTRALITY
# ============================================================

print("\nCalculating degree centrality...")

degree_centrality = nx.degree_centrality(
    G
)


# ============================================================
# 8. PAGERANK
# ============================================================

print("Calculating PageRank...")

pagerank = nx.pagerank(
    G,
    alpha=0.85
)


# ============================================================
# 9. BETWEENNESS CENTRALITY
# ============================================================

print(
    "Calculating betweenness centrality..."
)

betweenness = nx.betweenness_centrality(
    G,
    normalized=True
)


# ============================================================
# 10. COMBINE SCORES
# ============================================================

users = set(G.nodes())

results = []

for user in users:

    results.append(
        {
            "user_id": user,

            "degree_centrality":
                degree_centrality.get(
                    user,
                    0
                ),

            "pagerank":
                pagerank.get(
                    user,
                    0
                ),

            "betweenness_centrality":
                betweenness.get(
                    user,
                    0
                )
        }
    )


influence_df = pd.DataFrame(
    results
)


# ============================================================
# 11. NORMALIZE COMPONENTS
# ============================================================

score_columns = [
    "degree_centrality",
    "pagerank",
    "betweenness_centrality"
]

for column in score_columns:

    min_value = influence_df[
        column
    ].min()

    max_value = influence_df[
        column
    ].max()

    if max_value > min_value:

        influence_df[
            f"{column}_normalized"
        ] = (
            influence_df[column]
            - min_value
        ) / (
            max_value
            - min_value
        )

    else:

        influence_df[
            f"{column}_normalized"
        ] = 0.0


# ============================================================
# 12. COMPOSITE INFLUENCE SCORE
# ============================================================

influence_df[
    "influence_score"
] = (

    0.30
    * influence_df[
        "degree_centrality_normalized"
    ]

    +

    0.40
    * influence_df[
        "pagerank_normalized"
    ]

    +

    0.30
    * influence_df[
        "betweenness_centrality_normalized"
    ]
)


# ============================================================
# 13. SORT
# ============================================================

influence_df = influence_df.sort_values(
    "influence_score",
    ascending=False
).reset_index(
    drop=True
)

influence_df[
    "influence_rank"
] = (
    influence_df.index + 1
)


# ============================================================
# 14. DISPLAY TOP USERS
# ============================================================

print("\n" + "=" * 80)
print(f"TOP {TOP_N} INFLUENTIAL USERS")
print("=" * 80)

display_columns = [
    "influence_rank",
    "user_id",
    "degree_centrality",
    "pagerank",
    "betweenness_centrality",
    "influence_score"
]

print(
    influence_df[
        display_columns
    ]
    .head(TOP_N)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE
# ============================================================

influence_df[
    display_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("INFLUENCE ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print("=" * 80)