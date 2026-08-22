from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INFLUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fibvid_all_cascade_influence.csv"
)


CLAIM_ID = 281


print("=" * 80)
print("INTERVENTION TARGET VALIDATION")
print("=" * 80)


df = pd.read_csv(
    INFLUENCE_FILE
)


claim_df = df[
    df["claim_number"] == CLAIM_ID
].copy()


claim_df = claim_df.sort_values(
    "influence_score",
    ascending=False
)


print(
    f"\nClaim: {CLAIM_ID}"
)

print(
    f"Candidate users: "
    f"{len(claim_df):,}"
)


columns = [
    "influence_rank",
    "user_id",
    "network_role",
    "intervention_priority",
    "influence_score",
    "pagerank",
    "weighted_out_degree",
    "betweenness_centrality",
    "propagation_posts"
]


print("\nTOP 20 CANDIDATES")
print("=" * 80)

print(
    claim_df[
        columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("SELECTED INTERVENTION TARGETS")
print("=" * 80)

targets = (
    claim_df
    .head(10)
)

print(
    targets[
        columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)