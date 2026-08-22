from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
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

CLAIM_FILE = (
    FIBVID_ROOT
    / "news_claim"
    / "news_claim.csv"
)

USER_FILE = (
    FIBVID_ROOT
    / "user_information"
    / "user_information.csv"
)

ORIGIN_FILE = (
    FIBVID_ROOT
    / "claim_propagation"
    / "origin_tweet.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CLAIM AUDIT
# ============================================================

def audit_claims():

    print_section("1. CLAIM AUDIT")

    claims = pd.read_csv(CLAIM_FILE)

    print(f"Total claims: {len(claims)}")

    print("\nClaim columns:")
    print(list(claims.columns))

    print("\nGroup distribution:")

    group_counts = claims["group"].value_counts().sort_index()

    for group, count in group_counts.items():
        percentage = count / len(claims) * 100
        print(
            f"  Group {group}: "
            f"{count} claims "
            f"({percentage:.2f}%)"
        )

    print("\nSource distribution:")

    print(claims["source"].value_counts())

    print("\nMissing values:")

    print(claims.isna().sum())


# ============================================================
# PROPAGATION AUDIT
# ============================================================

def audit_propagation():

    print_section("2. PROPAGATION AUDIT")

    propagation = pd.read_csv(PROPAGATION_FILE)

    print(f"Total propagation records: {len(propagation):,}")

    print("\nUnique values:")

    print(
        f"Unique claims: "
        f"{propagation['claim_number'].nunique():,}"
    )

    print(
        f"Unique tweets: "
        f"{propagation['tweet_id'].nunique():,}"
    )

    print(
        f"Unique users: "
        f"{propagation['tweet_user'].nunique():,}"
    )

    print("\nPropagation depth:")

    print(
        f"Minimum depth: "
        f"{propagation['depth'].min()}"
    )

    print(
        f"Maximum depth: "
        f"{propagation['depth'].max()}"
    )

    print(
        f"Mean depth: "
        f"{propagation['depth'].mean():.2f}"
    )

    print(
        f"Median depth: "
        f"{propagation['depth'].median():.2f}"
    )

    print("\nDepth distribution:")

    depth_counts = (
        propagation["depth"]
        .value_counts()
        .sort_index()
    )

    print(depth_counts.to_string())

    return propagation


# ============================================================
# CASCADE AUDIT
# ============================================================

def audit_cascades(propagation):

    print_section("3. CASCADE SIZE AUDIT")

    cascade_sizes = (
        propagation
        .groupby("claim_number")
        .size()
    )

    print(
        f"Number of claims with propagation data: "
        f"{len(cascade_sizes):,}"
    )

    print(
        f"Minimum cascade size: "
        f"{cascade_sizes.min()}"
    )

    print(
        f"Maximum cascade size: "
        f"{cascade_sizes.max()}"
    )

    print(
        f"Mean cascade size: "
        f"{cascade_sizes.mean():.2f}"
    )

    print(
        f"Median cascade size: "
        f"{cascade_sizes.median():.2f}"
    )

    print("\nCascade size percentiles:")

    percentiles = cascade_sizes.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    for percentile, value in percentiles.items():
        print(
            f"  {percentile * 100:.0f}th percentile: "
            f"{value:.2f}"
        )


# ============================================================
# TEMPORAL AUDIT
# ============================================================

def audit_time(propagation):

    print_section("4. TEMPORAL AUDIT")

    propagation["create_date_parsed"] = pd.to_datetime(
        propagation["create_date"],
        errors="coerce",
        utc=True
    )

    invalid = propagation["create_date_parsed"].isna().sum()

    print(
        f"Invalid timestamps: {invalid:,}"
    )

    print(
        f"Earliest timestamp: "
        f"{propagation['create_date_parsed'].min()}"
    )

    print(
        f"Latest timestamp: "
        f"{propagation['create_date_parsed'].max()}"
    )

    print(
        f"Total time span: "
        f"{propagation['create_date_parsed'].max() - propagation['create_date_parsed'].min()}"
    )

    print("\nRecords by year:")

    print(
        propagation["create_date_parsed"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# PARENT-CHILD GRAPH AUDIT
# ============================================================

def audit_graph_integrity(propagation):

    print_section("5. GRAPH INTEGRITY AUDIT")

    # Root records
    root_records = propagation[
        (propagation["parent_id"] == 0)
        & (propagation["parent_user"] == 0)
    ]

    print(
        f"Records with parent_id=0 and parent_user=0: "
        f"{len(root_records):,}"
    )

    # Self-parenting
    self_parent = propagation[
        propagation["tweet_id"] == propagation["parent_id"]
    ]

    print(
        f"Self-parenting records: "
        f"{len(self_parent):,}"
    )

    # Missing parent tweets inside propagation dataset
    tweet_ids = set(propagation["tweet_id"])

    nonzero_parent_ids = propagation.loc[
        propagation["parent_id"] != 0,
        "parent_id"
    ]

    missing_parents = (
        ~nonzero_parent_ids.isin(tweet_ids)
    )

    print(
        f"Non-root parent IDs not found in propagation data: "
        f"{missing_parents.sum():,}"
    )

    # Duplicate tweet IDs
    duplicate_tweets = (
        propagation["tweet_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate tweet ID records: "
        f"{duplicate_tweets:,}"
    )


# ============================================================
# USER COVERAGE AUDIT
# ============================================================

def audit_users(propagation):

    print_section("6. USER COVERAGE AUDIT")

    users = pd.read_csv(USER_FILE)

    propagation_users = set(
        propagation["tweet_user"].unique()
    )

    information_users = set(
        users["user_id"].unique()
    )

    matched = propagation_users.intersection(
        information_users
    )

    missing = propagation_users - information_users

    print(
        f"Unique propagation users: "
        f"{len(propagation_users):,}"
    )

    print(
        f"Users with profile information: "
        f"{len(information_users):,}"
    )

    print(
        f"Matched propagation users: "
        f"{len(matched):,}"
    )

    print(
        f"Propagation users without profile information: "
        f"{len(missing):,}"
    )

    coverage = (
        len(matched)
        / len(propagation_users)
        * 100
    )

    print(
        f"User feature coverage: "
        f"{coverage:.2f}%"
    )

    print("\nUser feature missing values:")

    print(users.isna().sum())


# ============================================================
# CLAIM COVERAGE
# ============================================================

def audit_claim_coverage(propagation):

    print_section("7. CLAIM COVERAGE AUDIT")

    claims = pd.read_csv(CLAIM_FILE)

    claim_ids = set(
        claims["claim_num"].unique()
    )

    propagation_claims = set(
        propagation["claim_number"].unique()
    )

    matched = claim_ids.intersection(
        propagation_claims
    )

    missing = claim_ids - propagation_claims

    print(
        f"Total claims: "
        f"{len(claim_ids):,}"
    )

    print(
        f"Claims with propagation records: "
        f"{len(matched):,}"
    )

    print(
        f"Claims without propagation records: "
        f"{len(missing):,}"
    )

    coverage = (
        len(matched)
        / len(claim_ids)
        * 100
    )

    print(
        f"Propagation coverage: "
        f"{coverage:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("FIBVID PROPAGATION AND GRAPH INTEGRITY AUDIT")
    print("=" * 80)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nFibVID root:")
    print(FIBVID_ROOT)

    print(f"\nPropagation file:")
    print(PROPAGATION_FILE)

    # 1
    audit_claims()

    # 2
    propagation = audit_propagation()

    # 3
    audit_cascades(propagation)

    # 4
    audit_time(propagation)

    # 5
    audit_graph_integrity(propagation)

    # 6
    audit_users(propagation)

    # 7
    audit_claim_coverage(propagation)

    print_section("AUDIT COMPLETE")

    print("FibVID propagation audit completed successfully.")