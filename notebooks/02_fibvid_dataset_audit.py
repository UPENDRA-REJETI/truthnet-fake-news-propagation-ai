from pathlib import Path
import pandas as pd


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


# ============================================================
# FILES
# ============================================================

FILES = {
    "news_claim": FIBVID_ROOT / "news_claim" / "news_claim.csv",
    "origin_tweet": FIBVID_ROOT / "claim_propagation" / "origin_tweet.csv",
    "claim_propagation": FIBVID_ROOT / "claim_propagation" / "claim_propagation.csv",
    "user_information": FIBVID_ROOT / "user_information" / "user_information.csv",
}


# ============================================================
# BASIC FILE AUDIT
# ============================================================

def inspect_file(name, path):

    print("\n" + "=" * 80)
    print(f"FILE: {name}")
    print("=" * 80)

    print(f"Path: {path}")

    if not path.exists():
        print("ERROR: File does not exist!")
        return

    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"File size: {size_mb:.2f} MB")

    # Read only a small sample first.
    sample = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    for column in sample.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(sample.dtypes)

    print("\nFirst 5 rows:")
    print(sample.to_string(index=False))

    # Full row count without loading everything into memory.
    row_count = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1

    print(f"\nTotal rows: {row_count}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("FIBVID DATASET AUDIT")
    print("=" * 80)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nFibVID root:")
    print(FIBVID_ROOT)

    for name, path in FILES.items():
        inspect_file(name, path)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)