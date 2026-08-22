from pathlib import Path

from torch_geometric.datasets import UPFD
import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "upfd"


def audit_dataset(name, feature="profile"):
    print("\n" + "=" * 60)
    print(f"DATASET: {name.upper()}")
    print("=" * 60)

    splits = {}

    for split in ["train", "val", "test"]:
        print(f"\nLoading {split} split...")

        dataset = UPFD(
            root=DATA_ROOT,
            name=name,
            feature=feature,
            split=split,
        )

        splits[split] = dataset

        node_counts = [graph.num_nodes for graph in dataset]
        edge_counts = [graph.num_edges for graph in dataset]
        labels = [int(graph.y.item()) for graph in dataset]

        print(f"Graphs: {len(dataset)}")
        print(f"Features per node: {dataset.num_features}")
        print(f"Classes: {dataset.num_classes}")

        print(
            f"Nodes - min: {min(node_counts)}, "
            f"max: {max(node_counts)}, "
            f"mean: {np.mean(node_counts):.2f}, "
            f"median: {np.median(node_counts):.2f}"
        )

        print(
            f"Edges - min: {min(edge_counts)}, "
            f"max: {max(edge_counts)}, "
            f"mean: {np.mean(edge_counts):.2f}, "
            f"median: {np.median(edge_counts):.2f}"
        )

        unique, counts = np.unique(labels, return_counts=True)

        print("Label distribution:")

        for label, count in zip(unique, counts):
            print(f"  Label {label}: {count}")

    return splits


if __name__ == "__main__":

    print("UPFD DATASET AUDIT")
    print("Feature type: profile")

    audit_dataset("gossipcop")