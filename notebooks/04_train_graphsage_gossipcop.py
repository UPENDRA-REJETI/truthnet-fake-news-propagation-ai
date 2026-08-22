from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "upfd"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "graphsage_gossipcop.pt"


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GRAPH SAGE — UPFD GOSSIPCOP")
print("=" * 70)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading UPFD GossipCop...")

train_dataset = UPFD(
    root=str(DATA_ROOT),
    name="gossipcop",
    feature="profile",
    split="train"
)

val_dataset = UPFD(
    root=str(DATA_ROOT),
    name="gossipcop",
    feature="profile",
    split="val"
)

test_dataset = UPFD(
    root=str(DATA_ROOT),
    name="gossipcop",
    feature="profile",
    split="test"
)

print(f"Train graphs: {len(train_dataset)}")
print(f"Validation graphs: {len(val_dataset)}")
print(f"Test graphs: {len(test_dataset)}")

print(f"Node features: {train_dataset.num_features}")
print(f"Classes: {train_dataset.num_classes}")


# ============================================================
# 5. DATA LOADERS
# ============================================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. GRAPH SAGE MODEL
# ============================================================

class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_classes=2
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.classifier = torch.nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x, edge_index, batch):

        # GraphSAGE layer 1
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        # GraphSAGE layer 2
        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # Convert node representations
        # into graph representation
        x = global_mean_pool(
            x,
            batch
        )

        # Graph classification
        x = self.classifier(x)

        return x


# ============================================================
# 7. MODEL
# ============================================================

model = GraphSAGE(
    input_dim=train_dataset.num_features,
    hidden_dim=64,
    num_classes=2
).to(device)

print("\nModel:")
print(model)


# ============================================================
# 8. OPTIMIZER
# ============================================================

LEARNING_RATE = 0.001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. TRAIN FUNCTION
# ============================================================

def train():

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = F.cross_entropy(
            output,
            batch.y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * batch.num_graphs
        )

        total_graphs += batch.num_graphs

    return total_loss / total_graphs


# ============================================================
# 10. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    correct = 0
    total = 0

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        predictions = output.argmax(
            dim=1
        )

        correct += (
            predictions == batch.y
        ).sum().item()

        total += batch.num_graphs

    return correct / total


# ============================================================
# 11. TRAINING LOOP
# ============================================================

EPOCHS = 30

best_val_accuracy = 0.0

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    loss = train()

    train_accuracy = evaluate(
        train_loader
    )

    val_accuracy = evaluate(
        val_loader
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Loss: {loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )


# ============================================================
# 12. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 13. FINAL TEST
# ============================================================

test_accuracy = evaluate(
    test_loader
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Model saved to:\n{MODEL_PATH}"
)

print("=" * 70)