from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import UPFD

from app.models.graphsage import GraphSAGE


class PredictionService:

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.data_root = (
            self.project_root
            / "data"
            / "raw"
            / "upfd"
        )

        self.model_path = (
            self.project_root
            / "models"
            / "graphsage_gossipcop.pt"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dataset = UPFD(
            root=str(self.data_root),
            name="gossipcop",
            feature="profile",
            split="test"
        )

        self.model = GraphSAGE(
            input_dim=self.dataset.num_features,
            hidden_dim=64,
            num_classes=2
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device
            )
        )

        self.model.eval()

    def predict_graph(self, graph_index: int):

        if graph_index < 0 or graph_index >= len(self.dataset):
            raise ValueError(
                f"graph_index must be between "
                f"0 and {len(self.dataset) - 1}"
            )

        graph = self.dataset[graph_index]

        batch = Batch.from_data_list(
            [graph]
        ).to(self.device)

        with torch.no_grad():

            logits = self.model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            predicted_class = int(
                probabilities.argmax().item()
            )

        return {
            "prediction": (
                "Fake"
                if predicted_class == 1
                else "Real"
            ),
            "class_id": predicted_class,
            "real_probability": round(
                float(probabilities[0]),
                4
            ),
            "fake_probability": round(
                float(probabilities[1]),
                4
            ),
            "graph_index": graph_index,
            "model": "GraphSAGE",
            "dataset": "UPFD-GossipCop"
        }


prediction_service = PredictionService()