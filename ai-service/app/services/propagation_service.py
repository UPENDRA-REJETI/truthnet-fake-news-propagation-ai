from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np


class PropagationService:

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.model_root = (
            self.project_root
            / "models"
            / "propagation"
        )

        self.metadata_path = (
            self.model_root
            / "metadata.json"
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        self.models = {}

        for window in ["1", "6", "24"]:

            model_path = (
                self.project_root
                / self.metadata["windows"][window]["model_file"]
            )

            self.models[window] = joblib.load(
                model_path
            )

        print(
            "Propagation models loaded successfully:"
        )

        for window in self.models:

            print(
                f"  {window}h → "
                f"{self.metadata['windows'][window]['model_file']}"
            )


    def predict(
        self,
        window: int,
        early_tweets: int,
        early_users: int,
        early_max_depth: int,
        early_likes: int,
        early_retweets: int,
        engagement: int
    ):

        window_key = str(window)

        if window_key not in self.models:

            raise ValueError(
                "window must be one of: 1, 6, 24"
            )

        window_metadata = (
            self.metadata["windows"][window_key]
        )

        features = (
            window_metadata["features"]
        )

        values = {
            f"early_tweets_{window}h":
                early_tweets,

            f"early_users_{window}h":
                early_users,

            f"early_max_depth_{window}h":
                early_max_depth,

            f"early_likes_{window}h":
                early_likes,

            f"early_retweets_{window}h":
                early_retweets,

            f"engagement_{window}h":
                engagement
        }

        input_data = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features
        )

        model = self.models[window_key]

        predicted_log = float(
            model.predict(input_data)[0]
        )

        predicted_size = float(
            np.expm1(predicted_log)
        )

        predicted_size = max(
            1.0,
            predicted_size
        )

        return {
            "window_hours": window,

            "predicted_cascade_size": round(
                predicted_size,
                2
            ),

            "predicted_log_cascade_size": round(
                predicted_log,
                4
            ),

            "model":
                self.metadata["model"],

            "dataset":
                self.metadata["dataset"],

            "features_used":
                features
        }


propagation_service = PropagationService()