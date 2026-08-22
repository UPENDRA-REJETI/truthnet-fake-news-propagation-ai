from pathlib import Path

import pandas as pd


class InfluenceService:

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.data_path = (
            self.project_root
            / "data"
            / "processed"
            / "fibvid_all_cascade_influence.csv"
        )

        self.data = pd.read_csv(
            self.data_path
        )

        print(
            "Influence data loaded successfully:"
        )

        print(
            f"  Rows: {len(self.data):,}"
        )

        print(
            f"  Cascades: "
            f"{self.data['claim_number'].nunique():,}"
        )

        print(
            f"  Users: "
            f"{self.data['user_id'].nunique():,}"
        )


    def get_influence(
        self,
        cascade_id: int = 281,
        limit: int = 20
    ):

        cascade = self.data[
            self.data["claim_number"] == cascade_id
        ].copy()

        if cascade.empty:

            raise ValueError(
                f"Cascade {cascade_id} not found"
            )

        cascade = cascade.sort_values(
            "influence_score",
            ascending=False
        )

        users = cascade.head(limit)

        results = []

        for _, row in users.iterrows():

            results.append({

                "user_id": int(
                    row["user_id"]
                ),

                "network_role": str(
                    row["network_role"]
                ),

                "intervention_priority": str(
                    row["intervention_priority"]
                ),

                "influence_score": round(
                    float(row["influence_score"]),
                    6
                ),

                "pagerank": round(
                    float(row["pagerank"]),
                    8
                ),

                "weighted_out_degree": float(
                    row["weighted_out_degree"]
                ),

                "betweenness_centrality": float(
                    row["betweenness_centrality"]
                ),

                "propagation_posts": int(
                    row["propagation_posts"]
                )
            })

        return {

            "cascade_id": cascade_id,

            "total_users": int(
                cascade["user_id"].nunique()
            ),

            "users": results
        }


influence_service = InfluenceService()