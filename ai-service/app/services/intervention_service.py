from pathlib import Path

import pandas as pd


class InterventionService:

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.experiment_root = (
            self.project_root
            / "experiments"
        )

        self.causal_path = (
            self.experiment_root
            / "experiment_03_causal_intervention_v2.csv"
        )

        self.validation_path = (
            self.experiment_root
            / "experiment_03_multicascade_validation.csv"
        )

        self.summary_path = (
            self.experiment_root
            / "experiment_03_multicascade_summary.csv"
        )

        # ====================================================
        # VALIDATE EXPERIMENT FILES
        # ====================================================

        for path in [
            self.causal_path,
            self.validation_path,
            self.summary_path
        ]:

            if not path.exists():

                raise FileNotFoundError(
                    f"Experiment file not found: {path}"
                )

        # ====================================================
        # LOAD EXPERIMENT DATA
        # ====================================================

        self.causal_data = pd.read_csv(
            self.causal_path
        )

        self.validation_data = pd.read_csv(
            self.validation_path
        )

        self.summary_data = pd.read_csv(
            self.summary_path
        )

        print(
            "Intervention experiment data loaded successfully:"
        )

        print(
            f"  Causal rows: {len(self.causal_data)}"
        )

        print(
            f"  Validation rows: {len(self.validation_data)}"
        )

        print(
            f"  Summary rows: {len(self.summary_data)}"
        )


    # ========================================================
    # MULTI-CASCADE SUMMARY
    # ========================================================

    def get_summary(self):

        results = []

        for _, row in self.summary_data.iterrows():

            results.append({

                "strategy":
                    str(row["strategy"]),

                "timing":
                    str(row["timing"]),

                "mean":
                    round(
                        float(row["mean"]),
                        4
                    ),

                "median":
                    round(
                        float(row["median"]),
                        4
                    ),

                "std":
                    round(
                        float(row["std"]),
                        4
                    ),

                "min":
                    round(
                        float(row["min"]),
                        4
                    ),

                "max":
                    round(
                        float(row["max"]),
                        4
                    )
            })

        return {

            "experiment":
                "Experiment 03 — Multi-Cascade Causal Validation",

            "cascades_validated":
                int(
                    self.validation_data[
                        "claim_number"
                    ].nunique()
                ),

            "metric":
                "reach_reduction_percent",

            "results":
                results
        }


    # ========================================================
    # SINGLE-CASCADE CAUSAL RESULTS
    # ========================================================

    def get_causal_results(
        self,
        cascade_id: int
    ):

        # Current causal V2 experiment
        # was performed for Claim 281.

        if cascade_id != 281:

            raise ValueError(
                "Causal intervention results are "
                "currently available for cascade 281."
            )

        data = self.causal_data.copy()

        # ====================================================
        # FIND BASELINE
        # ====================================================

        baseline_row = data[
            data["strategy"] == "none"
        ]

        if baseline_row.empty:

            raise ValueError(
                "Baseline intervention result not found."
            )

        baseline_reach = int(
            baseline_row.iloc[0]["final_reach"]
        )

        baseline_edges = int(
            baseline_row.iloc[0]["remaining_edges"]
        )

        # ====================================================
        # INTERVENTION RESULTS
        # ====================================================

        results = []

        intervention_rows = data[
            data["strategy"] != "none"
        ]

        for _, row in intervention_rows.iterrows():

            results.append({

                "strategy":
                    str(row["strategy"]),

                "timing":
                    str(row["timing"]),

                "targets":
                    int(row["targets"]),

                "final_reach":
                    int(row["final_reach"]),

                "remaining_edges":
                    int(row["remaining_edges"]),

                "max_depth":
                    int(row["max_depth"]),

                "reach_reduction_percent":
                    round(
                        float(
                            row[
                                "reach_reduction_percent"
                            ]
                        ),
                        4
                    ),

                "edge_reduction_percent":
                    round(
                        float(
                            row[
                                "edge_reduction_percent"
                            ]
                        ),
                        4
                    )
            })

        return {

            "experiment":
                "Experiment 03 — Causal Dynamic Intervention V2",

            "cascade_id":
                cascade_id,

            "baseline_reach":
                baseline_reach,

            "baseline_edges":
                baseline_edges,

            "results":
                results
        }


    # ========================================================
    # MULTI-CASCADE DETAILED RESULTS
    # ========================================================

    def get_cascade_results(
        self,
        cascade_id: int
    ):

        data = self.validation_data[
            self.validation_data[
                "claim_number"
            ] == cascade_id
        ]

        if data.empty:

            raise ValueError(
                f"No intervention results found "
                f"for cascade {cascade_id}."
            )

        records = []

        for _, row in data.iterrows():

            records.append({

                "claim_number":
                    int(row["claim_number"]),

                "cascade_size":
                    int(row["cascade_size"]),

                "timing":
                    str(row["timing"]),

                "strategy":
                    str(row["strategy"]),

                "targets":
                    int(row["targets"]),

                "baseline_reach":
                    int(row["baseline_reach"]),

                "final_reach":
                    int(row["final_reach"]),

                "reach_reduction_percent":
                    round(
                        float(
                            row[
                                "reach_reduction_percent"
                            ]
                        ),
                        4
                    ),

                "baseline_edges":
                    int(row["baseline_edges"]),

                "remaining_edges":
                    int(row["remaining_edges"]),

                "edge_reduction_percent":
                    round(
                        float(
                            row[
                                "edge_reduction_percent"
                            ]
                        ),
                        4
                    ),

                "max_depth":
                    int(row["max_depth"])
            })

        return {

            "cascade_id":
                cascade_id,

            "records":
                records
        }


# ============================================================
# SERVICE INSTANCE
# ============================================================

intervention_service = InterventionService()