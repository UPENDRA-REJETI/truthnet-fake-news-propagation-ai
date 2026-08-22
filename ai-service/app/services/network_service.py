from pathlib import Path

import pandas as pd


class NetworkService:

    def __init__(self):
        self.project_root = (
            Path(__file__).resolve().parents[3]
        )

        self.data_path = (
            self.project_root
            / "data"
            / "raw"
            / "fibvid"
            / "extracted"
            / "merry555-FibVID-14b95c3"
            / "claim_propagation"
            / "claim_propagation.csv"
        )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Propagation dataset not found: {self.data_path}"
            )

        self.data = pd.read_csv(self.data_path)

        required_columns = {
            "tweet_user",
            "tweet_id",
            "like_count",
            "depth",
            "parent_user",
            "parent_id",
            "retweet_count",
            "claim_number",
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                "Missing required propagation columns: "
                + ", ".join(sorted(missing))
            )

        print(
            "Propagation network dataset loaded successfully:"
        )

        print(
            f"  Records: {len(self.data):,}"
        )

        print(
            f"  Claims: "
            f"{self.data['claim_number'].nunique():,}"
        )

    def get_network(
        self,
        cascade_id: int = 281,
        limit: int = 100,
    ):

        # --------------------------------------------------------
        # Select cascade
        # --------------------------------------------------------

        cascade = self.data[
            self.data["claim_number"] == cascade_id
        ].copy()

        if cascade.empty:
            raise ValueError(
                f"Cascade {cascade_id} not found"
            )

        total_records = len(cascade)

        # --------------------------------------------------------
        # Convert numeric columns
        # --------------------------------------------------------

        numeric_columns = [
            "tweet_user",
            "parent_user",
            "depth",
            "tweet_id",
            "parent_id",
            "like_count",
            "retweet_count",
        ]

        for column in numeric_columns:
            cascade[column] = pd.to_numeric(
                cascade[column],
                errors="coerce"
            ).fillna(0)

        # --------------------------------------------------------
        # Keep only real user -> user propagation
        # --------------------------------------------------------

        edges = cascade[
            (cascade["parent_user"] > 0)
            & (cascade["tweet_user"] > 0)
            & (
                cascade["tweet_user"]
                != cascade["parent_user"]
            )
        ].copy()

        if edges.empty:
            return {
                "cascade_id": cascade_id,
                "total_records": total_records,
                "total_edges": 0,
                "nodes": [],
                "edges": [],
            }

        # --------------------------------------------------------
        # Calculate engagement
        # --------------------------------------------------------

        edges["engagement"] = (
            edges["like_count"]
            + edges["retweet_count"]
        )

        # --------------------------------------------------------
        # Aggregate repeated user -> user relationships
        #
        # The raw dataset may contain many tweets between
        # the same pair of users. Keeping every tweet creates
        # visually repetitive star/ring structures.
        # --------------------------------------------------------

        pair_rows = []

        grouped = edges.groupby(
            ["parent_user", "tweet_user"],
            as_index=False
        )

        for (source, target), group in grouped:

            best_row = group.loc[
                group["engagement"].idxmax()
            ]

            pair_rows.append({
                "source": int(source),
                "target": int(target),

                "depth": int(
                    group["depth"].min()
                ),

                "tweet_id": int(
                    best_row["tweet_id"]
                ),

                "parent_id": int(
                    best_row["parent_id"]
                ),

                "likes": int(
                    group["like_count"].sum()
                ),

                "retweets": int(
                    group["retweet_count"].sum()
                ),

                "engagement": int(
                    group["engagement"].sum()
                ),

                "tweet_count": len(group),
            })

        pair_df = pd.DataFrame(pair_rows)

        # --------------------------------------------------------
        # Calculate user importance
        #
        # This is used ONLY to choose a good starting point
        # for visualization.
        # --------------------------------------------------------

        user_stats = {}

        for _, row in pair_df.iterrows():

            source = int(row["source"])
            target = int(row["target"])

            engagement = int(
                row["engagement"]
            )

            if source not in user_stats:
                user_stats[source] = {
                    "degree": 0,
                    "engagement": 0,
                }

            if target not in user_stats:
                user_stats[target] = {
                    "degree": 0,
                    "engagement": 0,
                }

            user_stats[source]["degree"] += 1
            user_stats[source]["engagement"] += engagement

            user_stats[target]["degree"] += 1
            user_stats[target]["engagement"] += engagement

        # --------------------------------------------------------
        # Select strongest starting node
        # --------------------------------------------------------

        start_user = max(
            user_stats,
            key=lambda user: (
                user_stats[user]["degree"],
                user_stats[user]["engagement"],
            )
        )

        # --------------------------------------------------------
        # Build adjacency structure
        #
        # Treat the graph as undirected ONLY while selecting
        # the connected visualization region.
        #
        # The actual returned edges remain directed.
        # --------------------------------------------------------

        adjacency = {}

        for _, row in pair_df.iterrows():

            source = int(row["source"])
            target = int(row["target"])

            adjacency.setdefault(
                source, []
            ).append(
                (target, row)
            )

            adjacency.setdefault(
                target, []
            ).append(
                (source, row)
            )

        # --------------------------------------------------------
        # Breadth-first expansion
        #
        # This guarantees that the selected nodes belong to
        # one connected propagation region.
        # --------------------------------------------------------

        selected_users = set()
        selected_edges = {}
        queue = [start_user]

        while queue and len(selected_users) < limit:

            current = queue.pop(0)

            if current in selected_users:
                continue

            selected_users.add(current)

            neighbors = adjacency.get(
                current,
                []
            )

            # Strongest relationships first
            neighbors = sorted(
                neighbors,
                key=lambda item: int(
                    item[1]["engagement"]
                ),
                reverse=True
            )

            for neighbor, row in neighbors:

                if len(selected_users) >= limit:
                    break

                edge_key = (
                    int(row["source"]),
                    int(row["target"])
                )

                selected_edges[
                    edge_key
                ] = row

                if neighbor not in selected_users:
                    queue.append(neighbor)

        # --------------------------------------------------------
        # Add edges connecting already selected nodes
        # --------------------------------------------------------

        for _, row in pair_df.iterrows():

            source = int(row["source"])
            target = int(row["target"])

            if (
                source in selected_users
                and target in selected_users
            ):

                edge_key = (
                    source,
                    target
                )

                selected_edges[
                    edge_key
                ] = row

        # --------------------------------------------------------
        # Limit edge count while preserving strongest
        # --------------------------------------------------------

        selected_edge_rows = list(
            selected_edges.values()
        )

        selected_edge_rows.sort(
            key=lambda row: int(
                row["engagement"]
            ),
            reverse=True
        )

        selected_edge_rows = (
            selected_edge_rows[:limit]
        )

        # --------------------------------------------------------
        # Build node statistics
        # --------------------------------------------------------

        node_rows = {}

        for row in selected_edge_rows:

            source = int(row["source"])
            target = int(row["target"])

            for user_id in [source, target]:

                if user_id not in node_rows:

                    node_rows[user_id] = {
                        "user_id": user_id,
                        "posts": 0,
                        "propagation_posts": 0,
                        "max_depth": 0,
                        "total_likes": 0,
                        "total_retweets": 0,
                        "connection_count": 0,
                    }

            # Source statistics
            node_rows[source]["posts"] += 1
            node_rows[source]["connection_count"] += 1

            # Target statistics
            node_rows[target]["posts"] += 1
            node_rows[target]["propagation_posts"] += 1
            node_rows[target]["connection_count"] += 1

            node_rows[target]["max_depth"] = max(
                node_rows[target]["max_depth"],
                int(row["depth"])
            )

            node_rows[target]["total_likes"] += int(
                row["likes"]
            )

            node_rows[target]["total_retweets"] += int(
                row["retweets"]
            )

        # --------------------------------------------------------
        # Sort nodes by importance
        # --------------------------------------------------------

        nodes = list(
            node_rows.values()
        )

        nodes.sort(
            key=lambda item: (
                item["connection_count"],
                item["propagation_posts"],
                item["total_retweets"],
                item["total_likes"],
            ),
            reverse=True,
        )

        # --------------------------------------------------------
        # Build frontend edges
        # --------------------------------------------------------

        result_edges = []

        for row in selected_edge_rows:

            result_edges.append({
                "source": int(
                    row["source"]
                ),

                "target": int(
                    row["target"]
                ),

                "depth": int(
                    row["depth"]
                ),

                "tweet_id": int(
                    row["tweet_id"]
                ),

                "parent_id": int(
                    row["parent_id"]
                ),

                "likes": int(
                    row["likes"]
                ),

                "retweets": int(
                    row["retweets"]
                ),

                "engagement": int(
                    row["engagement"]
                ),

                "tweet_count": int(
                    row["tweet_count"]
                ),
            })

        # --------------------------------------------------------
        # Final response
        # --------------------------------------------------------

        return {
            "cascade_id": cascade_id,
            "total_records": total_records,
            "total_edges": len(result_edges),
            "nodes": nodes,
            "edges": result_edges,
        }


network_service = NetworkService()