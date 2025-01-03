"""
analyzer.py.
Random Forest Analyzer class to analyze and visualize decision trees.
MIT License.
Author: @deburky
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier


# pylint: disable=too-many-instance-attributes, too-many-locals, line-too-long, invalid-name, cell-var-from-loop
class RandomForestAnalyzer:
    """
    A class to analyze and visualize decision trees in a RandomForestClassifier.
    """

    def __init__(
        self,
        rf_model: RandomForestClassifier,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
    ):
        self.rf_model = rf_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.tree_data = None
        self.leaf_data = None

    def extract_tree_data_with_conditions(self):
        """
        Extracts detailed information about all trees in the RandomForestClassifier,
        including split conditions and child relationships.
        """
        tree_data = []

        for tree_id, tree in enumerate(self.rf_model.estimators_):
            tree_ = tree.tree_

            for node_id in range(tree_.node_count):
                class_values = tree_.value[node_id].flatten()
                class_distributions = {
                    f"PClass{i}": class_values[i] for i in range(len(class_values))
                }

                if (
                    tree_.children_left[node_id] != tree_.children_right[node_id]
                ):  # Non-leaf node
                    left_condition = "<="
                    right_condition = ">"

                    tree_data.extend(
                        [
                            {
                                "TreeID": tree_id,
                                "NodeID": node_id,
                                "Feature": self.feature_names[tree_.feature[node_id]],
                                "Condition": left_condition,
                                "Threshold": tree_.threshold[node_id],
                                "Impurity": tree_.impurity[node_id],
                                "Samples": tree_.n_node_samples[node_id],
                                **class_distributions,
                                "ChildType": "Left",
                                "ChildNodeID": tree_.children_left[node_id],
                            },
                            {
                                "TreeID": tree_id,
                                "NodeID": node_id,
                                "Feature": self.feature_names[tree_.feature[node_id]],
                                "Condition": right_condition,
                                "Threshold": tree_.threshold[node_id],
                                "Impurity": tree_.impurity[node_id],
                                "Samples": tree_.n_node_samples[node_id],
                                **class_distributions,
                                "ChildType": "Right",
                                "ChildNodeID": tree_.children_right[node_id],
                            },
                        ]
                    )
                else:  # Leaf node
                    tree_data.append(
                        {
                            "TreeID": tree_id,
                            "NodeID": node_id,
                            "Feature": "Leaf",
                            "Condition": None,
                            "Threshold": None,
                            "Impurity": tree_.impurity[node_id],
                            "Samples": tree_.n_node_samples[node_id],
                            **class_distributions,
                            "ChildType": None,
                            "ChildNodeID": None,
                        }
                    )

        self.tree_data = pd.DataFrame(tree_data)
        return self.tree_data

    def extract_leaf_nodes_with_conditions(self):
        """
        Extracts detailed information about leaf nodes and their corresponding path conditions.
        """
        leaf_data = []

        for tree_id, tree in enumerate(self.rf_model.estimators_):
            tree_ = tree.tree_

            def trace_conditions(node_id, path_conditions):
                if (
                    tree_.children_left[node_id] == tree_.children_right[node_id]
                ):  # Leaf node
                    class_values = tree_.value[node_id].flatten()
                    class_distributions = {
                        f"PClass{i}": class_values[i] for i in range(len(class_values))
                    }

                    leaf_data.append(
                        {
                            "TreeID": tree_id,
                            "NodeID": node_id,
                            "Condition": " and ".join(path_conditions),
                            "Impurity": tree_.impurity[node_id],
                            "Samples": tree_.n_node_samples[node_id],
                            **class_distributions,
                        }
                    )
                else:  # Non-leaf node
                    feature = self.feature_names[tree_.feature[node_id]]
                    threshold = tree_.threshold[node_id]

                    # Trace left and right paths
                    trace_conditions(
                        tree_.children_left[node_id],
                        path_conditions + [f"{feature} <= {threshold:.4f}"],
                    )
                    trace_conditions(
                        tree_.children_right[node_id],
                        path_conditions + [f"{feature} > {threshold:.4f}"],
                    )

            trace_conditions(0, [])  # Start from the root

        self.leaf_data = pd.DataFrame(leaf_data)
        return self.leaf_data

    def predict_from_tree(
        self, data_point: Dict[str, float], tree_id: int
    ) -> Dict[str, float]:
        """
        Predict the class distribution for a single data point using a decision tree.

        Parameters:
            data_point: A dictionary of feature values (e.g., {"Income": 45}).
            tree_id: The ID of the tree to use for prediction.

        Returns:
            A dictionary with the class distribution at the leaf node.
        """
        if self.tree_data is None:
            raise ValueError(
                "Tree data is not extracted. Run extract_tree_data_with_conditions() first."
            )

        current_node = 0
        tree_df = self.tree_data.query(f"TreeID == {tree_id}")

        while True:
            node_data = tree_df.query(f"NodeID == {current_node}").iloc[0]

            if node_data["Feature"] == "Leaf":
                return {
                    f"p{i}": node_data[f"PClass{i}"]
                    for i in range(len(self.class_names or []))
                }

            feature = node_data["Feature"]
            threshold = node_data["Threshold"]

            if data_point[feature] <= threshold:
                current_node = int(node_data["ChildNodeID"])  # Move to left child
            else:
                current_node = int(
                    tree_df.query(f"NodeID == {current_node} and ChildType == 'Right'")[
                        "ChildNodeID"
                    ].iloc[0]
                )

    def prediction_score(
        self, rf: RandomForestClassifier, X: np.ndarray, dict: Optional[Dict] = None
    ):
        # sourcery skip: avoid-builtin-shadow
        """
        Calculate scaled prediction scores for each tree in the random forest and average them.

        Parameters:
            rf: Trained RandomForestClassifier object.
            X: Feature matrix (NumPy array or Pandas DataFrame).
            dict: Dictionary containing parameters for scaling (target_points, pdo, target_odds).

        Returns:
            Numpy array of averaged scores across all trees.
        """
        if dict is None:
            dict = {'target_points': 500, 'pdo': 20, 'target_odds': 19}

        # Clip probabilities to avoid log(0)
        X = np.array(X)  # Ensure X is a NumPy array
        n_samples = X.shape[0]
        all_tree_scores = np.zeros((len(rf.estimators_), n_samples))

        for tree_id, tree in enumerate(rf.estimators_):
            # Get probabilities for the current tree
            probas = tree.predict_proba(X)[:, 1]
            probas = np.clip(probas, 1e-5, 1 - 1e-5)
            prior = np.mean(probas)

            # Compute WoE for each sample
            woes = -np.log((probas / prior) / ((1 - probas) / (1 - prior)))

            # Compute scores
            factor = dict['pdo'] / np.log(2)
            offset = dict['target_points'] - factor * np.log(dict['target_odds'])
            scores = offset + factor * woes

            # Store scores for the current tree
            all_tree_scores[tree_id] = scores

        return np.mean(all_tree_scores, axis=0)

    def print_tree(self, tree_id: int):
        """
        Visualizes a decision tree structure with added probabilities and metrics.

        Parameters:
            tree_id: The ID of the tree to visualize.
        """
        if self.tree_data is None:
            raise ValueError(
                "Tree data is not extracted. Run extract_tree_data_with_conditions() first."
            )

        console = Console()

        # Filter the data for the specific tree
        tree_df = self.tree_data.query(f"TreeID == {tree_id}")

        # Recursive function to build the enriched tree
        def build_tree(node_id, depth=0, parent_condition=None):
            node_data = tree_df[tree_df["NodeID"] == node_id].iloc[0]
            feature = node_data["Feature"]
            threshold = node_data["Threshold"]
            samples = node_data["Samples"]
            impurity = node_data["Impurity"]
            class_1_prob = node_data["PClass1"]

            # Build the node description
            if feature == "Leaf":
                # Leaf inherits the parent's condition
                condition_label = "Yes" if parent_condition == "<=" else "No"
                description = (
                    f"{'|   ' * depth}[bold]Leaf ({condition_label})[/bold] "
                    f"[Samples: {samples}, Impurity: {impurity:.4f}, "
                    f"P(C=1): {class_1_prob:.2f}]"
                )
            else:
                description = (
                    f"{'|   ' * depth}[bold][yellow]{feature}[/yellow][/bold] <= {threshold:.2f} "
                    f"[Samples: {samples}, Impurity: {impurity:.4f}, "
                    f"P(C=1): {class_1_prob:.2f}]"
                )

            # Find children and recursively build their descriptions
            children = tree_df[tree_df["NodeID"] == node_id]
            child_descriptions = []
            for _, child in children.iterrows():
                if child["ChildType"] == "Left":
                    child_descriptions.append(build_tree(child["ChildNodeID"], depth + 1, parent_condition="<="))
                elif child["ChildType"] == "Right":
                    child_descriptions.append(build_tree(child["ChildNodeID"], depth + 1, parent_condition=">"))

            # Combine current node and its children
            return f"{description}\n" + "".join(child_descriptions)

        # Find the root node (NodeID == 0)
        root_node_id = tree_df["NodeID"].min()
        enriched_tree = build_tree(root_node_id)

        # Display the enriched tree
        console.print(enriched_tree, style="dim cyan")