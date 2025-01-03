# -*- coding: utf-8 -*-
"""
visualizer.py.
Random Forest Visualizer class to plot decision trees.
MIT License.
Author: @deburky
"""

import matplotlib.pyplot as plt


# pylint: disable=too-many-arguments
class RandomForestVisualizer:
    """
    Random Forest Visualizer class to plot decision trees.
    """
    def __init__(self, tree_data):
        """
        Initialize the tree visualizer with tree data.

        Parameters:
            tree_data (pd.DataFrame): DataFrame containing tree structure information.
        """
        self.tree_data = tree_data

    def construct_tree(self, tree_id):
        """
        Constructs a hierarchical tree structure from the flat DataFrame.

        Parameters:
            tree_id (int): The ID of the tree to construct.

        Returns:
            dict: Hierarchical tree structure.
        """
        # Filter data for the specific tree
        tree_df = self.tree_data.query(f"TreeID == {tree_id}")

        nodes = {
            row["NodeID"]: {
                "name": (
                    f"{row['Feature']}\n≤ {row['Threshold']:.2f}\nSamples={row['Samples']}"
                    if row["Feature"] != "Leaf"
                    else f"Leaf\nSamples={row['Samples']}\nP(Class 1)={row['PClass1']:.2f}"
                ),
                "children": {},
                "depth": None,
            }
            for _, row in tree_df.iterrows()
        }
        # Link parent and child nodes
        for _, row in tree_df.iterrows():
            if row["ChildType"] == "Left":
                nodes[row["NodeID"]]["children"]["Left"] = nodes.get(row["ChildNodeID"])
            elif row["ChildType"] == "Right":
                nodes[row["NodeID"]]["children"]["Right"] = nodes.get(
                    row["ChildNodeID"]
                )

        # Return the root node
        return nodes[0]

    def _draw_tree(
        self, node, depth, pos_x, level_distance, sibling_distance, yes_color, no_color
    ):
        """
        Recursively draw the tree structure, adding Yes/No labels on edges.
        """
        pos_y = -depth * level_distance

        # Draw the current node
        plt.text(
            pos_x,
            pos_y,
            node["name"],
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
        )

        # Handle children
        if "children" in node and node["children"]:
            child_offset = sibling_distance / (2 ** (depth + 1))
            left_x = pos_x - child_offset
            right_x = pos_x + child_offset

            if "Left" in node["children"]:
                # Draw left edge
                plt.plot(
                    [pos_x, left_x], [pos_y, pos_y - level_distance], color=yes_color
                )
                # Add "Yes" label to the left edge
                plt.text(
                    (pos_x + left_x) / 2 - 1,
                    (pos_y + pos_y - level_distance) / 2,
                    "Yes",
                    color=yes_color,
                    fontsize=10,
                    ha="center",
                )
                self._draw_tree(
                    node["children"]["Left"],
                    depth + 1,
                    left_x,
                    level_distance,
                    sibling_distance,
                    yes_color,
                    no_color,
                )

            if "Right" in node["children"]:
                # Draw right edge
                plt.plot(
                    [pos_x, right_x], [pos_y, pos_y - level_distance], color=no_color
                )
                # Add "No" label to the right edge
                plt.text(
                    (pos_x + right_x) / 2 + 1,
                    (pos_y + pos_y - level_distance) / 2,
                    "No",
                    color=no_color,
                    fontsize=10,
                    ha="center",
                )
                self._draw_tree(
                    node["children"]["Right"],
                    depth + 1,
                    right_x,
                    level_distance,
                    sibling_distance,
                    yes_color,
                    no_color,
                )
    def plot_tree(
        self,
        tree_id,
        figsize=(14, 10),
        level_distance=2,
        sibling_distance=5,
        yes_color="dodgerblue",
        no_color="orange",
        title=None,
    ): 
        """
        Plot a tree structure.

        Parameters:
            tree_id (int): The ID of the tree to plot.
            figsize (tuple): The size of the plot.
            level_distance (float): The vertical distance between levels.
            sibling_distance (float): The horizontal distance between siblings.
            yes_color (str): Color for "Yes" branches.
            no_color (str): Color for "No" branches.
        """
        tree = self.construct_tree(tree_id)

        # Set up the plot
        plt.figure(figsize=figsize)
        self._draw_tree(
            tree,
            depth=0,
            pos_x=0,
            level_distance=level_distance,
            sibling_distance=sibling_distance,
            yes_color=yes_color,
            no_color=no_color,
        )
        plt.title(title or f"Random Forest Decision Tree {tree_id + 1}", y=1.1, fontsize=16)
        plt.axis("off")
        plt.show()