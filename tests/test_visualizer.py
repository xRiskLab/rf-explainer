"""
Test module for visualizer.py.

This module contains unit tests for the RandomForestVisualizer class.
"""

import pytest
import pandas as pd
from rf_explainer.visualizer import RandomForestVisualizer
import matplotlib.pyplot as plt
from unittest.mock import patch

@pytest.fixture
def mock_tree_data():
    """
    Mock tree data for testing.
    """
    return pd.DataFrame({
        "TreeID": [0, 0, 0, 0, 0],
        "NodeID": [0, 1, 2, 3, 4],
        "Feature": ["Income", "Balance", "Leaf", "Leaf", "Leaf"],
        "Threshold": [50, 30, None, None, None],
        "Samples": [100, 60, 40, 20, 40],
        "PClass1": [0.6, 0.7, 0.0, 0.8, 0.2],
        "ChildType": [None, "Left", None, None, None],
        "ChildNodeID": [1, 2, None, None, None]
    })

def test_construct_tree(mock_tree_data):
    """
    Test case for the construct_tree method.
    """
    visualizer = RandomForestVisualizer(mock_tree_data)
    tree = visualizer.construct_tree(tree_id=0)

    assert isinstance(tree, dict)
    assert "name" in tree
    assert "children" in tree
    assert tree["name"] == "Income\n≤ 50.00\nSamples=100"

def test_plot_tree(mock_tree_data):
    """
    Test case for the plot_tree method.
    """
    visualizer = RandomForestVisualizer(mock_tree_data)

    with patch("matplotlib.pyplot.show") as mock_show:
        visualizer.plot_tree(
            tree_id=0,
            figsize=(10, 8),
            level_distance=1.5,
            sibling_distance=2.0,
            yes_color="blue",
            no_color="red",
            title="Mock Tree Visualization",
        )
        mock_show.assert_called_once()

def test_draw_tree_edge_cases(mock_tree_data):
    """
    Test edge cases in tree visualization.
    """
    visualizer = RandomForestVisualizer(mock_tree_data)

    with patch("matplotlib.pyplot.show") as mock_show:
        # Test with a tree that has only root
        single_node_tree_data = mock_tree_data[mock_tree_data["NodeID"] == 0]
        visualizer.tree_data = single_node_tree_data
        visualizer.plot_tree(tree_id=0, title="Single Node Tree")
        mock_show.assert_called_once()
