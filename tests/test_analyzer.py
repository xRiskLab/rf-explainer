import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from rf_explainer import RandomForestAnalyzer


@pytest.fixture
def rf_analyzer():
    # Create a random dataset and train a RandomForestClassifier
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # Initialize RandomForestAnalyzer
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    class_names = ["Class_0", "Class_1"]
    analyzer = RandomForestAnalyzer(rf, feature_names, class_names)
    return analyzer, X, y


def test_extract_tree_data_with_conditions(rf_analyzer):
    analyzer, X, y = rf_analyzer

    # Test tree data extraction
    tree_data = analyzer.extract_tree_data_with_conditions()
    assert not tree_data.empty, "Tree data should not be empty."
    assert "TreeID" in tree_data.columns, "TreeID column should exist in tree data."
    assert "NodeID" in tree_data.columns, "NodeID column should exist in tree data."


def test_extract_leaf_nodes_with_conditions(rf_analyzer):
    analyzer, X, y = rf_analyzer

    # Test leaf data extraction
    leaf_data = analyzer.extract_leaf_nodes_with_conditions()
    assert not leaf_data.empty, "Leaf data should not be empty."
    assert "Condition" in leaf_data.columns, "Condition column should exist in leaf data."
    assert "Samples" in leaf_data.columns, "Samples column should exist in leaf data."


def test_predict_from_tree(rf_analyzer):
    analyzer, X, y = rf_analyzer

    # Test predictions from a single tree
    analyzer.extract_tree_data_with_conditions()
    data_point = {f"Feature_{i}": X[0, i] for i in range(X.shape[1])}
    prediction = analyzer.predict_from_tree(data_point, tree_id=0)
    assert "p0" in prediction, "Prediction should include 'p0' for Class_0."
    assert "p1" in prediction, "Prediction should include 'p1' for Class_1."


def test_prediction_score(rf_analyzer):
    analyzer, X, y = rf_analyzer

    # Test scaled prediction scores
    scores = analyzer.prediction_score(analyzer.rf_model, X)
    assert scores.shape[0] == X.shape[0], "Scores should have one value per sample."
    assert np.all(scores > 0), "All scores should be positive."


def test_print_tree(rf_analyzer, capsys):
    analyzer, X, y = rf_analyzer

    # Test tree visualization
    analyzer.extract_tree_data_with_conditions()
    analyzer.print_tree(tree_id=0)
    captured = capsys.readouterr()
    assert "Leaf" in captured.out, "Tree visualization should include 'Leaf'."
    assert "Feature_" in captured.out, "Tree visualization should include features."

def test_handle_empty_tree_data():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    analyzer = RandomForestAnalyzer(rf, feature_names=[], class_names=[])

    # Test error handling when tree data is missing
    with pytest.raises(ValueError, match="Tree data is not extracted"):
        analyzer.print_tree(tree_id=0)

    with pytest.raises(ValueError, match="Tree data is not extracted"):
        analyzer.predict_from_tree({}, tree_id=0)
