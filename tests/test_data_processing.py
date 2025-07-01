import pytest
from sklearn.ensemble import RandomForestClassifier
from src.train import evaluate

# Sample helper function for feature importance (can also be placed in a separate utils module)
def get_feature_importance(model, feature_names):
    return sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])

# Unit test for evaluate() function
def test_evaluate_output():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

# Unit test for get_feature_importance helper
def test_feature_importance():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    model.fit(X, y)
    importances = get_feature_importance(model, ["feat1", "feat2"])
    assert isinstance(importances, list)
    assert all(isinstance(i, tuple) and len(i) == 2 for i in importances)
    assert all(isinstance(i[0], str) and isinstance(i[1], float) for i in importances)