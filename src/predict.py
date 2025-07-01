import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

def main():
    # Load dataset
    df = pd.read_csv("../data/processed/final_dataset.csv")
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier()
    }

    # Train and log each model
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            results = evaluate(model, X_test, y_test)

            # Log params and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(results)
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"{name}_model")

    # Hyperparameter tuning for RandomForest
    rf_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(), rf_grid, scoring='f1', cv=3)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_results = evaluate(best_model, X_test, y_test)

    with mlflow.start_run(run_name="random_forest_gridsearch"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(best_results)
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_random_forest")

if __name__ == "__main__":
    main()
