import pandas as pd
import yaml
import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load params
params = yaml.safe_load(open("params.yaml"))

# Load data
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)

# MLflow setup
mlflow.set_experiment("Quality_Defect_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["training"]["random_state"]
    )

    model.fit(X_train, y_train.values.ravel())

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("n_estimators", params["model"]["n_estimators"])
    mlflow.log_param("max_depth", params["model"]["max_depth"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact("models/model.pkl")

    print(f"[TRAIN] Model trained | Accuracy: {acc:.4f}, F1: {f1:.4f}")
