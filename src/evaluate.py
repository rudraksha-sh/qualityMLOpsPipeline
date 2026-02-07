import pandas as pd
import pickle
import mlflow
from sklearn.metrics import accuracy_score, f1_score

X_test = pd.read_csv("data/processed/X.csv")
y_test = pd.read_csv("data/processed/y.csv")

model = pickle.load(open("models/model.pkl", "rb"))
y_pred = model.predict(X_test)

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
