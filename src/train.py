import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load processed data
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model (NOT load)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train.values.ravel())

# Evaluate quickly
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[TRAIN] Model trained | Accuracy: {acc:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
