import pandas as pd
import yaml
import os

params = yaml.safe_load(open("params.yaml"))

cleaned_path = params["data"]["cleaned_path"]

df = pd.read_csv(cleaned_path)

# Fault columns (multi-label → single label)
fault_columns = [
    'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
    'Dirtiness', 'Bumps', 'Other_Faults'
]

# Create binary target: 1 = defective, 0 = good
df["Defective"] = df[fault_columns].max(axis=1)

# Features = drop fault columns
X = df.drop(columns=fault_columns + ["Defective"])
y = df["Defective"]

os.makedirs("data/processed", exist_ok=True)

X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)

print("[FEATURES] Binary target 'Defective' created successfully")
