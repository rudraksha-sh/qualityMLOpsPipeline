import pandas as pd
import yaml
import os

params = yaml.safe_load(open("params.yaml"))

loaded_path = params["data"]["loaded_path"]
cleaned_path = params["data"]["cleaned_path"]

df = pd.read_csv(loaded_path)

# Cleaning logic
df = df.dropna()

os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
df.to_csv(cleaned_path, index=False)

print(f"[PREPROCESS] Cleaned data saved to {cleaned_path}")
