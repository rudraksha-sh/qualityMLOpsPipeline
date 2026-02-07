import pandas as pd
import yaml
import os

# Load parameters
params = yaml.safe_load(open("params.yaml"))

# Paths
raw_path = params["data"]["raw_path"]
loaded_path = params["data"]["loaded_path"]

# Read raw dataset
df = pd.read_csv(raw_path)

# Ensure output directory exists
os.makedirs(os.path.dirname(loaded_path), exist_ok=True)

# Save loaded data
df.to_csv(loaded_path, index=False)

print(f"[LOAD] Data saved to {loaded_path}")
