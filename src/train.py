import pandas as pd
import pickle

model = pickle.load(open("models/model.pkl", "rb"))

importances = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
).sort_values(ascending=False)

print(importances.head(10))
