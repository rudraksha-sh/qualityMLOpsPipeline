import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Important features shown to user
# --------------------------------------------------
IMPORTANT_FEATURES = [
    "Pixels_Areas",
    "Sum_of_Luminosity",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "LogOfAreas",
    "SigmoidOfAreas"
]

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Product Quality Prediction", layout="wide")
st.title("🏭 Product Quality / Defect Prediction")

# --------------------------------------------------
# Sidebar (always visible)
# --------------------------------------------------
st.sidebar.title("📊 Classification Details")
st.sidebar.info("Click **Predict Quality** to see results")

# --------------------------------------------------
# Load reference data (used for defaults + report)
# --------------------------------------------------
X_ref = pd.read_csv("data/processed/X.csv")
y_ref = pd.read_csv("data/processed/y.csv")

feature_means = X_ref.mean()

# --------------------------------------------------
# MAIN INPUT AREA
# --------------------------------------------------
st.subheader("🔢 Enter Key Feature Values")

input_data = {}

for feature in IMPORTANT_FEATURES:
    input_data[feature] = st.number_input(
        label=feature,
        value=float(feature_means[feature]),
        step=0.01
    )

# Auto-fill remaining features
for feature in model.feature_names_in_:
    if feature not in input_data:
        input_data[feature] = float(feature_means[feature])

# Ensure correct column order
input_df = pd.DataFrame([input_data])[model.feature_names_in_]

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Quality"):

    # ---- Predict probabilities safely ----
    probs = model.predict_proba(input_df)[0]
    class_probs = dict(zip(model.classes_, probs))

    prob_good = class_probs.get(0, 0.0)
    prob_defective = class_probs.get(1, 0.0)

    # ---- Threshold-based decision (important fix) ----
    THRESHOLD = 0.65
    prediction = 1 if prob_defective >= THRESHOLD else 0

    # ---- Main Output ----
    if prediction == 1:
        st.error("❌ Defective Product Detected")
    else:
        st.success("✅ Good Quality Product")

    # --------------------------------------------------
    # SIDEBAR DETAILS
    # --------------------------------------------------
    st.sidebar.subheader("🔍 Prediction Probabilities")
    st.sidebar.metric("Good Quality Probability", f"{prob_good:.2%}")
    st.sidebar.metric("Defective Probability", f"{prob_defective:.2%}")
    st.sidebar.metric("Decision Threshold", THRESHOLD)

    st.sidebar.subheader("📌 Final Decision")
    if prediction == 1:
        st.sidebar.error("DEFECTIVE")
    else:
        st.sidebar.success("GOOD QUALITY")

    # --------------------------------------------------
    # MODEL CLASSIFICATION REPORT (SAFE)
    # --------------------------------------------------
    st.sidebar.subheader("📄 Model Performance Report")

    y_pred_all = model.predict(X_ref)

    try:
        report = classification_report(
            y_ref,
            y_pred_all,
            labels=[0, 1],
            target_names=["Good", "Defective"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose().round(3)
        st.sidebar.dataframe(report_df)

    except ValueError:
        st.sidebar.warning(
            "⚠️ Classification report unavailable (model predicted only one class)."
        )

    st.sidebar.caption("RandomForest • Binary Defect Classification")

st.caption("Powered by Streamlit • Explainable ML • End-to-End MLOps Pipeline")
