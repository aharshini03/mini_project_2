import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load artifacts ---
@st.cache_data
def load_artifacts():
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return rf_model, scaler, selected_features, encoders

rf_model, scaler, selected_features, encoders = load_artifacts()

# --- App title ---
st.title("🧬 Cancer DSS Prediction (Mini-1)")
st.write("""
Upload all 5 Mini-1 CSV files (BRCA, CHOL, OV, PAAD, STAD). 
The DSS will merge them, preprocess features, and predict cancer types.
""")

# --- Upload files ---
uploaded_files = st.file_uploader(
    "Upload the 5 Mini-1 CSV files", type="csv", accept_multiple_files=True
)

if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.write("✅ Combined dataset preview:")
    st.dataframe(df.head())
    st.write(f"Total samples: {df.shape[0]}")
    
    # --- Preprocessing ---
    X_new = df[selected_features].copy()
    for col in X_new.select_dtypes(include='object').columns:
        if col in encoders:
            le = encoders[col]
            X_new[col] = le.transform(X_new[col])
    
    X_new_scaled = scaler.transform(X_new)
    st.write("✅ Features scaled. Ready for prediction.")
    
    # --- Prediction ---
    y_pred = rf_model.predict(X_new_scaled)
    cancer_types = ["BRCA", "CHOL", "OV", "PAAD", "STAD"]
    y_pred_labels = [cancer_types[i] for i in y_pred]
    df["Predicted_CancerType"] = y_pred_labels
    st.write("📊 Predictions (first 20 rows):")
    st.dataframe(df[["Case ID", "Project ID", "Predicted_CancerType"]].head(20))
    
    # --- Download predictions ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="mini1_predictions.csv",
        mime="text/csv"
    )
