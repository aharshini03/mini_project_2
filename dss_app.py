import streamlit as st
import pandas as pd
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
st.title("🧬 Cancer Clinical DSS (Mini-2)")
st.write("""
Upload mutation CSV files (BRCA, CHOL, OV, PAAD, STAD).
The DSS will preprocess, scale, and predict cancer types using the Random Forest model.
""")

# --- Upload files ---
uploaded_files = st.file_uploader(
    "Upload CSV files (can upload multiple cancer types)", 
    type="csv", accept_multiple_files=True
)

if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.write("✅ Dataset preview:")
    st.dataframe(df.head())
    st.write(f"Total samples: {df.shape[0]}")

    # --- Preprocessing ---
    X_new = df.copy()

    # keep only selected features
    valid_features = [f for f in selected_features if f in X_new.columns]
    X_new = X_new[valid_features]

    # encode categorical
    for col in X_new.select_dtypes(include='object').columns:
        if col in encoders:
            le = encoders[col]
            X_new[col] = le.transform(X_new[col].astype(str))

    # scale
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
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
