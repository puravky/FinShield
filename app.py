from pathlib import Path

import joblib
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "finshield_model.pkl", 
    APP_DIR / "finshield_lgbm.pkl", 
]


def load_model():
    errors = []
    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue
        try:
            return joblib.load(model_path), model_path.name
        except Exception as exc:
            errors.append(f"{model_path.name}: {exc}")

    detail = "\n\n".join(errors) if errors else "No model file found."
    st.error("Unable to load a model. Please check model files and dependencies.")
    st.code(detail)
    st.stop()


model, model_name = load_model()

st.title("FinShield💳 - Fraud Risk Detector")
st.write("Enter transaction details:")

amount = st.number_input("Amount", min_value=0.0, value=0.0)
old_org = st.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
new_org = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
old_dest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
new_dest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

input_data = np.array([[amount, old_org, new_org, old_dest, new_dest]], dtype=float)
proba = float(model.predict_proba(input_data)[0][1])

st.subheader(f"Fraud Risk: {proba * 100:.2f}%")

if proba > 0.9:
    st.error("High Risk Transaction 🚫")
elif proba > 0.5:
    st.warning("Medium Risk Transaction ⚠️")
else:
    st.success("Low Risk Transaction ✅")
