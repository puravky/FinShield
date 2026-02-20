from pathlib import Path

import joblib
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAMES = (
    "finshield_model.pkl",
    "finshield_lgbm.pkl",
    "finsheild_lgbm.pkl",  # legacy typo from notebook exports
)
SEARCH_DIRS = (APP_DIR, APP_DIR / "models", Path.cwd(), Path.cwd() / "models")


@st.cache_resource
def load_model():
    candidates = []
    seen = set()
    for directory in SEARCH_DIRS:
        for filename in MODEL_FILENAMES:
            path = directory / filename
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)

    errors = []
    for model_path in candidates:
        if not model_path.exists():
            continue
        try:
            return joblib.load(model_path), model_path.name
        except Exception as exc:
            errors.append(f"{model_path}: {exc}")

    if errors:
        detail = "\n\n".join(errors)
    else:
        attempted = "\n".join(str(path) for path in candidates)
        detail = f"No model file found. Checked:\n{attempted}"

    st.error("Unable to load a model. Ensure a .pkl model file is committed to the repo.")
    st.code(detail)
    st.stop()


model, model_name = load_model()

st.title("💳 FinShield - Fraud Risk Detector")
st.caption(f"Loaded model: {model_name}")
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
    st.error("High Risk Transaction 🚨")
elif proba > 0.5:
    st.warning("Medium Risk Transaction ⚠️")
else:
    st.success("Low Risk Transaction ✳️")
