import streamlit as st
import pandas as pd
import joblib
import json
import sys
import os

# Ensure 'src' folder is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.preprocessing import preprocess_data

# 1. Page Configuration
st.set_page_config(page_title="Kidney Disease AI", layout="wide")

st.title("🏥 Kidney Disease Prediction System")
st.markdown("Enter the patient's lab results below to analyze the risk of Chronic Kidney Disease.")

# 2. Load Model and Feature Names
@st.cache_resource # Cache the model to optimize performance (load once)
def load_resources():
    model = joblib.load('models/kidney_model.joblib')
    with open('models/model_features.json', 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

try:
    model, feature_names = load_resources()
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'python src/train.py' first.")
    st.stop()

# 3. Input Form (prevents prediction until button is pressed)
with st.form("prediction_form"):
    st.header("Patient Lab Results")
    
    # Create a two-column layout for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Blood & Inflammation")
        wbc = st.number_input("White blood cell count (cells/cumm)", min_value=0.0, value=8000.0)
        crp = st.number_input("C-reactive protein (CRP) level", min_value=0.0, value=5.0)
        il6 = st.number_input("Interleukin-6 (IL-6) level", min_value=0.0, value=2.0)
        glucose = st.number_input("Random blood glucose level (mg/dl)", min_value=0.0, value=100.0)
        
        st.subheader("Kidney Function Tests")
        egfr = st.number_input("Estimated Glomerular Filtration Rate (eGFR)", min_value=0.0, value=90.0)
        creatinine = st.number_input("Serum creatinine (mg/dl)", min_value=0.0, value=1.0)
        urea = st.number_input("Blood urea (mg/dl)", min_value=0.0, value=30.0)
        cystatin = st.number_input("Cystatin C level", min_value=0.0, value=0.8)

    with col2:
        st.subheader("Electrolytes")
        sodium = st.number_input("Sodium level (mEq/L)", min_value=0.0, value=140.0)
        potassium = st.number_input("Potassium level (mEq/L)", min_value=0.0, value=4.0)
        calcium = st.number_input("Serum calcium level", min_value=0.0, value=9.0)
        pth = st.number_input("Parathyroid hormone (PTH) level", min_value=0.0, value=40.0)

        st.subheader("Urine Analysis")
        urine_output = st.number_input("Urine output (ml/day)", min_value=0.0, value=1500.0)
        albumin = st.number_input("Serum albumin level", min_value=0.0, value=4.0)
        protein_ratio = st.number_input("Urine protein-to-creatinine ratio", min_value=0.0, value=0.1)

    # Submit Button
    submitted = st.form_submit_button("🔍 Analyze Risk")

# 4. Prediction Logic (Executed upon submission)
if submitted:
    # Collect input data into a dictionary (matching exact column names used in training)
    input_data = {
        'Sodium level (mEq/L)': sodium,
        'Estimated Glomerular Filtration Rate (eGFR)': egfr,
        'Blood urea (mg/dl)': urea,
        'Potassium level (mEq/L)': potassium,
        'Urine output (ml/day)': urine_output,
        'White blood cell count (cells/cumm)': wbc,
        'Serum creatinine (mg/dl)': creatinine,
        'C-reactive protein (CRP) level': crp,
        'Interleukin-6 (IL-6) level': il6,
        'Parathyroid hormone (PTH) level': pth,
        'Serum albumin level': albumin,
        'Urine protein-to-creatinine ratio': protein_ratio,
        'Serum calcium level': calcium,
        'Random blood glucose level (mg/dl)': glucose,
        'Cystatin C level': cystatin
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([input_data])

    # Preprocess and align data with model features
    clean_df = preprocess_data(user_df)
    final_df = clean_df[feature_names]

    # Make Prediction
    prediction = model.predict(final_df)
    probability = model.predict_proba(final_df)[0][1] # Get confidence score

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk Detected! (Confidence: {probability:.1%})")
        st.warning("Please consult a Nephrologist immediately.")
    else:
        st.success(f"✅ Low Risk - Results look normal. (Confidence: {(1-probability):.1%})")