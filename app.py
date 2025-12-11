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
st.markdown("Enter the patient's lab results below to analyze the risk of Chronic Kidney Disease (CKD).")

# 2. Load Model and Feature Names
@st.cache_resource # Cache the model to optimize performance
def load_resources():
    try:
        model = joblib.load('models/kidney_model.joblib')
        with open('models/model_features.json', 'r') as f:
            feature_names = json.load(f)
        return model, feature_names
    except FileNotFoundError:
        return None, None

model, feature_names = load_resources()

if model is None:
    st.error("🚨 Error: Model files not found! Please run 'python src/train.py' first.")
    st.stop()

# 3. Input Form (Updated for New Top 15 Features)
with st.form("prediction_form"):
    st.header("📝 Patient Data Entry")
    
    # --- Section 1: Urine Analysis (The most critical indicators now) ---
    st.subheader("1. Urine Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protein_ratio = st.number_input("Urine Protein-to-Creatinine Ratio", min_value=0.0, max_value=10.0, value=0.2, step=0.1, help="Normal range is typically < 0.2")
        albumin_urine = st.selectbox("Albumin in Urine", options=[0, 1, 2, 3, 4, 5], help="0=None, 1=Trace, 2-5=Levels")
    
    with col2:
        urine_sediment = st.selectbox("Urinary Sediment Microscopy", options=["normal", "abnormal"])
        pus_cells = st.selectbox("Pus Cells in Urine", options=["normal", "abnormal"])
    
    with col3:
        rbc_urine = st.selectbox("Red Blood Cells (Urine)", options=["normal", "abnormal"])

    st.markdown("---")

    # --- Section 2: Kidney Function & Blood ---
    st.subheader("2. Kidney Function & Blood Tests")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        creatinine = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, max_value=20.0, value=0.9, step=0.1)
        egfr = st.number_input("eGFR", min_value=0.0, max_value=150.0, value=100.0, step=1.0)
    
    with col2:
        cystatin = st.number_input("Cystatin C level (mg/l)", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
        urea = st.number_input("Blood Urea (mg/dl)", min_value=0.0, max_value=300.0, value=30.0, step=1.0)
    
    with col3:
        pth = st.number_input("Parathyroid Hormone (PTH)", min_value=0.0, max_value=1000.0, value=40.0, step=1.0)

    st.markdown("---")

    # --- Section 3: Inflammation & History ---
    st.subheader("3. Clinical History & Inflammation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        il6 = st.number_input("Interleukin-6 (IL-6)", min_value=0.0, max_value=200.0, value=2.0, step=0.1)
        crp = st.number_input("CRP Level", min_value=0.0, max_value=200.0, value=1.0, step=0.1)
    
    with col2:
        bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=250, value=120)
        cad = st.selectbox("Coronary Artery Disease", options=["no", "yes"])
    
    with col3:
        appetite = st.selectbox("Appetite", options=["good", "poor"])

    # Submit Button
    submitted = st.form_submit_button("🔍 Analyze Risk")

# 4. Prediction Logic
if submitted:
    # Dictionary matching the EXACT Top 15 Feature Names
    input_data = {
        'Urine protein-to-creatinine ratio': protein_ratio,
        'Serum creatinine (mg/dl)': creatinine,
        'Estimated Glomerular Filtration Rate (eGFR)': egfr,
        'Cystatin C level': cystatin,
        'Albumin in urine': albumin_urine,
        'Interleukin-6 (IL-6) level': il6,
        'Urinary sediment microscopy results': urine_sediment,
        'Blood urea (mg/dl)': urea,
        'Red blood cells in urine': rbc_urine,
        'Parathyroid hormone (PTH) level': pth,
        'Pus cells in urine': pus_cells,
        'C-reactive protein (CRP) level': crp,
        'Coronary artery disease (yes/no)': cad,
        'Blood pressure (mm/Hg)': bp,
        'Appetite (good/poor)': appetite
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([input_data])

    try:
        # Preprocess and align data
        clean_df = preprocess_data(user_df)
        final_df = clean_df[feature_names]

        # Make Prediction
        prediction = model.predict(final_df)
        probability = model.predict_proba(final_df)[0][1] # Probability of Class 1 (Disease)

        st.divider()
        
        col_res1, col_res2 = st.columns([1, 3])
        
        with col_res1:
            if prediction[0] == 1:
                st.metric(label="Risk Level", value="High Risk ⚠️", delta="- Alert")
            else:
                st.metric(label="Risk Level", value="Low Risk ✅", delta="Normal")

        with col_res2:
            st.progress(float(probability))
            if prediction[0] == 1:
                st.error(f"**Potential Chronic Kidney Disease Detected** (Confidence: {probability:.1%})")
                st.write("The model suggests a high likelihood of kidney issues based on the provided urine analysis and clinical markers.")
            else:
                st.success(f"**Results appear within normal range** (Confidence: {(1-probability):.1%})")
                st.write("No significant signs of CKD detected based on the provided inputs.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")