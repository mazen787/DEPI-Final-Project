import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page setup (layout and title)
st.set_page_config(layout="wide", page_title="Simple Kidney Analysis")

st.title("🏥 Kidney Disease - Simplified Dashboard")
st.markdown("Easy-to-understand visual analysis of patient health and kidney risks.")

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        
        # Column Renaming Map (Covers the 43 columns)
        rename_map = {
            'Age of the patient': 'Age',
            'Blood pressure (mm/Hg)': 'Blood_Pressure',
            'Specific gravity of urine': 'Specific_Gravity',
            'Albumin in urine': 'Albumin_Urine',
            'Sugar in urine': 'Sugar_Urine',
            'Red blood cells in urine': 'RBC_Urine',
            'Pus cells in urine': 'Pus_Cells',
            'Pus cell clumps in urine': 'Pus_Clumps',
            'Bacteria in urine': 'Bacteria',
            'Random blood glucose level (mg/dl)': 'Glucose',
            'Blood urea (mg/dl)': 'Urea',
            'Serum creatinine (mg/dl)': 'Creatinine',
            'Sodium level (mEq/L)': 'Sodium',
            'Potassium level (mEq/L)': 'Potassium',
            'Hemoglobin level (gms)': 'Hemoglobin',
            'Packed cell volume (%)': 'PCV',
            'White blood cell count (cells/cumm)': 'WBC',
            'Red blood cell count (millions/cumm)': 'RBC_Count',
            'Hypertension (yes/no)': 'Hypertension',
            'Diabetes mellitus (yes/no)': 'Diabetes',
            'Coronary artery disease (yes/no)': 'CAD',
            'Appetite (good/poor)': 'Appetite',
            'Pedal edema (yes/no)': 'Edema',
            'Anemia (yes/no)': 'Anemia',
            'Estimated Glomerular Filtration Rate (eGFR)': 'eGFR',
            'Urine protein-to-creatinine ratio': 'Protein_Creatinine_Ratio',
            'Urine output (ml/day)': 'Urine_Output',
            'Serum albumin level': 'Serum_Albumin',
            'Cholesterol level': 'Cholesterol',
            'Parathyroid hormone (PTH) level': 'PTH',
            'Serum calcium level': 'Calcium',
            'Serum phosphate level': 'Phosphate',
            'Family history of chronic kidney disease': 'Family_History',
            'Smoking status': 'Smoking',
            'Body Mass Index (BMI)': 'BMI',
            'Physical activity level': 'Activity',
            'Duration of diabetes mellitus (years)': 'Diabetes_Duration',
            'Duration of hypertension (years)': 'HTN_Duration',
            'Cystatin C level': 'Cystatin_C',
            'Urinary sediment microscopy results': 'Urine_Sediment',
            'C-reactive protein (CRP) level': 'CRP',
            'Interleukin-6 (IL-6) level': 'IL6',
            'Target': 'Target'
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Convert to numeric
        numeric_cols = [
            'Age', 'Blood_Pressure', 'Glucose', 'Urea', 'Creatinine', 'Sodium', 
            'Potassium', 'Hemoglobin', 'PCV', 'WBC', 'RBC_Count', 'eGFR',
            'Protein_Creatinine_Ratio', 'Urine_Output', 'Serum_Albumin', 
            'Cholesterol', 'PTH', 'Calcium', 'Phosphate', 'BMI', 
            'Diabetes_Duration', 'HTN_Duration', 'Cystatin_C', 'CRP', 'IL6'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create Status column for visualization (Healthy vs Sick)
        if 'Target' in df.columns:
            df['Status'] = df['Target'].apply(lambda x: 'Healthy ✅' if x == 'No_Disease' else 'Sick ⚠️')

        return df

    data = load_data(uploaded_file)
    
    if 'Status' not in data.columns:
        st.error("Target column missing!")
    else:
        # --- Sidebar Filters ---
        st.sidebar.header("🔍 Filters")
        
        status_filter = st.sidebar.multiselect(
            'Filter by Status:', 
            options=data['Status'].unique(),
            default=data['Status'].unique()
        )
        
        filtered_data = data[data['Status'].isin(status_filter)]
        
        st.markdown("---")

        # --- 1. Key Metrics (Simple and Human-readable KPIs) ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(filtered_data)
        sick_count = len(filtered_data[filtered_data['Status'] == 'Sick ⚠️'])
        healthy_count = len(filtered_data[filtered_data['Status'] == 'Healthy ✅'])
        avg_age = filtered_data['Age'].mean()

        col1.metric("👥 Total Patients", f"{total_patients}")
        col2.metric("✅ Healthy People", f"{healthy_count}")
        col3.metric("⚠️ Sick Cases", f"{sick_count}")
        col4.metric("🎂 Average Age", f"{avg_age:.0f} Years")

        st.markdown("---")

        # --- TABS (Reduced to 3 Simple Tabs) ---
        tab1, tab2, tab3 = st.tabs([
            "📊 General Overview", 
            "🧪 Lab Results (Simple)", 
            "❤️ Lifestyle Risks"
        ])

        # --- Tab 1: General Overview (Demographics) ---
        with tab1:
            st.subheader("Who are the patients?")
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart: Sick vs Healthy ratio
                fig_pie = px.pie(
                    filtered_data, 
                    names='Status', 
                    title='Percentage of Sick vs Healthy',
                    color='Status',
                    color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Histogram: Age Distribution
                fig_age = px.histogram(
                    filtered_data, 
                    x='Age', 
                    color='Status', 
                    title='Age Distribution (How old are they?)',
                    color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                    barmode='overlay'
                )
                st.plotly_chart(fig_age, use_container_width=True)

        # --- Tab 2: Lab Results (Bar Charts for easy comparison) ---
        with tab2:
            st.subheader("Key Medical Differences")
            st.write("Comparing average test results between Healthy and Sick patients.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Avg Protein in Urine (Top Indicator)
                avg_protein = filtered_data.groupby('Status')['Protein_Creatinine_Ratio'].mean().reset_index()
                fig_prot = px.bar(
                    avg_protein, 
                    x='Status', 
                    y='Protein_Creatinine_Ratio', 
                    title='Avg. Urine Protein Level (Higher is Worse)',
                    color='Status',
                    color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                    text_auto='.2f'
                )
                st.plotly_chart(fig_prot, use_container_width=True)
            
            with col2:
                # Avg Inflammation (IL-6)
                avg_il6 = filtered_data.groupby('Status')['IL6'].mean().reset_index()
                fig_il6 = px.bar(
                    avg_il6, 
                    x='Status', 
                    y='IL6', 
                    title='Avg. Inflammation Level (IL-6)',
                    color='Status',
                    color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                    text_auto='.2f'
                )
                st.plotly_chart(fig_il6, use_container_width=True)

        # --- Tab 3: Lifestyle and Co-morbidities ---
        with tab3:
            st.subheader("Lifestyle & Chronic Diseases")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Impact of Smoking
                if 'Smoking' in filtered_data.columns:
                    df_smoke = filtered_data.groupby(['Smoking', 'Status']).size().reset_index(name='Count')
                    fig_smoke = px.bar(
                        df_smoke, 
                        x='Smoking', 
                        y='Count', 
                        color='Status', 
                        title='Smoking Impact on Health',
                        color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                        barmode='group'
                    )
                    st.plotly_chart(fig_smoke, use_container_width=True)
            
            with col2:
                # Impact of Hypertension
                if 'Hypertension' in filtered_data.columns:
                    df_htn = filtered_data.groupby(['Hypertension', 'Status']).size().reset_index(name='Count')
                    fig_htn = px.bar(
                        df_htn, 
                        x='Hypertension', 
                        y='Count', 
                        color='Status', 
                        title='Hypertension (High Blood Pressure) Impact',
                        color_discrete_map={'Healthy ✅': 'lightgreen', 'Sick ⚠️': 'salmon'},
                        barmode='group'
                    )
                    st.plotly_chart(fig_htn, use_container_width=True)

else:
    st.info("👋 Upload your CSV file to see the simplified charts.")