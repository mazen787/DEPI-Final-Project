import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration for a wide layout
st.set_page_config(layout="wide")

st.title("🩺 Chronic Kidney Disease (CKD) Interactive EDA Dashboard")
st.write("Upload your kidney dataset to begin analysis. The dashboard will dynamically update.")

# --- File Uploader in Sidebar ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # --- Data Loading and Caching ---
    @st.cache_data
    def load_data(file):
        """Loads, renames, and preprocesses the dataset."""
        df = pd.read_csv(file)
        
        # This renaming is CRUCIAL for clean code and reliable plotting
        # It maps your provided column names to shorter, code-friendly names
        rename_map = {
            'Age of the patient': 'Age',
            'Blood pressure (mm/Hg)': 'Blood_Pressure',
            'Specific gravity of urine': 'Specific_Gravity',
            'Albumin in urine': 'Albumin',
            'Sugar in urine': 'Sugar',
            'Red blood cells in urine': 'RBC',
            'Pus cells in urine': 'Pus_Cells',
            'Pus cell clumps in urine': 'Pus_Cell_Clumps',
            'Bacteria in urine': 'Bacteria',
            'Random blood glucose level (mg/dl)': 'Blood_Glucose',
            'Blood urea (mg/dl)': 'Blood_Urea',
            'Serum creatinine (mg/dl)': 'Serum_Creatinine',
            'Sodium level (mEq/L)': 'Sodium',
            'Potassium level (mEq/L)': 'Potassium',
            'Hemoglobin level (gms)': 'Hemoglobin',
            'Packed cell volume (%)': 'Packed_Cell_Volume',
            'White blood cell count (cells/cumm)': 'WBC_Count',
            'Red blood cell count (millions/cumm)': 'RBC_Count',
            'Hypertension (yes/no)': 'Hypertension',
            'Diabetes mellitus (yes/no)': 'Diabetes_Mellitus',
            'Coronary artery disease (yes/no)': 'Coronary_Artery_Disease',
            'Appetite (good/poor)': 'Appetite',
            'Pedal edema (yes/no)': 'Pedal_Edema',
            'Anemia (yes/no)': 'Anemia',
            'Estimated Glomerular Filtration Rate (eGFR)': 'eGFR',
            'Target': 'Target',
            'Target_binary': 'Target_binary'
        }
        
        # Only rename columns that actually exist in the uploaded file
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure numeric columns are treated as numbers
        # (This is a common issue with CSVs)
        numeric_cols = [
            'Age', 'Blood_Pressure', 'Specific_Gravity', 'Albumin', 'Sugar', 
            'Blood_Glucose', 'Blood_Urea', 'Serum_Creatinine', 'Sodium', 
            'Potassium', 'Hemoglobin', 'Packed_Cell_Volume', 'WBC_Count', 
            'RBC_Count', 'eGFR'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert target to string for clear labels in plots
        if 'Target_binary' in df.columns:
            df['Target_binary'] = df['Target_binary'].map({1: 'CKD', 0: 'Not CKD'})

        return df

    data = load_data(uploaded_file)
    
    # Check if essential columns exist before proceeding
    essential_cols = ['Age', 'eGFR', 'Hypertension', 'Diabetes_Mellitus', 'Target_binary']
    if not all(col in data.columns for col in essential_cols):
        st.error("The uploaded file is missing one or more essential columns. "
                 "Please check the column names (e.g., 'Age of the patient', 'eGFR', 'Target_binary').")
    else:
        # --- Sidebar Filters (This is the "Linked Interactivity") ---
        st.sidebar.header("Dashboard Filters")
        
        # Age Slider
        age_slider = st.sidebar.slider(
            'Select Age Range',
            min_value=int(data['Age'].min()),
            max_value=int(data['Age'].max()),
            value=(int(data['Age'].min()), int(data['Age'].max()))
        )
        
        # eGFR Slider
        egfr_slider = st.sidebar.slider(
            'Select eGFR Range',
            min_value=float(data['eGFR'].min()),
            max_value=float(data['eGFR'].max()),
            value=(float(data['eGFR'].min()), float(data['eGFR'].max()))
        )
        
        # Categorical Filters (Multiselect)
        htn_options = data['Hypertension'].dropna().unique()
        htn_select = st.sidebar.multiselect(
            'Hypertension Status',
            options=htn_options,
            default=htn_options
        )
        
        dm_options = data['Diabetes_Mellitus'].dropna().unique()
        dm_select = st.sidebar.multiselect(
            'Diabetes Mellitus Status',
            options=dm_options,
            default=dm_options
        )
        
        target_options = data['Target_binary'].dropna().unique()
        target_select = st.sidebar.multiselect(
            'Patient Status',
            options=target_options,
            default=target_options
        )

        # --- Filter Data Based on Sidebar Inputs ---
        # This `filtered_data` DataFrame is what all plots will use
        filtered_data = data[
            (data['Age'].between(age_slider[0], age_slider[1])) &
            (data['eGFR'].between(egfr_slider[0], egfr_slider[1])) &
            (data['Hypertension'].isin(htn_select)) &
            (data['Diabetes_Mellitus'].isin(dm_select)) &
            (data['Target_binary'].isin(target_select))
        ]

        st.info(f"Displaying **{len(filtered_data)}** of **{len(data)}** total records based on your filters.")

        # --- KPI Metrics ---
        st.header("Filtered Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate KPIs safely
        avg_egfr = filtered_data['eGFR'].mean() if not filtered_data.empty else 0
        ckd_patients = filtered_data[filtered_data['Target_binary'] == 'CKD'].shape[0] if not filtered_data.empty else 0
        
        col1.metric("Total Patients", f"{len(filtered_data)}")
        col2.metric("CKD Patients", f"{ckd_patients}")
        col3.metric("Avg. Age", f"{filtered_data['Age'].mean():.1f}" if not filtered_data.empty else "0")
        col4.metric("Avg. eGFR", f"{avg_egfr:.1f} mL/min/1.73m²")

        # --- Plotting with Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Target & Demographics", 
            "🔬 Core Kidney Indicators", 
            "📈 Advanced 3D & Correlation", 
            "📊 Risk Factor Analysis"
        ])

        with tab1:
            st.header("Target Distribution and Demographics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Status (CKD vs. Not CKD)")
                if not filtered_data.empty:
                    fig_pie = px.pie(
                        filtered_data, 
                        names='Target_binary', 
                        title='Distribution of Patient Status',
                        hole=0.3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("No data for pie chart.")
            
            with col2:
                st.subheader("Age Distribution by Patient Status")
                if not filtered_data.empty:
                    fig_hist = px.histogram(
                        filtered_data, 
                        x='Age', 
                        color='Target_binary',
                        barmode='overlay',
                        marginal='box',
                        title='Age Distribution'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("No data for histogram.")

        with tab2:
            st.header("Core Kidney Function Indicators")
            st.write("These plots show the most critical measures of kidney health.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("eGFR vs. Serum Creatinine")
                if not filtered_data.empty and 'Serum_Creatinine' in filtered_data.columns:
                    fig_scatter = px.scatter(
                        filtered_data, 
                        x='Serum_Creatinine', 
                        y='eGFR', 
                        color='Target_binary',
                        title='eGFR vs. Serum Creatinine (Hover for details)',
                        hover_data=['Age', 'Blood_Pressure']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("No data for scatter plot (requires eGFR, Serum_Creatinine).")
            
            with col2:
                st.subheader("Blood Urea Distribution")
                if not filtered_data.empty and 'Blood_Urea' in filtered_data.columns:
                    fig_box = px.box(
                        filtered_data, 
                        x='Target_binary', 
                        y='Blood_Urea', 
                        color='Target_binary',
                        title='Blood Urea by Patient Status',
                        points='all'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("No data for box plot (requires Blood_Urea).")

        with tab3:
            st.header("Advanced 3D & Correlation Analysis")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("3D View: Key Kidney Markers (Advanced Plot)")
                if not filtered_data.empty and 'Serum_Creatinine' in filtered_data.columns and 'Blood_Urea' in filtered_data.columns:
                    fig_3d = px.scatter_3d(
                        filtered_data,
                        x='eGFR',
                        y='Serum_Creatinine',
                        z='Blood_Urea',
                        color='Target_binary',
                        title='eGFR vs. Creatinine vs. Blood Urea',
                        hover_data=['Age']
                    )
                    fig_3d.update_layout(height=600)
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("No data for 3D plot (requires eGFR, Serum_Creatinine, Blood_Urea).")

            with col2:
                st.subheader("Correlation Heatmap (Numerical Features)")
                if not filtered_data.empty:
                    # Select only numeric columns for correlation
                    numeric_cols = filtered_data.select_dtypes(include=np.number).columns
                    
                    if len(numeric_cols) > 1:
                        corr = filtered_data[numeric_cols].corr()
                        
                        # Use Plotly for an interactive heatmap
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu_r', # Red-Blue reversed
                            zmin=-1, zmax=1,
                            text=corr.values,
                            texttemplate="%{text:.2f}",
                        ))
                        fig_heatmap.update_layout(
                            title="Feature Correlation Heatmap",
                            height=600,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.warning("Not enough numeric data to generate a heatmap.")
                else:
                    st.warning("No data for heatmap.")

        with tab4:
            st.header("Risk Factor Analysis (Hypertension & Diabetes)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Hypertension vs. Patient Status")
                if not filtered_data.empty:
                    fig_htn = px.density_heatmap(
                        filtered_data, 
                        x='Hypertension', 
                        y='Target_binary',
                        title='Hypertension Status vs. CKD',
                        text_auto=True
                    )
                    st.plotly_chart(fig_htn, use_container_width=True)
                else:
                    st.warning("No data for Hypertension plot.")
                    
            with col2:
                st.subheader("Diabetes vs. Patient Status")
                if not filtered_data.empty:
                    fig_dm = px.density_heatmap(
                        filtered_data, 
                        x='Diabetes_Mellitus', 
                        y='Target_binary',
                        title='Diabetes Mellitus Status vs. CKD',
                        text_auto=True
                    )
                    st.plotly_chart(fig_dm, use_container_width=True)
                else:
                    st.warning("No data for Diabetes plot.")
            
            st.subheader("Hemoglobin Level (Anemia Indicator)")
            if not filtered_data.empty and 'Hemoglobin' in filtered_data.columns:
                fig_hemo = px.box(
                    filtered_data, 
                    x='Target_binary', 
                    y='Hemoglobin', 
                    color='Target_binary',
                    title='Hemoglobin by Patient Status (Box Plot)',
                    points='all'
                )
                st.plotly_chart(fig_hemo, use_container_width=True)
            else:
                st.warning("No data for Hemoglobin plot.")

else:
    # Initial state when no file is uploaded
    st.info("Awaiting CSV file upload...")
    st.image("https://placehold.co/1200x600/f0f2f6/000000?text=Your+Dashboard+Will+Load+Here", use_column_width=True)
