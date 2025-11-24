import pandas as pd
import numpy as np

# 1. Define the top 15 features selected for the model
# These are used during both training and inference (in the Streamlit app)
TOP_15_FEATURES = [
    'Sodium level (mEq/L)', 'Estimated Glomerular Filtration Rate (eGFR)',
    'Blood urea (mg/dl)', 'Potassium level (mEq/L)', 'Urine output (ml/day)',
    'White blood cell count (cells/cumm)', 'Serum creatinine (mg/dl)',
    'C-reactive protein (CRP) level', 'Interleukin-6 (IL-6) level',
    'Parathyroid hormone (PTH) level', 'Serum albumin level',
    'Urine protein-to-creatinine ratio', 'Serum calcium level',
    'Random blood glucose level (mg/dl)', 'Cystatin C level'
]

def preprocess_data(df):
    """
    Cleaning pipeline to process raw data and convert text inputs to numbers.
    Designed to handle both Batch Data (Training) and Single Inputs (Streamlit).
    """
    # Create a copy to avoid modifying the original dataframe in place
    df = df.copy()

    # Define mapping dictionaries for categorical encoding
    binary_map = {
        'yes': 1, 'no': 0,
        'good': 1, 'poor': 0,
        'normal': 1, 'abnormal': 0,
        'present': 1, 'not present': 0,
        'former': 1, 'yes': 2, # Specific logic for smoking history
    }
    
    activity_map = {
        'low': 0, 'moderate': 1, 'high': 2
    }

    # Apply transformations
    # Using .replace() ensures safety if specific columns are missing in the input
    df = df.replace(binary_map)
    
    if 'Physical activity level' in df.columns:
        df['Physical activity level'] = df['Physical activity level'].replace(activity_map)

    # Ensure all columns are numeric; force non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values: Fill NaNs with 0
    # (Note: In a more advanced version, imputation could be used)
    df = df.fillna(0)

    return df