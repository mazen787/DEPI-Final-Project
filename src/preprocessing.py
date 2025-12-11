import pandas as pd
import numpy as np

# src/preprocessing.py

TOP_15_FEATURES = [
    'Urine protein-to-creatinine ratio',
    'Serum creatinine (mg/dl)',
    'Estimated Glomerular Filtration Rate (eGFR)',
    'Cystatin C level',
    'Albumin in urine',
    'Interleukin-6 (IL-6) level',
    'Urinary sediment microscopy results',  # جديد
    'Blood urea (mg/dl)',
    'Red blood cells in urine',             # جديد
    'Parathyroid hormone (PTH) level',
    'Pus cells in urine',                   # جديد
    'C-reactive protein (CRP) level',
    'Coronary artery disease (yes/no)',     # جديد
    'Blood pressure (mm/Hg)',               # جديد
    'Appetite (good/poor)'                  # جديد
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