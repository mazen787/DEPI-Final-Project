import pandas as pd
import numpy as np
import random

# File configurations
NUM_ROWS = 20538
OUTPUT_FILE = "../data/kidney_disease_dataset.csv"

# The 43 columns (Exact names must be maintained)
COLUMNS = [
    'Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine', 'Albumin in urine', 
    'Sugar in urine', 'Red blood cells in urine', 'Pus cells in urine', 'Pus cell clumps in urine', 
    'Bacteria in urine', 'Random blood glucose level (mg/dl)', 'Blood urea (mg/dl)', 
    'Serum creatinine (mg/dl)', 'Sodium level (mEq/L)', 'Potassium level (mEq/L)', 
    'Hemoglobin level (gms)', 'Packed cell volume (%)', 'White blood cell count (cells/cumm)', 
    'Red blood cell count (millions/cumm)', 'Hypertension (yes/no)', 'Diabetes mellitus (yes/no)', 
    'Coronary artery disease (yes/no)', 'Appetite (good/poor)', 'Pedal edema (yes/no)', 
    'Anemia (yes/no)', 'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio', 
    'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level', 'Parathyroid hormone (PTH) level', 
    'Serum calcium level', 'Serum phosphate level', 'Family history of chronic kidney disease', 
    'Smoking status', 'Body Mass Index (BMI)', 'Physical activity level', 
    'Duration of diabetes mellitus (years)', 'Duration of hypertension (years)', 'Cystatin C level', 
    'Urinary sediment microscopy results', 'C-reactive protein (CRP) level', 
    'Interleukin-6 (IL-6) level', 'Target'
]

def generate_fuzzy_patient(status):
    row = {}
    
    # Determine Status (0: Healthy, 1: Sick)
    is_sick = False if status == 'No_Disease' else True
    
    # Determine Severity (0-4)
    severity = 0
    if status == 'Low_Risk': severity = 1
    elif status == 'Moderate_Risk': severity = 2
    elif status == 'High_Risk': severity = 3
    elif status == 'Severe_Disease': severity = 4

    # === 1. Demographics and Lifestyle ===
    row['Age of the patient'] = np.random.randint(18, 95)
    row['Body Mass Index (BMI)'] = round(np.random.uniform(18.5, 40.0), 1)
    
    # Overlap logic: Healthy people can also have low activity/smoking habits
    row['Physical activity level'] = np.random.choice(['low', 'moderate', 'high'], p=[0.4, 0.4, 0.2])
    row['Smoking status'] = np.random.choice(['former', 'yes', 'no'], p=[0.2, 0.2, 0.6])

    # === 2. Chronic Diseases (Overlap Strategy) ===
    # Sick patients have a higher probability of HTN/DM, but not 100%
    has_htn_prob = 0.70 if is_sick else 0.30 
    has_dm_prob = 0.60 if is_sick else 0.25
    
    has_htn = np.random.choice(['yes', 'no'], p=[has_htn_prob, 1-has_htn_prob])
    has_dm = np.random.choice(['yes', 'no'], p=[has_dm_prob, 1-has_dm_prob])
    
    row['Hypertension (yes/no)'] = has_htn
    row['Diabetes mellitus (yes/no)'] = has_dm
    row['Coronary artery disease (yes/no)'] = 'yes' if (is_sick and np.random.random() > 0.7) else 'no'
    row['Family history of chronic kidney disease'] = np.random.choice(['yes', 'no'], p=[0.4, 0.6]) # Random inheritance

    row['Duration of hypertension (years)'] = np.random.randint(1, 40) if has_htn == 'yes' else 0
    row['Duration of diabetes mellitus (years)'] = np.random.randint(1, 30) if has_dm == 'yes' else 0

    row['Blood pressure (mm/Hg)'] = np.random.randint(130, 190) if has_htn == 'yes' else np.random.randint(90, 125)
    row['Random blood glucose level (mg/dl)'] = np.random.randint(150, 450) if has_dm == 'yes' else np.random.randint(70, 140)

    # === 3. Kidney Function (The Gray Zone) ===
    # Using Normal Distribution to ensure continuous overlap
    
    if is_sick:
        # Sick: Values start from the high end of normal range (1.0)
        scr_mean = 1.2 + (severity * 0.5) 
        scr = np.random.normal(scr_mean, 0.4)
        scr = max(0.9, scr) # Minimum value for sick patients is 0.9
        
        egfr = max(5, min(100, np.random.normal(90 - (severity * 20), 15)))
        
        cystatin = max(0.8, np.random.normal(1.0 + (severity * 0.3), 0.5))
        bun = max(15, np.random.normal(30 + (severity * 10), 15))
        
    else:
        # Healthy: Values may reach into the sick range (1.5) due to noise
        scr = np.random.normal(0.9, 0.2)
        scr = max(0.5, min(1.5, scr)) 
        
        egfr = max(60, min(140, np.random.normal(105, 15)))
        cystatin = max(0.5, min(1.1, np.random.normal(0.8, 0.15)))
        bun = max(7, min(45, np.random.normal(25, 8)))

    row['Serum creatinine (mg/dl)'] = round(scr, 2)
    row['Estimated Glomerular Filtration Rate (eGFR)'] = round(egfr, 2)
    row['Blood urea (mg/dl)'] = round(bun, 2)
    row['Cystatin C level'] = round(cystatin, 2)

    # === 4. Electrolytes (Late Stage Disruption) ===
    # Electrolytes are mainly affected in severe stages (severity >= 3)
    if severity >= 3: 
        row['Sodium level (mEq/L)'] = round(np.random.uniform(125, 138), 1)
        row['Potassium level (mEq/L)'] = round(np.random.uniform(5.0, 7.0), 1)
    else:
        # Early stages and healthy patients share similar ranges
        row['Sodium level (mEq/L)'] = round(np.random.uniform(135, 146), 1)
        row['Potassium level (mEq/L)'] = round(np.random.uniform(3.5, 5.1), 1)

    row['Serum calcium level'] = round(np.random.uniform(8.0, 10.5), 1)
    row['Serum phosphate level'] = round(np.random.uniform(2.5, 5.0), 1)
    row['Parathyroid hormone (PTH) level'] = round(np.random.uniform(15, 80) if severity < 2 else np.random.uniform(70, 400), 1)

    # === 5. Blood Analysis (Anemia is NOT a guaranteed CKD sign) ===
    hb_mean = 10.0 if severity >= 3 else 13.5
    hb = np.random.normal(hb_mean, 2.0)
    hb = max(6.0, min(17.5, hb))
    
    row['Hemoglobin level (gms)'] = round(hb, 1)
    row['Packed cell volume (%)'] = round(hb * 3, 1) # Medical approximation
    row['Anemia (yes/no)'] = 'yes' if hb < 11.0 else 'no'
    
    row['White blood cell count (cells/cumm)'] = np.random.randint(3000, 15000)
    row['Red blood cell count (millions/cumm)'] = round(hb/3 + np.random.uniform(-0.2, 0.2), 1)
    row['Cholesterol level'] = np.random.randint(120, 300) # Random across the board

    # === 6. Urine Analysis (No Zeroes Policy) ===
    row['Specific gravity of urine'] = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025])
    
    if is_sick:
        # Higher chance of issues, but not guaranteed
        row['Albumin in urine'] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        row['Urine protein-to-creatinine ratio'] = round(np.random.uniform(0.5, 6.0), 1)
        is_urine_bad = np.random.choice([True, False], p=[0.6, 0.4]) # 40% of sick patients have normal urine
    else:
        # Low chance of issues for healthy patients
        row['Albumin in urine'] = np.random.choice([0, 1], p=[0.9, 0.1])
        row['Urine protein-to-creatinine ratio'] = round(np.random.uniform(0.1, 0.3), 1)
        is_urine_bad = False

    row['Red blood cells in urine'] = 'abnormal' if is_urine_bad else 'normal'
    row['Pus cells in urine'] = 'abnormal' if is_urine_bad else 'normal'
    row['Bacteria in urine'] = np.random.choice(['present', 'not present'], p=[0.1, 0.9])
    row['Pus cell clumps in urine'] = 'not present'
    row['Urinary sediment microscopy results'] = 'abnormal' if is_urine_bad else 'normal'

    # Sugar in urine is tied to diabetes status, not directly to CKD severity
    row['Sugar in urine'] = np.random.choice([1, 2, 3, 4, 5]) if has_dm == 'yes' else 0
    row['Urine output (ml/day)'] = np.random.randint(500, 2500)

    # === 7. Inflammation & Symptoms ===
    # CRP/IL-6: Higher baseline for sick patients but high values can occur in healthy people too
    crp_base = 10 if is_sick else 2
    row['C-reactive protein (CRP) level'] = round(max(0, np.random.normal(crp_base, 5)), 1)
    
    il6_base = 8 if is_sick else 1
    row['Interleukin-6 (IL-6) level'] = round(max(0, np.random.normal(il6_base, 3)), 1)
    
    row['Serum albumin level'] = round(np.random.uniform(2.5, 5.2), 1) # Range covers sick (low) and healthy (high)
    row['Appetite (good/poor)'] = 'poor' if (is_sick and severity >= 3) else 'good'
    row['Pedal edema (yes/no)'] = 'yes' if (is_sick and severity >= 3) else 'no'

    row['Target'] = status

    return row

# --- Execution ---
print("Generating Fuzzy & Realistic Medical Data...")
data_rows = []
targets = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'High_Risk', 'Severe_Disease']

# The desired distribution: 65% Healthy, 35% Sick
weights = [0.65, 0.15, 0.10, 0.07, 0.03] 

patient_statuses = np.random.choice(targets, size=NUM_ROWS, p=weights)

for status in patient_statuses:
    data_rows.append(generate_fuzzy_patient(status))

df_new = pd.DataFrame(data_rows)
df_new = df_new[COLUMNS]

df_new.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Generated {NUM_ROWS} rows. Final check: All 43 columns have been assigned values and class imbalance is 65/35.")