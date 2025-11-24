import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
# Import custom preprocessing function and feature list
from preprocessing import preprocess_data, TOP_15_FEATURES

def train_model():
    print("Loading data...")
    # Load dataset (navigating up one directory to 'data' folder)
    df = pd.read_csv('../data/kidney_disease_dataset.csv')
    
    # 1. Prepare Target Variable (Binary Classification)
    target_map = {
        'No_Disease': 0, 'Low_Risk': 1, 'Moderate_Risk': 1, 
        'High_Risk': 1, 'Severe_Disease': 1
    }
    y = df['Target'].map(target_map)
    X = df.drop('Target', axis=1)

    # 2. Apply Data Cleaning Pipeline
    print("Cleaning data...")
    X_clean = preprocess_data(X)
    
    # 3. Select Top 15 Features (Feature Selection)
    X_final = X_clean[TOP_15_FEATURES]

    # 4. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Handle Class Imbalance using SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 6. Train Random Forest Model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_smote, y_train_smote)

    # 7. Serialize and Save Model & Artifacts
    print("Saving model and features...")
    joblib.dump(model, '../models/kidney_model.joblib')
    
    with open('../models/model_features.json', 'w') as f:
        json.dump(TOP_15_FEATURES, f)

    print("Done! Model saved in models/ folder.")

if __name__ == "__main__":
    train_model()