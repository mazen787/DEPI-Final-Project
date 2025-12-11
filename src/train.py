import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
# Import custom preprocessing function and feature list
from preprocessing import preprocess_data, TOP_15_FEATURES

def train_model():
    # 1. Define Dynamic Paths (Ensures the script runs from any directory)
    # Get the absolute path of the directory containing train.py (which is 'src')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level (..) to the project root, then navigate to 'data' or 'models'
    data_path = os.path.join(current_dir, '..', 'data', 'kidney_disease_dataset.csv')
    model_path = os.path.join(current_dir, '..', 'models', 'kidney_model.joblib')
    features_path = os.path.join(current_dir, '..', 'models', 'model_features.json')

    print(f"Loading data from: {data_path}")
    
    # Check if the dataset file exists before proceeding
    if not os.path.exists(data_path):
        print("❌ Error: Dataset not found! Please check the path.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Prepare Target Variable (Convert multi-class risk levels to binary: 0 or 1)
    target_map = {
        'No_Disease': 0, 'Low_Risk': 1, 'Moderate_Risk': 1, 
        'High_Risk': 1, 'Severe_Disease': 1
    }
    y = df['Target'].map(target_map)
    X = df.drop('Target', axis=1)

    # 3. Apply Data Cleaning Pipeline
    print("Cleaning data...")
    # Calls the central preprocessing script to handle cleaning, scaling, and encoding
    X_clean = preprocess_data(X)
    
    # 4. Select Top 15 Features (Feature Selection)
    X_final = X_clean[TOP_15_FEATURES]

    # 5. Split Data into Training and Testing Sets (20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    # stratify=y ensures the ratio of sick/healthy patients is maintained in both sets

    # 6. Handle Class Imbalance using SMOTE (Oversampling the minority class)
    print("Applying SMOTE...")
    # 
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 7. Train Random Forest Model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_smote, y_train_smote)

    # 8. Investigation Report (To check for Data Leakage or perfect separation)
    print("\n🔍 Investigation Report:")
    importances = model.feature_importances_
    # Calculate feature importance and display the most influential features
    feature_importance_df = pd.DataFrame({
        'Feature': X_final.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print("🏆 Top 5 Features driving the decision:")
    print(feature_importance_df.head(5))

    # 9. Serialize and Save Model & Artifacts
    print(f"\nSaving model to: {model_path}")
    # Save the trained model using joblib
    joblib.dump(model, model_path)
    
    # Save the list of feature names (artifacts) used by the model
    with open(features_path, 'w') as f:
        json.dump(TOP_15_FEATURES, f)

    print("✅ Done! Model saved successfully.")

if __name__ == "__main__":
    train_model()