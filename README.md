# 🏥 Kidney Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Status](https://img.shields.io/badge/Status-Production-success)

## 📌 Overview
This project is a Machine Learning application designed to predict the risk of **Chronic Kidney Disease (CKD)** based on patient physiological data and lab results. 

The system uses a **Random Forest Classifier** trained on clinical records, utilizing **SMOTE** technique to handle dataset imbalance, ensuring high sensitivity and accuracy in detection.

## 🚀 Key Features
* **Interactive Web UI:** Built with [Streamlit](https://streamlit.io/) for easy data entry and instant results.
* **Robust Preprocessing:** Automated pipeline for cleaning data, handling missing values, and scaling features.
* **Balanced Training:** Implements SMOTE (Synthetic Minority Over-sampling Technique) to ensure fair predictions.
* **Dockerized:** Fully containerized for easy deployment on any machine or cloud server.

## 📂 Project Structure
```text
KIDNEY-PROJECT/
│
├── data/                   # Raw dataset (CSV)
├── models/                 # Serialized models (.joblib) and feature lists (.json)
├── notebooks/              # Jupyter notebooks for experimentation and EDA
├── src/                    # Source code for data processing and training
│   ├── __init__.py
│   ├── data_cleaning.py    # Preprocessing pipeline
│   └── train.py            # Model training script
│
├── app.py                  # Main Streamlit application entry point
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
## 🛠️ Installation & Usage

### Option 1: Running Locally (Python)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mazen787/kidney-project.git
    cd kidney-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    Access the app at `http://localhost:8501`

---

### Option 2: Running with Docker (Recommended) 🐳

1.  **Build the Docker image:**
    ```bash
    docker build -t kidney-app .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 kidney-app
    ```
    
3.  **Access the application:**
    Open your browser and go to `http://localhost:8501`

## 🧠 Model Details
* **Algorithm:** Random Forest Classifier
* **Key Features:** The model analyzes 15 critical indicators including:
    * Serum Creatinine
    * Hemoglobin
    * Specific Gravity
    * Albumin
    * (And 11 others...)

## 🤝 Contributing
1.  Fork the repository
2.  Create a Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## 📜 License
This project is licensed under the MIT License.