# 🏦 Loan Approval Predictor

A machine learning project that predicts whether a loan application will be approved, using a Random Forest classifier. The model is deployed using a Streamlit web app for interactive input and live predictions.

---

## 📌 Features

- Streamlit-based user interface
- Random Forest model with 85%+ accuracy
- Confidence score for each prediction
- SHAP-based explanation toggle
- Ready-to-run with a `.bat` launcher

---

## 🚀 How to Run Locally

1. Clone the repository:
  git clone https://github.com/rahul1842/Loan-Predictor.git

2. Install required packages:
  pip install -r requirements.txt

  If no `requirements.txt`: pip install streamlit pandas matplotlib shap scikit-learn

3. Run the app:
  streamlit run loan_app.py



Or just double-click `run_app.bat`.

---

## 📁 Files in This Project

| File | Description |
|------|-------------|
| `loan_approval.ipynb` | Notebook for data preprocessing, training, evaluation |
| `loan_model.pkl` | Saved model used in the app |
| `loan_app.py` | Streamlit app script |
| `train.csv` | Dataset used for training |
| `img.jpeg` | Image/logo shown in UI |
| `run_app.bat` | One-click launcher |
| `README.md` | This file |

---

## 📊 Model Performance

- **Model:** Random Forest Classifier
- **Accuracy:** 85.37%
- **Precision (Approved):** 0.83
- **Recall (Approved):** 0.99
- **F1-score (Approved):** 0.90

---

## THANK YOU!!!!!!!!!!
