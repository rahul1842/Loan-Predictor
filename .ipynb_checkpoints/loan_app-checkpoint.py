# 📌 Streamlit Loan Approval Predictor with Confidence Score & SHAP Explanation

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# --- UI Styling ---
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.markdown(
    """
    <style>
    .main {background-color: #f4f6f7; padding: 20px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.image("img.jpeg", width=250)
st.title("🏦 Loan Approval Prediction App")
st.markdown("### Enter the loan applicant's information below 👇")

# --- Load model ---
model = pickle.load(open("loan_model.pkl", "rb"))

# --- SHAP Explainer (optional) ---
explainer = shap.TreeExplainer(model)

# --- Input Form ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 240, 120, 60])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

# --- Preprocess Inputs ---
def preprocess_input():
    data = {
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Education': 1 if education == 'Graduate' else 0,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
        'Dependents_1': 1 if dependents == '1' else 0,
        'Dependents_2': 1 if dependents == '2' else 0,
        'Dependents_3+': 1 if dependents == '3+' else 0
    }
    return pd.DataFrame([data])

# --- Prediction Block ---
if st.button("Check Loan Approval"):
    input_df = preprocess_input()
    result = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][result] * 100

    if result == 1:
        st.success(f"✅ Loan Approved (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {confidence:.2f}%)")

    # --- Optional: SHAP Explanation ---
    if st.checkbox("Show Explanation"):
        st.subheader("📊 Prediction Explanation (SHAP)")
        shap_values = explainer.shap_values(input_df)

        # Prevent deprecation warning
        st.set_option('deprecation.showPyplotGlobalUse', False)

        shap.initjs()
        plt.title("Feature impact on prediction")
        shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight')
