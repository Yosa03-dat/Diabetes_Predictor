import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title('🩺 Diabetes Risk Predictor')
st.markdown("<p style='color: #6c757d;'>Advanced clinical screening tool powered by Random Forest Analysis</p>", unsafe_allow_html=True)
st.markdown('---')

# Load model and scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ Error: Model files not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Layout: Form Container
with st.container():
    st.subheader('📋 Patient Vital Signs')
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input('Number of Pregnancies', 0, 20, 3)
        glucose = st.slider('Glucose Level (mg/dL)', 0, 200, 117)
        bp = st.number_input('Blood Pressure (mm Hg)', 0, 150, 72)
        skin = st.number_input('Skin Thickness (mm)', 0, 100, 23)

    with col2:
        insulin = st.number_input('Insulin Level (μU/mL)', 0, 900, 30)
        bmi = st.slider('BMI (Body Mass Index)', 0.0, 70.0, 32.0, 0.1)
        dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.37, 0.01)
        age = st.slider('Age (years)', 18, 100, 29)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Logic
if st.button('🔍 Calculate Clinical Risk Score'):
    # Feature Engineering
    base_features = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    
    engineered = {
        'Glucose_BMI_Ratio': glucose / (bmi + 1),
        'Age_BMI_Interaction': age * bmi,
        'Insulin_Glucose_Ratio': insulin / (glucose + 1),
        'Pregnancy_Age_Ratio': pregnancies / (age + 1),
        'Glucose_Squared': glucose ** 2,
        'BMI_Squared': bmi ** 2,
        'Risk_Score': (glucose / 100) * (bmi / 25) * (age / 50),
        'Age_Group': 0 if age <= 30 else (1 if age <= 50 else 2),
        'BMI_Category': 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3)),
        'Glucose_Level': 0 if glucose < 100 else (1 if glucose < 125 else 2)
    }
    
    all_features = {**base_features, **engineered}
    feature_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'Glucose_BMI_Ratio', 'Age_BMI_Interaction', 'Insulin_Glucose_Ratio',
        'Pregnancy_Age_Ratio', 'Glucose_Squared', 'BMI_Squared', 'Risk_Score',
        'Age_Group', 'BMI_Category', 'Glucose_Level'
    ]
    
    # Scale and Predict
    input_df = pd.DataFrame([all_features])[feature_order]
    scaled_data = scaler.transform(input_df)
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Results Presentation
    st.markdown("### 🎯 Assessment Results")
    
    with st.container():
        # Probability Metric
        risk_pct = probability * 100
        st.metric('Diabetes Risk Probability', f'{risk_pct:.1f}%')
        
        # Risk Categorization
        if risk_pct < 30:
            st.success('**Status: Low Risk**')
            st.info('Your diabetes risk appears low. Maintain a balanced diet and regular physical activity.')
        elif risk_pct < 60:
            st.warning('**Status: Moderate Risk**')
            st.info('You have moderate risk. Lifestyle modifications and routine screening are recommended.')
        else:
            st.error('**Status: High Risk**')
            st.info('High risk factors detected. Please consult a healthcare professional for clinical testing.')

        # Sub-metrics for key indicators
        st.markdown("<br>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Glucose", f"{glucose} mg/dL", delta="Elevated" if glucose > 125 else None, delta_color="inverse")
        m_col2.metric("BMI", f"{bmi:.1f}", delta="Obese" if bmi > 30 else None, delta_color="inverse")
        m_col3.metric("Risk Score", f"{engineered['Risk_Score']:.2f}")

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption('⚕️ **Medical Disclaimer:** This tool provides a statistical estimate and is not a clinical diagnosis.')
st.caption('📈 **Model Data:** Pima Indians Diabetes Dataset | Accuracy: 78% | AUC: 0.83')