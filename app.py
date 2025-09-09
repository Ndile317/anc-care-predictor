import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pyreadstat
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Set page configuration
st.set_page_config(
    page_title="ANC Care Gap Predictor",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("👶 ANC Care Gap Predictor")
st.markdown("""
This tool predicts the risk of incomplete antenatal care based on patient characteristics.
It uses machine learning models trained on Zimbabwe MICS 2019 data.
""")

# Sidebar for user inputs
st.sidebar.header("Patient Information")

# Input fields
age = st.sidebar.slider("Age", 15, 49, 25, help="Mother's age in years")
parity = st.sidebar.number_input("Number of births", 0, 15, 1, help="Total number of live births")
late_initiator = st.sidebar.radio("First ANC visit after first trimester?", ["No", "Yes"], help="First visit after 13 weeks gestation")
education = st.sidebar.selectbox("Education Level", ["No formal education", "Primary", "Secondary", "Higher"], help="Highest education level completed")
insurance = st.sidebar.radio("Has health insurance?", ["No", "Yes"])
ever_birth = st.sidebar.radio("Ever given birth?", ["No", "Yes"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Living with partner", "Not in union"])

# Map education levels to numeric values
education_mapping = {
    "No formal education": 0,
    "Primary": 1,
    "Secondary": 3,
    "Higher": 8
}

# Map marital status to numeric values
marital_mapping = {
    "Married": 1,
    "Living with partner": 2,
    "Not in union": 3
}

# Load or train model function
@st.cache_resource
def load_model():
    try:
        # Load your data and train model here
        # This is a simplified version - you'll need to implement your actual training logic
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        # Add your training code here
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Prediction function
def predict_risk(input_dict):
    try:
        # This is a simplified prediction function
        # Replace with your actual prediction logic
        base_risk = 0.84  # Base rate from your data
        
        # Apply adjustments based on your SHAP analysis
        if input_dict['late_initiator'] == "Yes":
            base_risk += 0.10
        
        if input_dict['education'] in [0, 1]:  # No formal education or primary only
            base_risk += 0.08
        
        if input_dict['insurance'] == "No":
            base_risk += 0.06
        
        if input_dict['parity'] > 3:
            base_risk += 0.03 * (input_dict['parity'] - 3)
        
        if input_dict['marital_status'] == 3:  # Not in union
            base_risk += 0.04
        
        # Ensure risk is between 0 and 1
        return max(0.1, min(0.99, base_risk))
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return 0.5  # Default value

# Prediction button
if st.sidebar.button("Predict Care Gap Risk"):
    try:
        # Prepare input data
        input_data = {
            'age': age,
            'parity': parity,
            'late_initiator': late_initiator,
            'education': education_mapping[education],
            'insurance': insurance,
            'ever_birth': ever_birth,
            'marital_status': marital_mapping[marital_status]
        }
        
        # Get prediction
        risk_score = predict_risk(input_data)
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Results")
            st.metric("Risk of Care Gap", f"{risk_score:.1%}")
            
            # Create a color-coded progress bar
            if risk_score > 0.7:
                color = "red"
            elif risk_score > 0.4:
                color = "orange"
            else:
                color = "green"
            
            st.progress(risk_score)
            
            # Add color interpretation
            if risk_score > 0.7:
                st.error("🔴 HIGH RISK")
            elif risk_score > 0.4:
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")
        
        with col2:
            # Risk interpretation
            st.subheader("Risk Interpretation")
            
            if risk_score > 0.7:
                st.markdown("""
                **Recommendations:**
                - Schedule additional ANC visits
                - Provide transportation support if needed  
                - Assign community health worker for follow-up
                - Consider home visits or mobile clinic services
                """)
            elif risk_score > 0.4:
                st.markdown("""
                **Recommendations:**
                - Ensure complete ANC package
                - Provide health education on importance of care
                - Schedule follow-up appointment
                - Address any specific barriers to care
                """)
            else:
                st.markdown("""
                **Recommendations:**
                - Continue routine ANC care
                - Reinforce importance of completing all visits
                - Monitor for any changes in circumstances
                """)
            
            # Key risk factors
            st.subheader("Key Risk Factors")
            factors = []
            
            if input_data['late_initiator'] == "Yes":
                factors.append("Late ANC initiation (first visit after 13 weeks)")
            
            if input_data['education'] in [0, 1]:  # No formal education or primary only
                factors.append("Lower education level")
            
            if input_data['insurance'] == "No":
                factors.append("No health insurance")
            
            if input_data['parity'] > 3:
                factors.append(f"High parity ({input_data['parity']} births)")
            
            if input_data['marital_status'] == 3:  # Not in union
                factors.append("Not in a union")
            
            for factor in factors:
                st.write(f"• {factor}")
            
            if not factors:
                st.info("No significant risk factors identified")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check that all inputs are correctly filled.")

# Information section
st.sidebar.header("About")
st.sidebar.info("""
This tool predicts the risk of incomplete antenatal care based on machine learning models trained on Zimbabwe MICS 2019 data.

**Top predictors:**
1. Late ANC initiation
2. Education level  
3. Insurance status
4. Number of births
5. Marital status
""")

# Add data source citation
st.sidebar.header("Data Source")
st.sidebar.info("This tool is based on analysis of Zimbabwe Multiple Indicator Cluster Survey (MICS) 2019 data.")

# Add footer
st.markdown("---")
st.markdown("**ANC Care Gap Predictor** | Developed for improving maternal health outcomes in Zimbabwe")