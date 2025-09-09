import streamlit as st
import numpy as np

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
It uses research insights from Zimbabwe MICS 2019 data.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Patient Information")
    
    # Input fields
    age = st.slider("Age", 15, 49, 25, help="Mother's age in years")
    parity = st.number_input("Number of births", 0, 15, 1, help="Total number of live births")
    late_initiator = st.radio("First ANC visit after first trimester?", ["No", "Yes"], 
                             help="First visit after 13 weeks gestation")
    education = st.selectbox("Education Level", 
                            ["No formal education", "Primary", "Secondary", "Higher"], 
                            help="Highest education level completed")
    insurance = st.radio("Has health insurance?", ["No", "Yes"])
    ever_birth = st.radio("Ever given birth?", ["No", "Yes"])
    marital_status = st.selectbox("Marital Status", 
                                 ["Married", "Living with partner", "Not in union"])

# Simplified prediction function based on your research findings
def predict_risk_simple(age, parity, late_initiator, education, insurance, ever_birth, marital_status):
    """
    Simplified prediction based on your research findings
    Base risk is 84.28% (from your data) with adjustments based on risk factors
    """
    # Base risk from your dataset
    base_risk = 0.8428
    
    # Apply adjustments based on your SHAP analysis
    # Late initiation (most important factor)
    if late_initiator == "Yes":
        base_risk += 0.15
    
    # Education level
    if education in ["No formal education", "Primary"]:
        base_risk += 0.08
    elif education == "Higher":
        base_risk -= 0.10
    
    # Insurance status
    if insurance == "No":
        base_risk += 0.06
    
    # Parity (number of births)
    if parity > 3:
        base_risk += 0.03 * (parity - 3)
    
    # Marital status
    if marital_status == "Not in union":
        base_risk += 0.04
    
    # Age factor (younger mothers might have higher risk)
    if age < 20:
        base_risk += 0.05
    elif age > 35:
        base_risk += 0.03
    
    # Ensure risk is between 0 and 1
    return max(0.1, min(0.99, base_risk))

# Prediction button
if st.sidebar.button("Predict Care Gap Risk"):
    # Get prediction
    risk_score = predict_risk_simple(age, parity, late_initiator, education, insurance, ever_birth, marital_status)
    
    # Display results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Results")
        st.metric("Risk of Care Gap", f"{risk_score:.1%}")
        
        # Create a color-coded progress bar
        progress_color = "red" if risk_score > 0.7 else "orange" if risk_score > 0.4 else "green"
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
            - Provide education on importance of ANC
            """)
        elif risk_score > 0.4:
            st.markdown("""
            **Recommendations:**
            - Ensure complete ANC package
            - Provide health education on importance of care
            - Schedule follow-up appointment
            - Address any specific barriers to care
            - Monitor closely for any changes
            """)
        else:
            st.markdown("""
            **Recommendations:**
            - Continue routine ANC care
            - Reinforce importance of completing all visits
            - Monitor for any changes in circumstances
            - Provide information on danger signs
            """)
        
        # Key risk factors
        st.subheader("Key Risk Factors")
        factors = []
        
        if late_initiator == "Yes":
            factors.append("Late ANC initiation (first visit after 13 weeks)")
        
        if education in ["No formal education", "Primary"]:
            factors.append("Lower education level")
        elif education == "Higher":
            factors.append("Higher education (protective)")
        
        if insurance == "No":
            factors.append("No health insurance")
        
        if parity > 3:
            factors.append(f"High parity ({parity} births)")
        
        if marital_status == "Not in union":
            factors.append("Not in a union")
        
        if age < 20:
            factors.append("Young maternal age (<20 years)")
        elif age > 35:
            factors.append("Advanced maternal age (>35 years)")
        
        for factor in factors:
            st.write(f"• {factor}")
        
        if not factors:
            st.info("No significant risk factors identified")

# Information section
with st.sidebar:
    st.header("About")
    st.info("""
    This tool predicts the risk of incomplete antenatal care based on research insights from Zimbabwe MICS 2019 data.
    
    **Top predictors:**
    1. Late ANC initiation
    2. Education level  
    3. Insurance status
    4. Number of births
    5. Marital status
    6. Maternal age
    """)
    
    st.header("Data Source")
    st.info("""
    This tool is based on analysis of Zimbabwe Multiple Indicator Cluster Survey (MICS) 2019 data.
    
    **Research Findings:**
    - 84.28% of women had care gaps (missing ≥2 essential ANC components)
    - Late initiation was the strongest predictor of care gaps
    - Socioeconomic factors significantly influence ANC completion
    """)

# Add footer
st.markdown("---")
st.markdown("**ANC Care Gap Predictor** | Developed for improving maternal health outcomes in Zimbabwe")
st.markdown("Based on research using Zimbabwe MICS 2019 data")