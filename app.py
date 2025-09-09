# app.py
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
import plotly.graph_objects as go

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

# Load and preprocess data function
@st.cache_resource
def load_and_train_model():
    # Load data
    file_path = "wm.sav"  # Make sure this file is in the same directory
    df, meta = pyreadstat.read_sav(file_path)
    
    # Create CareGapScore
    def create_care_gap_score(df):
        components = []
        
        if 'MN6A' in df.columns:
            df['bp_checked'] = np.where(df['MN6A'] == 1, 1, 0)
            components.append('bp_checked')
        
        if 'MN6B' in df.columns:
            df['urine_tested'] = np.where(df['MN6B'] == 1, 1, 0)
            components.append('urine_tested')
        
        if 'MN6C' in df.columns:
            df['blood_tested'] = np.where(df['MN6C'] == 1, 1, 0)
            components.append('blood_tested')
        
        if 'MN9' in df.columns:
            df['tetanus_adequate'] = np.where(df['MN9'] >= 2, 1, 0)
            components.append('tetanus_adequate')
        elif 'MN8' in df.columns:
            df['tetanus_any'] = np.where(df['MN8'] == 1, 1, 0)
            components.append('tetanus_any')
        
        if 'HA14' in df.columns:
            df['hiv_tested'] = np.where(df['HA14'] == 1, 1, 0)
            components.append('hiv_tested')
        
        df['received_components'] = df[components].sum(axis=1)
        total_components = len(components)
        df['CareGapScore'] = (df['received_components'] < (total_components - 1)).astype(int)
        
        return df, components

    df, components_used = create_care_gap_score(df)
    
    # Create LateInitiator feature
    def create_late_initiator(df):
        if 'MN4AN' in df.columns and 'MN4AU' in df.columns:
            gestational_age_weeks = np.zeros(len(df))
            
            weeks_mask = df['MN4AU'] == 1
            gestational_age_weeks[weeks_mask] = df.loc[weeks_mask, 'MN4AN']
            
            months_mask = df['MN4AU'] == 2
            gestational_age_weeks[months_mask] = df.loc[months_mask, 'MN4AN'] * 4.345
            
            other_mask = ~weeks_mask & ~months_mask
            gestational_age_weeks[other_mask] = np.nan
            
            df['LateInitiator'] = np.where(gestational_age_weeks > 13, 1, 0)
        else:
            df['LateInitiator'] = np.nan
        
        return df

    df = create_late_initiator(df)
    
    # Define feature set
    all_features = ['WB4', 'WB6A', 'WB18', 'CM1', 'CM11', 'MA1', 'LateInitiator']
    
    # Prepare the data
    X = df[all_features].copy()
    y = df['CareGapScore'].copy()
    
    # Remove rows where target is missing
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Also remove rows where LateInitiator is missing
    valid_late_initiator = ~X['LateInitiator'].isna()
    X = X[valid_late_initiator]
    y = y[valid_late_initiator]
    
    # Define numerical and categorical features
    numerical_features = ['WB4', 'CM11', 'LateInitiator']
    categorical_features = ['WB6A', 'WB18', 'CM1', 'MA1']
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Train the model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_processed, y)
    
    return model, preprocessor

# Load or train model
try:
    model = joblib.load('best_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    st.sidebar.success("Loaded pre-trained model")
except:
    st.sidebar.info("Training model... This may take a few minutes.")
    model, preprocessor = load_and_train_model()
    joblib.dump(model, 'best_model.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    st.sidebar.success("Model trained and saved")

# Input fields
age = st.sidebar.slider("Age", 15, 49, 25, help="Mother's age in years")

parity = st.sidebar.number_input("Number of births", 0, 15, 1, 
                                help="Total number of live births")

late_initiator = st.sidebar.radio("First ANC visit after first trimester?", 
                                 ["No", "Yes"], 
                                 help="First visit after 13 weeks gestation")

education = st.sidebar.selectbox("Education Level", [
    "No formal education", 
    "Primary", 
    "Secondary",
    "Higher"
], help="Highest education level completed")

insurance = st.sidebar.radio("Has health insurance?", ["No", "Yes"])

ever_birth = st.sidebar.radio("Ever given birth?", ["No", "Yes"])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married", 
    "Living with partner", 
    "Not in union"
])

# Prediction function
def predict_risk(input_dict):
    # Map inputs to model format
    input_df = pd.DataFrame([{
        'WB4': input_dict['age'],
        'CM11': input_dict['parity'],
        'LateInitiator': 1 if input_dict['late_initiator'] == "Yes" else 0,
        'WB6A': input_dict['education'],
        'WB18': 1 if input_dict['insurance'] == "Yes" else 2,
        'CM1': 1 if input_dict['ever_birth'] == "Yes" else 2,
        'MA1': input_dict['marital_status']
    }])
    
    # Preprocess input
    processed_input = preprocessor.transform(input_df)
    
    # Make prediction
    risk_score = model.predict_proba(processed_input)[0][1]
    
    return risk_score

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

# Predict button
if st.sidebar.button("Predict Care Gap Risk"):
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
        
        # Create visual gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Care Gap Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk interpretation
        st.subheader("Risk Interpretation")
        
        if risk_score > 0.7:
            st.error("🔴 HIGH RISK")
            st.markdown("""
            **Recommendations:**
            - Schedule additional ANC visits
            - Provide transportation support if needed  
            - Assign community health worker for follow-up
            - Consider home visits or mobile clinic services
            """)
        elif risk_score > 0.4:
            st.warning("🟡 MEDIUM RISK")
            st.markdown("""
            **Recommendations:**
            - Ensure complete ANC package
            - Provide health education on importance of care
            - Schedule follow-up appointment
            - Address any specific barriers to care
            """)
        else:
            st.success("🟢 LOW RISK")
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