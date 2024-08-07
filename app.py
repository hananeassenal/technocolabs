import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load pre-trained models
logistic_pipe = joblib.load('logistic_model.pkl')
naive_bayes_pipe = joblib.load('naive_bayes_model.pkl')
decision_tree_pipe = joblib.load('decision_tree_model.pkl')

# Define all feature columns used during model training
all_feature_cols = [
    'total_payment', 'CreditScore', 'OrigInterestRate', 'MonthlyIncome', 'MIP',
    'OCLTV', 'MonthlyRate', 'MSA', 'OrigLoanTerm', 'interest_amt', 'EMI', 'cur_principal',
    'MonthsDelinquent', 'DTI', 'OrigUPB', 'MonthsInRepayment'
]

# Sample values for testing
sample_values = {
    'total_payment': 500.0,
    'CreditScore': 700.0,
    'OrigInterestRate': 3.5,
    'MonthlyIncome': 5000.0,
    'MIP': 0.5,
    'OCLTV': 80.0,
    'MonthlyRate': 0.03,
    'MSA': 100.0,
    'OrigLoanTerm': 30.0,
    'interest_amt': 1000.0,
    'EMI': 1500.0,
    'cur_principal': 200000.0,
    'MonthsDelinquent': 3.0,
    'DTI': 0.35,
    'OrigUPB': 180000.0,
    'MonthsInRepayment': 24.0
}

# Define Streamlit UI
st.set_page_config(page_title='Model Prediction Web Application', layout='wide')
st.title('Model Prediction Web Application')
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
    }
    .css-18e3th9 {
        padding: 2rem;
    }
    .css-1r6slb0 {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input
st.sidebar.header('Input Features')
st.sidebar.write("Please enter the values for the features below:")

input_data = {}
for col in all_feature_cols:
    input_data[col] = st.sidebar.number_input(
        f'{col}', 
        value=float(sample_values[col]),  # Ensure value is float
        format="%.2f", 
        step=0.01, 
        min_value=-1e10,  # Set practical minimum value
        max_value=1e10    # Set practical maximum value
    )

input_df = pd.DataFrame([input_data])

# Display input data for debugging
st.write("### Input Data")
st.write(input_df)

# Make predictions
if st.sidebar.button('Predict'):
    try:
        # Make predictions with Logistic Regression
        logistic_pred = logistic_pipe.predict(input_df)
        result_logistic = 'Accepted for Credit' if logistic_pred[0] == 1 else 'Rejected for Credit'
        st.write(f"### Logistic Regression Result: **{result_logistic}**")
    except Exception as e:
        st.write(f"**Error in Logistic Regression prediction:** {e}")

    try:
        # Make predictions with Naive Bayes
        naive_bayes_pred = naive_bayes_pipe.predict(input_df)
        result_naive_bayes = 'Accepted for Credit' if naive_bayes_pred[0] == 1 else 'Rejected for Credit'
        st.write(f"### Naive Bayes Result: **{result_naive_bayes}**")
    except Exception as e:
        st.write(f"**Error in Naive Bayes prediction:** {e}")

    try:
        # Make predictions with Decision Tree
        decision_tree_pred = decision_tree_pipe.predict(input_df)
        result_decision_tree = 'Accepted for Credit' if decision_tree_pred[0] == 1 else 'Rejected for Credit'
        st.write(f"### Decision Tree Result: **{result_decision_tree}**")
    except Exception as e:
        st.write(f"**Error in Decision Tree prediction:** {e}")
