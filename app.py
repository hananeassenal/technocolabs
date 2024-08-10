import streamlit as st
import pickle
import os

# Function to load the combined model pipeline
@st.cache_resource
def load_model(file_path):
    st.write(f"Attempting to load model from: {file_path}")
    
    if not os.path.isfile(file_path):
        st.error(f"Error: {file_path} not found. Current directory: {os.getcwd()}")
        return None
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the combined model pipeline
combined_pipeline = load_model('combined_pipeline.pkl')

# Streamlit UI
st.title("Model Deployment with Streamlit")

st.subheader("Input Features")
# Input fields for the features
input_features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(10)]

# Collecting input data
input_data = [input_features]

if st.button("Predict"):
    if combined_pipeline:
        try:
            prediction = combined_pipeline.predict(input_data)
            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.write("Model could not be loaded.")
