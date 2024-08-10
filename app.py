import streamlit as st
import pickle
import os

# Function to load a pickled model
@st.cache_resource
def load_model(file_path):
    # Print the current directory for debugging
    st.write(f"Attempting to load model from: {file_path}")
    
    if not os.path.isfile(file_path):
        st.error(f"Error loading model: {file_path} not found. Current directory: {os.getcwd()}")
        return None
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the models
logistic_model = load_model('Logistic_Regression_model.pkl')
decision_tree_model = load_model('Decision_Tree_model.pkl')

# Streamlit UI
st.title("Model Deployment with Streamlit")

st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree"])

st.subheader("Input Features")
# Example input fields; adjust as needed for your model
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Collecting input data
input_data = [feature1, feature2, feature3]

if st.button("Predict"):
    if option == "Logistic Regression":
        model = logistic_model
    elif option == "Decision Tree":
        model = decision_tree_model

    if model:
        try:
            prediction = model.predict([input_data])
            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.write("Model could not be loaded.")
