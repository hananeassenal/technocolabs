import streamlit as st
import pickle
import os

# Function to load a pickled model
@st.cache
def load_model(file_path):
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

# Adjust the number of input fields based on the model selected
if option == "Logistic Regression":
    num_features = 10  # Assuming the Logistic Regression model expects 10 features
elif option == "Decision Tree":
    num_features = 3  # Adjust this number based on your Decision Tree model

# Create input fields dynamically based on the number of features
input_data = []
for i in range(num_features):
    input_value = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(input_value)

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
