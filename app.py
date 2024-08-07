import streamlit as st
import pickle
import os

# Function to load a pickled model
@st.cache
def load_model(file_path):
    if not os.path.isfile(file_path):
        st.error(f"Error loading model: {file_path} not found")
        return None
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the models
logistic_model = load_model('Logistic_Regression_model.pkl')
decision_tree_model = load_model('Decision_Tree_model.pkl')
random_forest_model = load_model('Random_Forest_model.pkl')

# Streamlit UI
st.title("Model Deployment with Streamlit")

st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

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
    elif option == "Random Forest":
        model = random_forest_model

    if model:
        prediction = model.predict([input_data])
        st.write(f"Prediction: {prediction[0]}")
    else:
        st.write("Model could not be loaded.")
