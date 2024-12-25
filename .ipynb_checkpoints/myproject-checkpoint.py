import streamlit as st
import pickle
import numpy as np


Model_of_Regression = [
    'LinearRegressionModel.pkl', 'lasso_model.pkl', 'ridge_model.pkl',
    'knn_model.pkl', 'dt_model.pkl'
]

Model_of_Classifier = [
    'LogisticRegressionModel.pkl', 'DecisionTreeClassifierModel.pkl', 'RandomForestClassifierModel.pkl'
]


st.title("Prediction App")
st.write("Choose your model and provide input values to get predictions:")

model_type = st.sidebar.radio("Select Model Type:", ("Regression", "Classification"))

if model_type == "Regression":
    selected_model = st.sidebar.selectbox("Please choose your Regression model:", Model_of_Regression)
elif model_type == "Classification":
    selected_model = st.sidebar.selectbox("Please choose your Classification model:", Model_of_Classifier)


try:
    with open(selected_model, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: {selected_model} not found.")
    st.stop()


def get_inputs():
    inputs = []
    num_inputs = st.number_input("Number of features:", min_value=1, step=1, value=6)  # Adjust as needed
    for i in range(num_inputs):
        val = st.number_input(f"Feature {i+1}:", step=0.01)
        inputs.append(val)
    return np.array([inputs])


features = get_inputs()


if st.button("Predict"):
    try:
        prediction = model.predict(features)
        if model_type == "Regression":
            st.success(f"The Prediction is: {prediction[0]:.2f}")
        elif model_type == "Classification":
            st.success(f"The Predicted Class is: {int(prediction[0])}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
