import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define available models and their metrics
Model_of_Regression = {
    'LinearRegressionModel.pkl': 0.92,
    'lasso_model.pkl': 0.88,
    'ridge_model.pkl': 0.89,
    'knn_model.pkl': 0.84,
    'dt_model.pkl': 0.78
}

Model_of_Classifier = {
    'LogisticRegressionModel.pkl': 0.87,
    'DecisionTreeClassifierModel.pkl': 0.82,
    'RandomForestClassifierModel.pkl': 0.91
}

# Define feature sets
Regression_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'math_score',
                       'reading_score', 'writing_score', 'total_score']

Classification_features = ['gender', 'parental_level_of_education', 'math_score',
                           'reading_score', 'writing_score', 'total_score', 'average_score']

# App title and description
st.title("Advanced Prediction App")
st.sidebar.header("Options and Settings")

# Sidebar - User Guide
with st.sidebar.expander("User Guide"):
    st.write("""
    - **Step 1**: Select the type of model (Regression or Classification).
    - **Step 2**: Choose a model from the dropdown menu.
    - **Step 3**: Provide inputs manually or upload a CSV file for bulk predictions.
    - **Step 4**: View predictions and model performance metrics.
    - **Step 5**: Use visualization options for better insights.
    """)

# Sidebar - Select Model Type
model_type = st.sidebar.radio("Select Model Type:", ("Regression", "Classification"))

# Dynamic model and feature selection
if model_type == "Regression":
    selected_model = st.sidebar.selectbox("Choose a Regression Model:", list(Model_of_Regression.keys()))
    features = Regression_features
    model_accuracy = Model_of_Regression[selected_model]
else:
    selected_model = st.sidebar.selectbox("Choose a Classification Model:", list(Model_of_Classifier.keys()))
    features = Classification_features
    model_accuracy = Model_of_Classifier[selected_model]

# Display Model Accuracy
st.sidebar.metric(label="Model Accuracy", value=f"{model_accuracy * 100:.2f}%", help="The accuracy of the selected model.")

# Function to dynamically create input fields with tooltips
def get_inputs(feature_list):
    inputs = {}
    tooltips = {
        'gender': "Gender of the individual (e.g., male, female).",
        'race_ethnicity': "Race/ethnicity group of the individual.",
        'parental_level_of_education': "Highest education level of the individual's parents.",
        'math_score': "Score in mathematics (numeric).",
        'reading_score': "Score in reading (numeric).",
        'writing_score': "Score in writing (numeric).",
        'total_score': "Total score (numeric, sum of math, reading, and writing).",
        'average_score': "Average score across all subjects (numeric)."
    }
    for feature in feature_list:
        if feature in ['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score']:
            val = st.number_input(f"{feature}:", min_value=0.0, step=0.01, help=tooltips.get(feature))
        else:
            val = st.text_input(f"{feature}:", help=tooltips.get(feature))
        inputs[feature] = val
    return inputs

# Load the model
try:
    with open(selected_model, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file '{selected_model}' not found.")
    st.stop()

# Data upload option
st.sidebar.subheader("Bulk Predictions")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for bulk predictions:")

# Reset button
if st.sidebar.button("Reset"):
    st.experimental_rerun()

# Get user inputs or process uploaded data
if uploaded_file:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_data)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
else:
    st.subheader("Provide Input Values")
    user_inputs = get_inputs(features)

# Predict button
if st.button("Predict"):
    try:
        if uploaded_file:
            input_array = input_data.values
            predictions = model.predict(input_array)
            st.write("Predictions:")
            st.dataframe(predictions)

            # Visualization: Class Distribution or Predictions
            st.subheader("Visualization")
            if model_type == "Classification":
                sns.countplot(x=predictions)
                plt.title("Class Distribution")
                st.pyplot(plt)
            else:
                plt.figure(figsize=(10, 5))
                plt.plot(range(len(predictions)), predictions, marker='o')
                plt.title("Predictions Visualization")
                st.pyplot(plt)
        else:
            input_array = np.array([[float(user_inputs[feature]) if feature in ['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score']
                                     else 0 for feature in features]])
            prediction = model.predict(input_array)
            if model_type == "Regression":
                st.success(f"The Prediction is: {prediction[0]:.2f}")
            else:
                st.success(f"The Predicted Class is: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Model Comparison
st.sidebar.subheader("Compare Models")
if st.sidebar.button("Compare All Models"):
    st.subheader("Model Comparison")
    comparison_data = pd.DataFrame({
        'Model': list(Model_of_Regression.keys() if model_type == "Regression" else Model_of_Classifier.keys()),
        'Accuracy (%)': [val * 100 for val in (Model_of_Regression.values() if model_type == "Regression" else Model_of_Classifier.values())]
    })
    st.dataframe(comparison_data)
    st.bar_chart(comparison_data.set_index('Model'))
