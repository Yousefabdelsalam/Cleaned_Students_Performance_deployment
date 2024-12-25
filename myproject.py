import streamlit as st
import pickle
import numpy as np
import pandas as pd

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

# Define classification labels (numeric -> text)
classification_labels = {
    0: "Group A",
    1: "Group B",
    2: "Group C"
}

# Define feature sets
Regression_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'math_score',
                       'reading_score', 'writing_score', 'total_score']

Classification_features = ['gender', 'parental_level_of_education', 'math_score',
                           'reading_score', 'writing_score', 'total_score', 'average_score']

# App title and description
st.title("Advanced Prediction and Model Comparison App")
st.sidebar.header("Options and Settings")

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

# Compare Models Button
if st.sidebar.button("Compare Models"):
    st.subheader("Model Comparison")

    if model_type == "Regression":
        models = Model_of_Regression
    else:
        models = Model_of_Classifier

    # Create a DataFrame for model comparison
    comparison_data = pd.DataFrame({
        'Model': list(models.keys()),
        'Accuracy (%)': [accuracy * 100 for accuracy in models.values()]
    })

    # Display the comparison table
    st.write("Model Performance Comparison:")
    st.dataframe(comparison_data)

    # Visualize the comparison
    st.bar_chart(comparison_data.set_index('Model'))

# Function to dynamically create input fields
def get_inputs(feature_list):
    inputs = {}
    for feature in feature_list:
        if feature in ['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score']:
            val = st.number_input(f"{feature}:", min_value=0.0, step=0.01)
        else:
            val = st.text_input(f"{feature}:")
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

            if model_type == "Classification":
                # Convert numeric predictions to text labels
                predictions = [classification_labels.get(int(pred), "Unknown") for pred in predictions]

            st.write("Predictions:")
            st.dataframe(predictions)

            # Visualization: Class Distribution or Predictions
            st.subheader("Histogram Visualization")
            if model_type == "Classification":
                # Count occurrences for each class
                unique, counts = np.unique(predictions, return_counts=True)
                histogram_data = pd.DataFrame({
                    "Class": unique,
                    "Count": counts
                })
                st.bar_chart(histogram_data.set_index("Class"))
            else:
                # Generate histogram for regression predictions
                hist, bins = np.histogram(predictions, bins=10)
                histogram_data = pd.DataFrame({
                    "Bins": bins[:-1],
                    "Frequency": hist
                })
                st.bar_chart(histogram_data.set_index("Bins"))
        else:
            input_array = np.array([[float(user_inputs[feature]) if feature in ['math_score', 'reading_score', 'writing_score', 'total_score', 'average_score']
                                     else 0 for feature in features]])
            prediction = model.predict(input_array)
            if model_type == "Regression":
                st.success(f"The Prediction is: {prediction[0]:.2f}")
            else:
                # Convert numeric prediction to text label
                predicted_label = classification_labels.get(int(prediction[0]), "Unknown")
                st.success(f"The Predicted Class is: {predicted_label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
