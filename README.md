# Cleaned_Students_Performance_deployment

# **Prediction Application**

🚀 A user-friendly **Streamlit-based Prediction App** for performing machine learning predictions using pre-trained models. This project supports both **regression** and **classification** tasks, providing an interactive platform to explore the power of machine learning models.

---

## **✨ Features**
- 🎨 **Interactive Interface**: Streamlit-powered web app with an intuitive interface for model selection and input.
- 📈 **Support for Regression Models**:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree Regression
- 🔍 **Support for Classification Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- ⚙️ **Customizable Inputs**: Users can specify the number of features and input their values.
- ⚡ **Real-Time Predictions**: Instant output based on the selected model.

---

## **📂 File Overview**
### **📜 Application Code**
- **`myproject.py`**: The main application file built with Streamlit. Handles model loading, user inputs, and predictions.

### **🧠 Pre-trained Models**
- **Regression Models**:
  - `LinearRegressionModel.pkl`
  - `lasso_model.pkl`
  - `ridge_model.pkl`
  - `knn_model.pkl`
  - `dt_model.pkl`
- **Classification Models**:
  - `LogisticRegressionModel.pkl`
  - `DecisionTreeClassifierModel.pkl`
  - `RandomForestClassifierModel.pkl`

### **📦 Dependency Files**
- **`requirements.txt`**: Core dependencies for running the app, such as `scikit-learn`, `numpy`, and `Streamlit`.
- **`installation.txt`**: Additional libraries for extended functionalities, including `pandas`, `plotly`, `seaborn`, and more.

---

## **🛠️ Setup Instructions**
### **1. Create a New Environment**
Create a new Python environment using Anaconda or another tool:
```bash
conda create -n prediction_app python=3.9
conda activate prediction_app
```

### **2. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
Launch the Streamlit app:
```bash
streamlit run myproject.py
```
The app will open in your default web browser. Follow the interface to select models, input features, and make predictions.

---

## **💡 Usage Instructions**
1. **Select Model Type**:
   - Choose between "Regression" or "Classification" from the sidebar.
2. **Select Model**:
   - Pick a pre-trained model from the dropdown.
3. **Input Features**:
   - Specify the number of features and enter their values.
4. **Predict**:
   - Click the "Predict" button to generate predictions. The result is displayed instantly.

---

## **📚 Dependencies**
Key libraries used in this project:
- **Core Machine Learning**:
  - `scikit-learn`: For loading pre-trained models and making predictions.
  - `numpy`: For numerical operations.
- **Visualization and UI**:
  - `Streamlit`: For building the interactive app.
  - `seaborn`, `plotly`, `matplotlib`: Optional visualization tools.
- **Others**:
  - `pandas`, `joblib`, `threadpoolctl`

---

## **🎯 Example**
### Predicting with a Regression Model
1. Select "Regression" from the sidebar.
2. Choose a model, e.g., "Ridge Regression."
3. Specify feature values like `5.2`, `3.1`, `2.8`, etc.
4. Click "Predict" to get the regression value.

---

## **🚀 Future Enhancements**
- Add support for live data inputs via APIs.
- Include model evaluation metrics like accuracy, RMSE, or confusion matrix.
- Extend support to deep learning models.

---

## **📜 License**
This project is open source and available under the [MIT License](LICENSE).

