import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler  # For data standardization

# Function to preprocess the uploaded data based on whether it's for binary or multi-class tasks
def preprocess_data(df, binary=False):
    data = df.copy()
    X = data.drop(data.columns[len(data.columns)-1], axis=1)  # Drop the last column (target)
    if not binary:  # Only apply scaling if not binary classification
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Standardize the features for better model performance
    return X

# Logistic Regression Model Prediction
def logistic_regression(X):
    model = joblib.load('logistic_regression_model.pkl')  # Load the pre-trained Logistic Regression model
    y_pred = model.predict(X)  # Predict the heartbeat types
    print(y_pred)  # Output the predictions
    return y_pred

# Binary Classification followed by Multi-Class Classification for abnormal heartbeats
def binary_then_multiclass(df):
    # Step 1: Binary Classification (Normal vs Abnormal)
    X_bin = preprocess_data(df, binary=True)  # Preprocess the data for binary classification
    binary_model = joblib.load('binary_model.pkl')  # Load the pre-trained binary XGBoost model
    y_pred_bin = binary_model.predict(X_bin)  # Predict normal (0) or abnormal (1)
    print(y_pred_bin)

    # Initialize final predictions with binary results (0 for normal cases)
    y_pred_full = y_pred_bin.copy()

    # Step 2: Multi-Class Classification for abnormal heartbeats
    abnormal_index = np.where(y_pred_bin == 1)[0]  # Find indices of abnormal cases (1)
    if len(abnormal_index) > 0:  # If there are abnormal cases
        X_abnormal = X_bin.iloc[abnormal_index]  # Extract features for the abnormal cases

        # Load the pre-trained multi-class XGBoost model
        multi_model = joblib.load('multi_class_model.pkl')
        y_pred_multi = multi_model.predict(X_abnormal) + 1  # Predict the abnormal class and increment by 1

        # Update the final predictions with multi-class results for abnormal cases
        y_pred_full[abnormal_index] = y_pred_multi
        print(y_pred_full)

    return y_pred_full

# Streamlit App Interface
st.title("Heartbeat Classification System")  # Title of the web app

# File upload option
uploaded_file = st.file_uploader("Upload your heartbeat data file (.csv)", type="csv")  # Accept only .csv files

# Label mapping for the final output (prediction labels)
label_mapping = {
    0: 'N - Normal Beat',
    1: 'S - Supraventricular premature or ectopic beat',
    2: 'V - Premature ventricular contraction',
    3: 'F - Fusion of ventricular and normal beat',
    4: 'Q - Unclassified beat'
}

if uploaded_file:  # Check if a file has been uploaded
    df = pd.read_csv(uploaded_file, header=None)  # Read the uploaded CSV file
    st.write("Data Preview:")
    st.dataframe(df.head())  # Display the first few rows of the uploaded file

    # User selects classification method
    option = st.selectbox(
        "Select Classification Method:",
        ("Logistic Regression", "Binary + Multi-Class")  # Two options for classification methods
    )

    # Predict button
    if st.button("Predict Output"):
        if option == "Logistic Regression":
            st.write("Running Logistic Regression...")
            X = preprocess_data(df)  # Preprocess data for Logistic Regression
            y_pred = logistic_regression(X)  # Predict using Logistic Regression
            y_pred_labels = pd.Series(y_pred).map(label_mapping)  # Map predicted labels to human-readable form
            st.write("Predicted Labels:")
            st.write(y_pred_labels)  # Display the predicted labels

        elif option == "Binary + Multi-Class":
            st.write("Running Binary + Multi-Class Classification...")
            y_pred_full = binary_then_multiclass(df)  # Predict using Binary + Multi-Class models
            y_pred_labels = pd.Series(y_pred_full).map(label_mapping)  # Map predicted labels
            st.write("Predicted Labels (Normal and Abnormal):")
            st.write(y_pred_labels)  # Display the predicted labels
