# **Heartbeat Classification Project**

This project focuses on classifying heartbeats using data from the MIT-BIH Arrhythmia dataset, available on Kaggle. The project implements three machine learning models to classify ECG (Electrocardiogram) heartbeats: 
1. **Logistic Regression** for initial classification.
2. **Binary Classification** to separate normal and abnormal heartbeats.
3. **Multi-Class Classification** to further classify abnormal heartbeats into specific categories.

Additionally, the project is set up to be run in **Google Colab** and uses **Streamlit** and **Ngrok** for deployment, allowing users to interact with the classification models via a web interface.

## **Project Workflow**

### 1. **Data Preprocessing**
   - The dataset is loaded, and features are scaled using `StandardScaler` for improved model performance.
   - The data is split into **training** and **testing** sets.
   - Class imbalance is addressed using **SMOTE (Synthetic Minority Over-sampling Technique)** for the binary and multi-class classification models.

### 2. **Models Used**
   - **Logistic Regression**: Initially, a logistic regression model is trained to classify the heartbeat data. However, the accuracy was limited due to challenges with imbalanced data and the variety of heartbeat types.
   
   - **Binary Classification (XGBoost)**: A more effective approach is used by first classifying heartbeats as normal (0) or abnormal (1). This improves the accuracy significantly compared to logistic regression.

   - **Multi-Class Classification (XGBoost)**: For heartbeats classified as abnormal, a multi-class classifier further categorizes them into specific abnormalities.

### 3. **Model Accuracy**
   - **Logistic Regression Accuracy**: The Logistic Regression model achieved an accuracy of **66.87%**. It struggled to classify the abnormal heartbeat classes correctly, with particularly low precision and recall for minority classes.
   
   - **Binary Classification (XGBoost)**: After switching to a binary classification model, the performance increased, leading to more accurate predictions of normal vs. abnormal heartbeats.
   
   - **Multi-Class Classification (XGBoost)**: When focusing on abnormal heartbeats, the multi-class model was able to better classify different types of abnormalities. This model significantly improved the results for the more complex classes.

## **Deployment Using Streamlit and Ngrok**

### 1. **Streamlit for Web Interface**
   - The models are deployed using **Streamlit**, a simple and intuitive Python framework to create interactive web applications.
   - The user can upload a CSV file containing heartbeat data, select which model to use (Logistic Regression or Binary + Multi-Class), and view the model predictions directly on the web interface.

### 2. **Ngrok for Public Access**
   - **Ngrok** is used to expose the Streamlit app to the internet, allowing users to interact with the models remotely.
   - A tunnel is created using Ngrok, and a public URL is provided, which can be accessed by anyone to test the model predictions.

## **How to Use This Project**

### **Running the Project in Google Colab**
1. **Set Up the Environment**:
   - Start by uploading the notebook to **Google Colab**.
   - Install the necessary libraries by running:
     ```bash
     !pip install -r requirements.txt
     ```

2. **Download and Preprocess the Data**:
   - The heartbeat dataset is downloaded from Kaggle. In Colab, you can use the Kaggle API to download the dataset.
   - The data is preprocessed, scaled, and split into training and testing sets.

3. **Train the Models**:
   - The models (Logistic Regression, Binary XGBoost, and Multi-Class XGBoost) are trained on the dataset, with the binary and multi-class models trained using SMOTE to handle class imbalance.

4. **Save the Models**:
   - Once the models are trained, they are saved as `.pkl` files using **Joblib** for deployment.

5. **Streamlit App**:
   - The app is designed to allow users to upload a CSV file with heartbeat data, and choose between Logistic Regression or Binary + Multi-Class models for prediction.
   
### **Deploying the Project Using Ngrok**
1. **Install Ngrok and Streamlit**
   - Install **Streamlit** and **Ngrok** to run the web interface and create a tunnel for public access
2. **Set Up Ngrok**:
   - Set up the Ngrok authentication token (replace with your own token) in Colab
3. **Run Streamlit App**
4. **Expose the App Using Ngrok**
   - The generated **public URL** can be used to access the deployed model and interact with it remotely.

### **Testing the Model with Sample Data**
   - A script is provided to create a test sample from the dataset. The code will generate a small random sample (10 rows per class) and save it as `random_test_sample.csv`

## **Project Files**
- **`ecg_classification_pipeline.ipynb`**: The main Jupyter notebook where the models are trained and evaluated.
- **`app.py`**: The Python script for the Streamlit web app.
- **`random_test_sample.csv`**: A sample test dataset to evaluate the models.
- **`logistic_regression_model.pkl`**: The saved Logistic Regression model.
- **`binary_model.pkl`**: The saved Binary Classification model.
- **`multi_class_model.pkl`**: The saved Multi-Class Classification model.
- **`requirements.txt`**: file that includes the libraries needed for your project.

### **How to Install the Dependencies**
You can install all required dependencies using the following command:
```bash
pip install -r requirements.txt
```

