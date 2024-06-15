# churn_prediction.py.

import subprocess

# Function to install dependencies from requirements.txt.
def install_dependencies():
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

# Called the function to install dependencies.
install_dependencies()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import warnings

# Suppress warnings.
warnings.filterwarnings('ignore')

# Configure logging for the log file.
logging.basicConfig(filename='churn_prediction.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading the dataset
logging.info("Loading dataset...")
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 
logging.info("Dataset loaded successfully.")

# Converting 'TotalCharges' to numeric
logging.info("Converting 'TotalCharges' to numeric...")
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].mean(), inplace=True)
logging.info("Conversion complete.")

# Dropping 'customerID'
logging.info("Dropping 'customerID'...")
dataset.drop('customerID', axis=1, inplace=True)
logging.info("Column 'customerID' dropped successfully.")

# Encoding binary categorical features
logging.info("Encoding binary categorical features...")
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_columns:
    dataset[col] = dataset[col].apply(lambda x: 1 if x == 'Yes' or x == 'Male' else 0)
logging.info("Encoding complete.")

# Using get_dummies for other categorical features
logging.info("Encoding categorical features using get_dummies...")
dataset = pd.get_dummies(dataset, drop_first=True)
logging.info("Encoding complete.")

# Defining the feature and the target variable
X = dataset.drop(columns=['Churn'])
y = dataset['Churn']

# Splitting the data into training and testing sets
logging.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split complete.")

# Then initialized the scaler
logging.info("Initializing the scaler...")
scaler = StandardScaler()

# Then fitted the scaler on the training set and transform both the training and testing set
logging.info("Fitting the scaler on the training set and transforming the data...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info("Scaling complete.")

# Initialized the logistic regression model
logging.info("Initializing the logistic regression model...")
log_reg = LogisticRegression(random_state=42)

# Training the model
logging.info("Training the logistic regression model...")
log_reg.fit(X_train_scaled, y_train)
logging.info("Training complete.")

# Making predictions with the model
logging.info("Making predictions with the logistic regression model...")
log_reg_y_pred = log_reg.predict(X_test_scaled)

# Evaluating the model performance
logging.info("Evaluating the logistic regression model performance...")
log_reg_accuracy = accuracy_score(y_test, log_reg_y_pred)
log_reg_precision = precision_score(y_test, log_reg_y_pred)
log_reg_recall = recall_score(y_test, log_reg_y_pred)
log_reg_f1 = f1_score(y_test, log_reg_y_pred)

# Logging the performance metrics
logging.info("Performance metrics for logistic regression model:")
logging.info(f"Accuracy: {log_reg_accuracy}")
logging.info(f"Precision: {log_reg_precision}")
logging.info(f"Recall: {log_reg_recall}")
logging.info(f"F1 Score: {log_reg_f1}")
