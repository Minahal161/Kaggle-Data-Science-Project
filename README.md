# Kaggle-Data-Science-Project
# Diabetes Risk Classification Using Health Indicators
An exploratory data science project using machine learning to predict diabetes risk factors. This project analyzes the CDC's health indicators dataset to identify the most significant lifestyle and demographic predictors of diabetes and pre-diabetes. This repository implements a machine learning pipeline to classify diabetes risk using the CDC's Health Indicators dataset sourced from the UCI Machine Learning Repository:https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
## Overview
The goal of this project was to determine if a patient is at risk for diabetes based on indicators such as BMI, cardiac history, and cholesterol. Using the CDC's health dataset, I set this up as a classification task to predict risk levels. I compared three different machine learning models: Random Forest, XGBoost, and K-Nearest Neighbors (KNN). While the initial baseline model had difficulty identifying at-risk patients, as there were significantly fewer diabetic instances than non-diabetic ones, the final Random Forest model achieved a balanced 79% accuracy and 79% weighted recall, making it the most effective version for identifying potential health risks.
## Summary of Work Done
## Data
- Type: 
The project utilizes a tabular dataset directly forked from the UCI Machine Learning Repository. The input consists of a CSV file containing various health indicators, while the output is a binary classification of diabetes risk.
- Attributes: The dataset features 22 columns, including detailed feature names, roles, data types, and demographic information. I conducted an audit for missing values and utilized the raw dataset with all instances to maintain data integrity.
- Size: The dataset contains 253,680 rows.
- Instances (Split): I used the raw dataset with all instances.
- Training: 177,576 patients for training, 38,052 for testing, and 38,052 for validation
