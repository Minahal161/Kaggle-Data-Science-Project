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
## Preprocessing/Data Clean Up
For the preprocessing, I handled the outliers in the numerical columns by removing any data points that had a Z-score greater than 3 or less than -3. After removing the outliers, I standardized the numerical columns using a standard scaler so that the features would have a mean close to 0 and a standard deviation of 1, which is essential for models like KNN. Finally, I ensured that all numerical entries were converted to integers where necessary to prevent errors during model training.
## Data Visualization
<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/a7c9cf90-9d37-4fe4-b36f-0903b0e005b0" />
<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/48305146-30e3-44e8-94f3-fd45ac8199e3" />
Here are two of the more promising features that have a relatively good distribution between the classes. The distribution in these histograms is slightly skewed in these examples. These histograms also show a clear imbalance as there are more non-diabetic instances than diabetic, which is something to keep in mind when running our models.
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/4f8990b0-68be-4c3c-9162-2cf71aac4186" />
This feature does not have enough meaningful data for our model to learn from, especially in the diabetic class.
## Problem Formulation
### Input/Output
- Input: Health indicators and patient history (BMI, cholesterol, etc.).
- Output: Prediction of diabetes risk (Binary Classification).
## Models
To accurately predict diabetes risk, I needed models that could analyze multiple health factors at the same time to find hidden patterns. I chose to use ensemble methods like Random Forest and XGBoost, as well as KNN, because they are designed to look at how different features interact with each other. These models help detect patterns in the data that a simple baseline might miss.
## Training
I used a variety of models for training, specifically Random Forest, XGBoost, and K-Nearest Neighbors (KNN). During the process, I ran into some initial warnings with XGBoost where the model wouldn't finish properly; I resolved this by ensuring all data types were consistent before starting the training. For the KNN model, I made sure to standardize the data first since that algorithm is sensitive to the scale of the numbers. To keep the process efficient, I used a structured training and testing split to ensure the model saw enough examples of both classes. After noticing the baseline model had trouble with the smaller group of diabetic cases, I adjusted the training data to be more balanced. This allowed the models to learn the patterns for both groups more effectively, resulting in the final performance scores.
## Performance Comparison
<img width="652" height="296" alt="image" src="https://github.com/user-attachments/assets/4744d06a-6ed7-4a26-9a2c-ef9478d11ac8" />
Baseline Model Evalution Metrics
<img width="671" height="505" alt="image" src="https://github.com/user-attachments/assets/5ed75323-9947-46df-a4ff-a5f2e7c243e1" />
Baseline Model ROC Curve
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/0fb23860-e73a-4e5f-91d7-95899daae13a" />
Decision Function Histogram
<img width="675" height="262" alt="image" src="https://github.com/user-attachments/assets/9b01471b-ca09-42f4-908d-2f0eb3d0f77d" />
SMOTE Random Forest Evaluation Metrics
<img width="789" height="590" alt="image" src="https://github.com/user-attachments/assets/897f2a02-ba02-4a43-a426-d2f3d384ff97" />
SMOTE Random Forest ROC Curve (slightly different)
<img width="686" height="142" alt="image" src="https://github.com/user-attachments/assets/0347619b-8cc8-432e-ac1e-0eb178c1d483" />
Model Comparison Table (Only focused on recall as false negatives are a priority)
<img width="802" height="378" alt="image" src="https://github.com/user-attachments/assets/d596efea-9d3c-429c-ab42-b8d611c9beeb" />
For the project, I focused on Recall and Accuracy as the primary metrics. In the medical field, Recall is especially important because it ensures we are identifying as many true positive cases as possible and minimizing "false negatives" (missing a patient who is actually at risk). Accuracy was also tracked to see the overall correctness of the predictions. While the ROC curve helps us see how well the model separates the two groups, our results showed that reaching a high true-positive rate without increasing false positives is a challenge with this specific data. However, the Random Forest model provided the most balanced results.
## Conclusions
Random Forest with SMOTE has the highest scores relatively for both classes, considering the class imbalance solution. But our baseline had the highest without addressing the class imbalance, meaning that it was really only classifying non-diabetic patients really well and failing to catch those at risk. SMOTE is a good way to oversample, but there is nuance to this technique as it might not always apply to the dataset properly. Other models surprisingly performed poorly considering the class imbalance solution and their nature in classification problem contexts.
## Future Work
### Next Steps
The next thing I would try is testing different ways to balance the dataset to see if it helps the models catch more cases. I would also try removing the less important health indicators to see if a smaller, more focused list of features helps models like KNN perform better. Additionally, adding a "pre-diabetes" category could be very helpful for healthcare workers to identify patients before they become high-risk.
### Other Potential Studies
Starting from this project, further studies could be done to find the exact threshold where a patient's risk significantly increases. It would also be interesting to study how non-medical factors, such as education levels or income, impact diabetes risk compared to physical indicators like BMI. This could help create a more well-rounded approach to predicting health risks in different communities.
## How to Reproduce Results 
To reproduce the results of this study, follow the steps below:
### 1. Environment and Resources
I recommend using Google Colab or a Jupyter Notebook environment for this project. The following Python libraries are required:
- pandas and numpy for data handling.
- matplotlib and seaborn for visualizations.
- ucimlrepo is mandatory to automate the download(to fetch the datset)
- scipy to support the mathematical and statistical calculations used in preprocessing.
- scikit-learn for the Random Forest and KNN models.
- xgboost for the gradient boosting model.
- imblearn for balancing the dataset.
### 2. Execution Steps
- Download the Data: Obtain the diabetes health indicators dataset from the UCI Machine Learning Repository.
- Run Preprocessing: Execute the cleaning cells to remove outliers using the Z-score method and to scale the numerical features.
- Data Partitioning: Run the split cells to divide the data into 70% Training, 15% Validation, and 15% Testing sets.
- Balance the Training Set: Apply the balancing technique (found in the "Using SMOTE" section of the code) to the training data.
- Train the Model: Run the Random Forest Classifier cell with n_estimators=100.
- Evaluate: Use the validation set to generate the classification report and confirm the 79% accuracy.
## Overview of files in repository
Below are the relevant files and their roles:
- Kaggle Tabular Data Project.ipynb: This is the primary notebook containing the full pipeline. It includes data fetching via ucimlrepo, outlier removal, feature scaling, data balancing, and the training of all three models (Random Forest, XGBoost, and KNN).
- diabetes_health_indicators_data.csv: (If downloaded) The raw dataset containing health indicators for 253,680 patients.
- README.md: This file, providing a high-level overview of the project, performance metrics, and instructions for reproduction.
## Software Setup
### Required Packages
- pandas: For data manipulation and analysis.
- numpy: For numerical operations and array handling.
- matplotlib & seaborn: For generating the density histograms and feature importance plots.
- scikit-learn: For the core Machine Learning algorithms (Random Forest, KNN) and evaluation metrics.
### Specialized Libraries:
- ucimlrepo: Required to pull the dataset directly from the UCI Machine Learning Repository via API.
- imbalanced-learn (imblearn): Necessary for the balancing techniques used to address the class imbalance.
- xgboost: Provides the Gradient Boosting framework used for the XGBoost model.
- scipy: Used for advanced statistical calculations like Z-score outlier detection.
### Installation Instructions
If you are running this project locally or in a new environment, you can install all dependencies at once using pip. Open your terminal or a notebook cell and run:
  pip install ucimlrepo imbalanced-learn xgboost scipy pandas numpy matplotlib scikit-learn
## Data
These directions are also directly on the website when clicking Import in Python:
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
Instead of downloading the data manually, this project uses the ucimlrepo library to import the data directly into the notebook. This ensures that the most up-to-date version of the dataset is used and that the feature names are correctly formatted.
## Preprocessing Steps
To prepare the data for the machine learning models, I followed these specific steps in the notebook:
-1 Direct Import: I used the UCI Python import tool to pull the dataset (ID 891) into a pandas dataframe.
-2 Outlier Removal: I calculated the Z-scores for numerical columns and removed any rows that were more than 3 standard deviations away from the mean to prevent extreme values from confusing the model.
-3 Feature Scaling: I applied a StandardScaler to the data. This is a critical step because it ensures that features with large numbers (like BMI) don't overpower features with small numbers (like Age) during the training process.
-4 Handling Imbalance: Because there were many more healthy patients than diabetic patients in the raw data, I balanced the training set using a resampling technique. This step was essential to ensure the model could actually recognize the signs of diabetes rather than just guessing "healthy" every time.
-5 Data Splitting: Finally, I split the data into three sets: Training, Validation, and Testing to properly evaluate the model's performance on unseen data.
## Training
To perform training, make sure no instances are continuous and that the variables are consistent with their data type listed at the beginning. If there are continuous variables that are supposed to be integers or floats that are supposed to be integers, convert all numerical columns to integers. Remove the ID column as well so it doesn't interfere. Set X equal to your feature list (ENSURE YOU ARE USING THE PREPROCESSED VARIABLES) and y equal to your risk (yes or no) list and then split the training and the testing set using the 70-30 method (70% is used for training and 30% is used for testing). Use X_temp as a temporary version of the dataset as to not change anything with the regular dataset. If using SMOTE, make sure to apply SMOTE to your X and Y training and test sets in order to have it apply properly to the class distribution
### Perform Evaluation
To evaluate the model's performance, you can use a combination of statistical reports and visual tools provided in the notebook. By generating a Classification Report, you can track the Recall and Accuracy scores, which are vital for ensuring that diabetic cases are correctly identified. To see exactly where the model is making errors, the Confusion Matrix provides a clear breakdown of true positives versus false negatives. Additionally, the Decision Function Histogram allows you to visualize how well the model distinguishes between classes by showing the distribution of scores for both groups; a clear separation between the distributions indicates a more reliable model. Finally, the ROC Curve serves as a final check to ensure the model maintains a high true-positive rate without significantly increasing false positives, confirming the model's effectiveness in a healthcare context.
## Citations
- Centers for Disease Control and Prevention. (2022, April 29). CDC - 2014 BRFSS survey data and Documentation. Centers for Disease Control and Prevention.
https://www.cdc.gov/brfss/annual_data/annual_2014.html
- Centers for Disease Control and Prevention. (2024, May 15). Diabetes Risk Factors. Diabetes.https://www.cdc.gov/diabetes/risk-factors/index.html
