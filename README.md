# Telco-Customer-Churn-Prediction-and-Retention-Strategy
# Customer Churn Prediction

This project focuses on predicting customer churn for a telecom company using various machine learning models. The project aims to build a model that effectively identifies customers likely to churn and provides insights into the factors influencing their decision.

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Evaluation](#model-evaluation)
8. [Using SHAP for Model Interpretation](#using-shap-for-model-interpretation)
9. [How to Use](#how-to-use)
10. [Contributing](#contributing)
11. [License](#license)

## **Project Overview**
The goal of this project is to build a robust machine-learning model that can predict customer churn. The dataset includes various features about customer demographics, services they signed up for, and their account information. We have used different machine learning techniques, including Logistic Regression, Decision Tree, and Random Forest, to identify the best-performing model.

## **Data Description**
The dataset consists of the following information:

- **Customer Demographics**: Gender, Senior Citizen status, Partner status, Dependents.
- **Services Signed Up**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
- **Customer Account Information**: Tenure, Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
- **Target Variable**: Churn (Yes/No).

## **Data Preprocessing**
1. **Handling Missing Values**: Imputed missing values for numerical and categorical columns using mean and mode & dropped some data with missing values.
2. **Encoding Categorical Variables**: Converted categorical variables to numerical format using one-hot encoding.
3. **Data Scaling**: Applied standard scaling to ensure all features are on the same scale.
4. **Splitting the Data**: Split the dataset into training (80%) and testing (20%) sets.

## **Feature Engineering**
1. **Synthetic Minority Over-sampling Technique (SMOTE)**: Applied SMOTE to handle class imbalance by oversampling the minority class.
2. **Combined Resampling**: Combined SMOTE with undersampling of the majority class to balance the dataset more effectively.
3. **Correlation Analysis**: Conducted correlation analysis to identify important features and reduce multicollinearity.

## **Model Training**
We trained the following models using the processed dataset:
1. **Logistic Regression**: A baseline model to understand the relationships between features and the target.
2. **Decision Tree Classifier**: To capture non-linear relationships in the data.
3. **Random Forest Classifier**: An ensemble method to improve prediction accuracy and control overfitting.

Each model was evaluated using metrics like accuracy, precision, recall, and F1-score.

## **Hyperparameter Tuning**
- Used **Grid Search Cross-Validation** to fine-tune hyperparameters for the Random Forest model. Key parameters tuned include:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the trees.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `max_features`: Number of features to consider when looking for the best split.
  - `bootstrap`: Whether bootstrap samples are used when building trees.

- Evaluated the best hyperparameters to improve model performance.

## **Model Evaluation**
Models were evaluated using the following metrics:
- **Accuracy**: Proportion of correct predictions over the total number of cases.
- **Precision**: Number of true positive predictions divided by the total number of positive class predictions.
- **Recall**: Number of true positive predictions divided by the total number of actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.
- **Confusion Matrix**: To visualize true vs. predicted classes.

## **Using SHAP for Model Interpretation**
To understand which features are most important in predicting churn, we used SHAP (SHapley Additive exPlanations):
1. **SHAP Values Calculation**: Calculated SHAP values for Logistic Regression, Decision Tree, and Random Forest models.
2. **Visualization**: Created summary plots to interpret feature importance and their impact on model predictions.


## **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
