# IBM HR Analytics: Employee Attrition Prediction 

This project uses Machine Learning (Random Forest Classifier) to predict employee attrition based on various HR and demographic features.

Additionally, the dataset is imbalanced (fewer employees leaving compared to staying). To tackle this, SMOTE (Synthetic Minority Oversampling Technique) is applied, ensuring fair representation of minority classes and improving the model’s predictive performance.

##  Key Features:

Data Preprocessing

One-Hot Encoding for categorical variables

Label Encoding for binary target (Yes/No → 1/0)

SMOTE for handling imbalanced dataset

Model Training

Random Forest Classifier with 200 trees

Train-Test split with 80-20 ratio

Cross-validation for performance consistency

Evaluation Metrics

Accuracy, Precision, Recall, F1-Score

Confusion Matrix (counts & percentages heatmap)

ROC Curve & AUC Score

Feature Importance

Ranking and visualization of top 15 most important features impacting attrition

Prediction Example

User-defined employee profile prediction

## Project Structure:

├── Employee.csv                # Input dataset

├── LogisticRegression.py       # Main script using logistic regression

├── RandomForestClassifier.py   # Main script using RF classifier

├── README.md                    # Project documentation

├── Requirements.txt             #Required libraries

## Installation and Requirements:

### 1.Clone this repository:

git clone https://github.com/Sahil-Shinde-101/employee-attrition-prediction.git

cd employee-attrition-prediction

### 2.Install dependencies:

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn


## Workflow Pipeline:



## Model Performance:



## Visualizations:



## Key Learnings:

1.Importance of handling imbalanced datasets using SMOTE

2.Using Random Forest for feature importance analysis

3.Comprehensive evaluation through ROC, AUC, Confusion Matrix, and Cross-validation

4.Deployment-ready pipeline with preprocessing and prediction

## Contributing:

Pull requests are welcome! For significant changes, please open an issue first to discuss.


