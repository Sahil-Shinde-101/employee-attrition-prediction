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


<img width="542" height="606" alt="Screenshot 2025-09-28 142004" src="https://github.com/user-attachments/assets/20f8e057-e6a6-4d2d-95ce-acf8353f9932" />



## Model Performance:

Before SMOTE: [1233  237]

After SMOTE: [1233 1233]

AUC Score: 0.9795081967213115

Accuracy scores for each fold: [0.6194332  0.97971602 0.98580122 0.97971602 0.79513185]

Average accuracy: 0.8719596619884864

Confusion Matrix:

[[245   5]

 [ 28 216]]

Classification Report:

               precision    recall  f1-score   support

          No       0.90      0.98      0.94       250
         Yes       0.98      0.89      0.93       244

    accuracy                           0.93       494
   macro avg       0.94      0.93      0.93       494
   
weighted avg       0.94      0.93      0.93       494

Accuracy Score: 0.9331983805668016

Precision Score: 0.9773755656108597

Recall Score: 0.9331983805668016

F1 Score: 0.9290322580645162



## Visualizations:


<img width="712" height="496" alt="Screenshot 2025-09-28 140940" src="https://github.com/user-attachments/assets/52f6fc7a-5833-41db-9413-69311421b3cb" />

### Comparision of heatmap on the basis of counts between Logistic Regression and RF Classifier:

<img width="704" height="493" alt="Screenshot 2025-09-28 140923" src="https://github.com/user-attachments/assets/f3d30b98-f5ff-4406-ad34-14ff14eabfc1" />

<img width="681" height="498" alt="Screenshot 2025-09-28 141050" src="https://github.com/user-attachments/assets/adae9d7d-bf12-4617-8f1b-bcd994a9d4e0" />

### ROC Curve:

<img width="712" height="597" alt="Screenshot 2025-09-28 140844" src="https://github.com/user-attachments/assets/e550b568-ad68-4558-96ef-79764b592a6a" />

### Top 15 features:

<img width="1170" height="744" alt="Screenshot 2025-09-28 140826" src="https://github.com/user-attachments/assets/307e366c-6c7a-44ed-86db-92209ff48673" />




## Key Learnings:

1.Importance of handling imbalanced datasets using SMOTE

2.Using Random Forest for feature importance analysis

3.Comprehensive evaluation through ROC, AUC, Confusion Matrix, and Cross-validation

4.Deployment-ready pipeline with preprocessing and prediction

## Contributing:

Pull requests are welcome! For significant changes, please open an issue first to discuss.


