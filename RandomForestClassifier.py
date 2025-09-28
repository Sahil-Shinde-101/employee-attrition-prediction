import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report,roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
#All imports

df = pd.read_csv('Employee.csv') #reading a csv file or loading dataset

X = df.drop('Attrition', axis=1)  #dropping the column which is our target(Y)
Y = df['Attrition']               #target column which is employee attrition(yes/no)

#using one hot encoding
column_transformer = ColumnTransformer(   #using column transformer to transform all categorical values into numerical
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=True, drop="first",handle_unknown="ignore"),
         ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","Over18","OverTime"])
    ],
    remainder="passthrough"  #keeping other columns as it it
)

X_transformed = column_transformer.fit_transform(X)                      #fitting and transforming
X_transformed_features = pd.DataFrame(X_transformed, columns=column_transformer.get_feature_names_out()) #converting to dataframe for easier handling


encoder = LabelEncoder()    #using label encoder to encode (yes/no) into (1/0)
Y_encoded = encoder.fit_transform(Y)  #then fit and transform


smote = SMOTE(random_state=42)   #using smote to handle class imbalance or to handle minority classes as RFC and LR both failed in it and showed poor scores
X_resampled, Y_resampled = smote.fit_resample(X_transformed_features, Y_encoded)  #fit and resample x and y

print("Before SMOTE:", np.bincount(Y_encoded))   #printing before SMOTE results
print("After SMOTE:", np.bincount(Y_resampled))  #printing after SMOTE results


x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)
#splitting the resampled data into training and testing sets

#using Random Forest Classifier to train model
rf = RandomForestClassifier(
    n_estimators=200,     #Number of tress
    max_depth=None,
    random_state=42
)
rf.fit(x_train, y_train) #fit model on training data

y_pred = rf.predict(x_test)  #making predictions on test data provided by Train-Test split


importances = rf.feature_importances_   #get feature importances from trained RF model
feature_names = column_transformer.get_feature_names_out()  #getting feature names 

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
#create DataFrame for easier plotting

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#sorting features by imporance 

top_n = 15   #plotting barplot graph of top 15 features to understand which feature matters the most for attrition
top_features = feature_importance_df.head(top_n)
plt.figure(figsize=(10,6))
sb.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title(f'Top {top_n} Most Important Features for Employee Attrition')
plt.xlabel('Importance')
plt.ylabel('Feature')

y_prob = rf.predict_proba(x_test)[:,1]  #here, predicting probabilities for ROC-AUC

fpr, tpr, thresholds = roc_curve(y_test, y_prob) #calulating relation between True Positive Rate and False Positive Rate

auc_score = roc_auc_score(y_test, y_prob)  #calculating AUC score and printing
print("AUC Score:", auc_score)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], color='red', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
#finally plotting the ROC curve

scores = cross_val_score(rf, X_resampled, Y_resampled, cv=5, scoring='accuracy') #performing Cross-Validation on resampled data provided by SMOTE to check accuracy for 5 folds
print("Accuracy scores for each fold:", scores)   #printing accuracy scores for each fold
print("Average accuracy:", scores.mean())         #printing average score among those 5 scores

confusion = confusion_matrix(y_test, y_pred)  #Confusion Matrix to evaluate predicted and actual class labels
print(f"Confusion Matrix:\n{confusion}\n")

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
#Classification Report

#Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred,average='weighted')
f1 = f1_score(y_test, y_pred)

#printing scores of trained model
print(f"Accuracy Score: {accuracy}")
print(f"Precision Score: {precision}")
print(f"Recall Score: {recall}")
print(f"F1 Score: {f1}")

prediction = pd.DataFrame([[38,'Travel_Rarely',371,'Research & Development',2,3,'Life Sciences',1,24,4,'Male',45,3,1,'Research Scientist',4,'Single',3944,4306,5,'Y','Yes',11,3,3,80,0,6,3,3,3,2,1,2]],
                          columns=['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField',
'EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement',
'JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate',
'NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating',
'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
'YearsSinceLastPromotion','YearsWithCurrManager'])
#creating a data frame and putting the values to predict attrition

new_data_transformed = column_transformer.transform(prediction) #it will transform the input using the same ColumnTransformer used during training
                                                                #output will be numpy array
data_frame_to_predict = pd.DataFrame(new_data_transformed, columns=column_transformer.get_feature_names_out())
#it will convert the numpy array to back to data frame
result = rf.predict(data_frame_to_predict) #prints the predicted value but in 0 or as label encoder tansformed it initially 

output = encoder.inverse_transform(result)  #now, we will inverse transform (0/1) into (yes/no)

print(output)            #prints the predicted attrition

#Confusion Matrix Heatmap (counts)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sb.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Attrition", "Yes Attrition"],
            yticklabels=["No Attrition", "Yes Attrition"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Random Forest Classifier Confusion Matrix (Counts)')

#Confusion Matrix Heatmap (percentages)
conf_matrix_percent = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(6,4))
sb.heatmap(conf_matrix_percent, annot=True, fmt=".1f", cmap="coolwarm",
            xticklabels=["No Attrition", "Yes Attrition"], 
            yticklabels=["No Attrition", "Yes Attrition"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Random Forest Confusion Matrix Heatmap (%)')

plt.tight_layout()
plt.show() #showing all graphs 

