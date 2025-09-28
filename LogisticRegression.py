import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  


df = pd.read_csv('Employee.csv')

X = df.drop('Attrition', axis=1)
Y = df['Attrition']

column_transformer = ColumnTransformer(
    transformers=[
        ("onehot",OneHotEncoder(sparse_output=True, drop="first"), ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","Over18","OverTime"])
    ],
    remainder="passthrough"  #Keep other column as it is
)

X_transformed = column_transformer.fit_transform(X)

X_transformed_features = pd.DataFrame(X_transformed, columns=column_transformer.get_feature_names_out())

encoder = LabelEncoder()

Y_encoded = encoder.fit_transform(Y)

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_transformed_features, Y_encoded)

print("Before SMOTE:", np.bincount(Y_encoded))
print("After SMOTE:", np.bincount(Y_resampled))

x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000,class_weight="balanced")

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

scores = cross_val_score(model, X_transformed_features, Y_encoded, cv=5, scoring='accuracy')

print("Accuracy scores for each fold:", scores)
print("Average accuracy:", scores.mean())


confusion = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: {confusion}\n")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}\n")

precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision Score: {precision}\n")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall Score: {recall}")

F1Score = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {F1Score}")

conf_matrix_lr = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sb.heatmap(confusion, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Attrition", "Yes Attrition"],
            yticklabels=["No Attrition", "Yes Attrition"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Logistic Regression Confusion Matrix (Counts)')


conf_matrix_lr_percent = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(6,4))
sb.heatmap(conf_matrix_lr_percent, annot=True, fmt=".1f", cmap="plasma",
            xticklabels=["No Attrition", "Yes Attrition"], 
            yticklabels=["No Attrition", "Yes Attrition"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Logistic Regression Confusion Matrix Heatmap (%)')
plt.show()




