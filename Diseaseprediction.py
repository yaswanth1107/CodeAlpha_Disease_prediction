#Unzip the Files
import zipfile
import os
#Creating Folder to extract each Zip
os.makedirs("data/breast_cancer", exist_ok=True)
os.makedirs("data/heart", exist_ok=True)
os.makedirs("data/diabetes", exist_ok=True)
# Checking the Files That are Extracted
print("Breast Cancer Folder:", os.listdir("data/breast_cancer"))
print("Heart Disease Folder:", os.listdir("data/heart"))
print("Diabetes Folder:", os.listdir("data/diabetes"))

# Loading all Datasets
import pandas as pd

# Breast Cancer
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
breast = pd.read_csv(r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\wdbc.data", header=None, names=columns)

# Map diagnosis to binary values
breast['diagnosis'] = breast['diagnosis'].map({'M': 1, 'B': 0})

# Diabetes
diabetes = pd.read_csv(r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\diabetes.csv")

# Heart Disease
heart_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
heart = pd.read_csv(
    r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\processed.cleveland.data",
    names=heart_columns,
    na_values="?"
)
heart.dropna(inplace=True)
heart["target"] = heart["target"].apply(lambda x: 1 if x > 0 else 0)
#Preprocessing & Model Training 
#Importing the required files
!pip install numpy==1.25.2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

#Breast Cancer
# Drop ID column
X_breast = breast.drop(['id', 'diagnosis'], axis=1)
y_breast = breast['diagnosis']

# Heart Disease
# Drop ID column
X_heart = heart.drop('target', axis=1)
y_heart = heart['target']

# Diabetes
# Drop ID column
X_diabetes = diabetes.drop('Outcome', axis=1)
y_diabetes = diabetes['Outcome']
# Split the columns for test and Training all the three different Medical Files
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)


# Now Scaling the train test 
# Breast Cancer
scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)

# Heart Disease
scaler_h = StandardScaler()
X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)

#Diabetes
scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)
# Using the LogisticRegression for Training and Testing
# Breast Cancer
model_b = LogisticRegression()
model_b.fit(X_train_b, y_train_b)

# Heart Disease
model_h = LogisticRegression(max_iter=1000)
model_h.fit(X_train_h, y_train_h)

#Diabetes
model_d = LogisticRegression(max_iter=1000)
model_d.fit(X_train_d, y_train_d)
#Predict & Evaluate

# Breast Cancer
y_pred_b = model_b.predict(X_test_b)

# Heart Disease
y_pred_h = model_h.predict(X_test_h)

# Diabetes
y_pred_d = model_d.predict(X_test_d)

# Breast Cancer
print("Breast Cancer Accuracy:", accuracy_score(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b))


# Heart Disease
print("Heart Disease Accuracy:", accuracy_score(y_test_h, y_pred_h))
print(classification_report(y_test_h, y_pred_h))

# Diabetes
print("Diabetes Accuracy:", accuracy_score(y_test_d, y_pred_d))
print(classification_report(y_test_d, y_pred_d))

# Comparing all the 3 Accuracies
print("Breast Cancer Accuracy:", accuracy_score(y_test_b, y_pred_b))
print("Heart Disease Accuracy:", accuracy_score(y_test_h, y_pred_h))
print("Diabetes Accuracy:", accuracy_score(y_test_d, y_pred_d))
# Using SUPPORT VECTOR MACHINE(SVM)
from sklearn.svm import SVC

# FOR BREAST CANCER
df = pd.read_csv(r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\wdbc.data", header=None)
# Let's assume the format is correct — extract features & labels
X = df.iloc[:, 2:]  # Features start from column index 2
y = df.iloc[:, 1]   # Diagnosis column
# Convert diagnosis to 0 and 1 (M = 1, B = 0)
y = y.map({'M': 1, 'B': 0})
# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
# SVM FOR Heart Disease
df_heart = pd.read_csv(r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\processed.cleveland.data", header=None, na_values='?')

# Droping the rows with missing values
df_heart.dropna(inplace=True)
# Assign column names if not present
df_heart.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Convert target to binary: 0 = no disease, 1 = disease
df_heart['target'] = df_heart['target'].apply(lambda x: 1 if x > 0 else 0)
# Split into X and y
X_heart = df_heart.drop('target', axis=1)
y_heart = df_heart['target']
# Standardize features
scaler = StandardScaler()
X_heart_scaled = scaler.fit_transform(X_heart)
# Train/test split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42
)
# SVM model
svm_heart = SVC(kernel='linear')
svm_heart.fit(X_train_h, y_train_h)
y_pred_h = svm_heart.predict(X_test_h)
# Accuracy
acc_heart_svm = accuracy_score(y_test_h, y_pred_h)
print("Heart Disease SVM Accuracy:", acc_heart_svm)
# SVM FOR DIABETES
# Load Diabetes data
df_diabetes = pd.read_csv(r"C:\Users\Yaswanth\OneDrive\Desktop\Projects\diseace\diabetes.csv")
# Features and target
X_diabetes = df_diabetes.drop('Outcome', axis=1)
y_diabetes = df_diabetes['Outcome']
# Standardize
X_d_scaled = scaler.fit_transform(X_diabetes)
# Train/test split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d_scaled, y_diabetes, test_size=0.2, random_state=42
)
# SVM model
svm_diabetes = SVC(kernel='linear')
svm_diabetes.fit(X_train_d, y_train_d)
y_pred_d = svm_diabetes.predict(X_test_d)
# Accuracy
acc_diabetes_svm = accuracy_score(y_test_d, y_pred_d)
print("Diabetes SVM Accuracy:", acc_diabetes_svm)
print("Breast Cancer SVM Accuracy:", accuracy_svm)
print("Heart Disease SVM Accuracy:", acc_heart_svm)
print("Diabetes SVM Accuracy:", acc_diabetes_svm)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier



# Breast Cancer - Random Forest
# Model
rf_breast = RandomForestClassifier(n_estimators=100, random_state=42)
rf_breast.fit(X_train_b, y_train_b)
y_pred_rf_breast = rf_breast.predict(X_test_b)


# Heart Disease - Random Forest
# Model
rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
rf_heart.fit(X_train_h, y_train_h)
y_pred_rf_heart = rf_heart.predict(X_test_h)


# Diabetes - Random Forest
# Model
rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_diabetes.fit(X_train_d, y_train_d)
y_pred_rf_diabetes = rf_diabetes.predict(X_test_d)



# Finding Accuracy For all the Three

# Breast Cancer
# Accuracy
acc_rf_breast = accuracy_score(y_test_b, y_pred_rf_breast)
print("Breast Cancer Random Forest Accuracy:", acc_rf_breast)


# Heart Disease
# Accuracy
acc_rf_heart = accuracy_score(y_test_h, y_pred_rf_heart)
print("Heart Disease Random Forest Accuracy:", acc_rf_heart)


# Diabetes
# Accuracy
acc_rf_diabetes = accuracy_score(y_test_d, y_pred_rf_diabetes)
print("Diabetes Random Forest Accuracy:", acc_rf_diabetes)




!pip install xgboost
#XGBoost
from xgboost import XGBClassifier



# Breast Cancer
xgb_breast = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_breast.fit(X_train_b, y_train_b)
y_pred_xgb_breast = xgb_breast.predict(X_test_b)



# Heart Disease
xgb_heart = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_heart.fit(X_train_h, y_train_h)
y_pred_xgb_heart = xgb_heart.predict(X_test_h)



# Diabetes
xgb_diabetes = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_diabetes.fit(X_train_d, y_train_d)
y_pred_xgb_diabetes = xgb_diabetes.predict(X_test_d)




# Finding Accuracy for all the Three

# Breast Cancer
acc_xgb_breast = accuracy_score(y_test_b, y_pred_xgb_breast)
print("Breast Cancer XGBoost Accuracy:", acc_xgb_breast)

# Heart Disease
acc_xgb_heart = accuracy_score(y_test_h, y_pred_xgb_heart)
print("Heart Disease XGBoost Accuracy:", acc_xgb_heart)

#Diabetes
acc_xgb_diabetes = accuracy_score(y_test_d, y_pred_xgb_diabetes)
print("Diabetes XGBoost Accuracy:", acc_xgb_diabetes)



# Plotting Bar Graph to show all the 4 models Prediction of Accuracy
!pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns



# Accuracy data
data = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
    'Breast Cancer': [97.36, 95.61, 96.49, 95.61],
    'Heart Disease': [86.66, 88.33, 88.33, 83.33],
    'Diabetes': [75.32, 75.97, 72.72, 72.08]
}




# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier



# Breast Cancer - Random Forest
# Model
rf_breast = RandomForestClassifier(n_estimators=100, random_state=42)
rf_breast.fit(X_train_b, y_train_b)
y_pred_rf_breast = rf_breast.predict(X_test_b)


# Heart Disease - Random Forest
# Model
rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
rf_heart.fit(X_train_h, y_train_h)
y_pred_rf_heart = rf_heart.predict(X_test_h)


# Diabetes - Random Forest
# Model
rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_diabetes.fit(X_train_d, y_train_d)
y_pred_rf_diabetes = rf_diabetes.predict(X_test_d)



# Finding Accuracy For all the Three

# Breast Cancer
# Accuracy
acc_rf_breast = accuracy_score(y_test_b, y_pred_rf_breast)
print("Breast Cancer Random Forest Accuracy:", acc_rf_breast)


# Heart Disease
# Accuracy
acc_rf_heart = accuracy_score(y_test_h, y_pred_rf_heart)
print("Heart Disease Random Forest Accuracy:", acc_rf_heart)


# Diabetes
# Accuracy
acc_rf_diabetes = accuracy_score(y_test_d, y_pred_rf_diabetes)
print("Diabetes Random Forest Accuracy:", acc_rf_diabetes)




!pip install xgboost
#XGBoost
from xgboost import XGBClassifier



# Breast Cancer
xgb_breast = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_breast.fit(X_train_b, y_train_b)
y_pred_xgb_breast = xgb_breast.predict(X_test_b)



# Heart Disease
xgb_heart = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_heart.fit(X_train_h, y_train_h)
y_pred_xgb_heart = xgb_heart.predict(X_test_h)



# Diabetes
xgb_diabetes = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_diabetes.fit(X_train_d, y_train_d)
y_pred_xgb_diabetes = xgb_diabetes.predict(X_test_d)




# Finding Accuracy for all the Three

# Breast Cancer
acc_xgb_breast = accuracy_score(y_test_b, y_pred_xgb_breast)
print("Breast Cancer XGBoost Accuracy:", acc_xgb_breast)

# Heart Disease
acc_xgb_heart = accuracy_score(y_test_h, y_pred_xgb_heart)
print("Heart Disease XGBoost Accuracy:", acc_xgb_heart)

#Diabetes
acc_xgb_diabetes = accuracy_score(y_test_d, y_pred_xgb_diabetes)
print("Diabetes XGBoost Accuracy:", acc_xgb_diabetes)



# Plotting Bar Graph to show all the 4 models Prediction of Accuracy
!pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns



# Accuracy data
data = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
    'Breast Cancer': [97.36, 95.61, 96.49, 95.61],
    'Heart Disease': [86.66, 88.33, 88.33, 83.33],
    'Diabetes': [75.32, 75.97, 72.72, 72.08]
}




# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)




# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier



# Breast Cancer - Random Forest
# Model
rf_breast = RandomForestClassifier(n_estimators=100, random_state=42)
rf_breast.fit(X_train_b, y_train_b)
y_pred_rf_breast = rf_breast.predict(X_test_b)


# Heart Disease - Random Forest
# Model
rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
rf_heart.fit(X_train_h, y_train_h)
y_pred_rf_heart = rf_heart.predict(X_test_h)


# Diabetes - Random Forest
# Model
rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_diabetes.fit(X_train_d, y_train_d)
y_pred_rf_diabetes = rf_diabetes.predict(X_test_d)



# Finding Accuracy For all the Three

# Breast Cancer
# Accuracy
acc_rf_breast = accuracy_score(y_test_b, y_pred_rf_breast)
print("Breast Cancer Random Forest Accuracy:", acc_rf_breast)


# Heart Disease
# Accuracy
acc_rf_heart = accuracy_score(y_test_h, y_pred_rf_heart)
print("Heart Disease Random Forest Accuracy:", acc_rf_heart)


# Diabetes
# Accuracy
acc_rf_diabetes = accuracy_score(y_test_d, y_pred_rf_diabetes)
print("Diabetes Random Forest Accuracy:", acc_rf_diabetes)




!pip install xgboost
#XGBoost
from xgboost import XGBClassifier



# Breast Cancer
xgb_breast = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_breast.fit(X_train_b, y_train_b)
y_pred_xgb_breast = xgb_breast.predict(X_test_b)



# Heart Disease
xgb_heart = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_heart.fit(X_train_h, y_train_h)
y_pred_xgb_heart = xgb_heart.predict(X_test_h)



# Diabetes
xgb_diabetes = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_diabetes.fit(X_train_d, y_train_d)
y_pred_xgb_diabetes = xgb_diabetes.predict(X_test_d)




# Finding Accuracy for all the Three

# Breast Cancer
acc_xgb_breast = accuracy_score(y_test_b, y_pred_xgb_breast)
print("Breast Cancer XGBoost Accuracy:", acc_xgb_breast)

# Heart Disease
acc_xgb_heart = accuracy_score(y_test_h, y_pred_xgb_heart)
print("Heart Disease XGBoost Accuracy:", acc_xgb_heart)

#Diabetes
acc_xgb_diabetes = accuracy_score(y_test_d, y_pred_xgb_diabetes)
print("Diabetes XGBoost Accuracy:", acc_xgb_diabetes)



# Plotting Bar Graph to show all the 4 models Prediction of Accuracy
!pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns



# Accuracy data
data = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
    'Breast Cancer': [97.36, 95.61, 96.49, 95.61],
    'Heart Disease': [86.66, 88.33, 88.33, 83.33],
    'Diabetes': [75.32, 75.97, 72.72, 72.08]
}




# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)



# Plot
plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
df.plot(kind='bar', figsize=(12, 6), colormap='tab10')

plt.title('Model Accuracy Comparison Across Medical Datasets')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=0)
plt.ylim(65, 100)
plt.legend(title='Dataset')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()