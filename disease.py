import streamlit as st
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add custom background here ---
st.markdown(\"""
<style>
.stApp {
    background-color: #e7f6e7;
    /* Uncomment and use a DIRECT image URL if you want a background image! */
    /* background-image: url('https://www.shutterstock.com/image-photo/digital-healthcare-on-futuristic-hologram-concept-2342155133'); */
    background-size: cover;
    background-attachment: fixed;
}
</style>
\""", unsafe_allow_html=True)
# -------------------

st.set_page_config(page_title="Medical Dataset Model Comparison", layout="wide")
st.title("Medical Dataset Model Comparison App")
st.write(\"""
This app loads local copies of the **Breast Cancer**, **Heart Disease**, and **Diabetes** datasets, trains four models (Logistic Regression, SVM, Random Forest, XGBoost), and compares their accuracy on each dataset.
\""")

# Paths to local data files
breast_csv = r"C:\\Users\\Yaswanth\\OneDrive\\Desktop\\Projects\\diseace\\wdbc.data"
heart_csv = r"C:\\Users\\Yaswanth\\OneDrive\\Desktop\\Projects\\diseace\\processed.cleveland.data"
diabetes_csv = r"C:\\Users\\Yaswanth\\OneDrive\\Desktop\\Projects\\diseace\\diabetes.csv"

# --- Data Loading Functions ---
def load_datasets():
    # Breast Cancer
    bc_columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    breast = pd.read_csv(breast_csv, header=None, names=bc_columns)
    breast['diagnosis'] = breast['diagnosis'].map({'M': 1, 'B': 0})

    # Heart Disease
    heart_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    heart = pd.read_csv(heart_csv, names=heart_cols, na_values="?")
    heart.dropna(inplace=True)
    heart["target"] = heart["target"].apply(lambda x: 1 if x > 0 else 0)

    # Diabetes
    diabetes = pd.read_csv(diabetes_csv)

    return breast, heart, diabetes

with st.spinner("Loading datasets..."):
    breast, heart, diabetes = load_datasets()

# --- Model Helper Function ---
def run_models(X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr) * 100
    # SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results['SVM'] = accuracy_score(y_test, y_pred_svm) * 100
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = accuracy_score(y_test, y_pred_rf) * 100
    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    results['XGBoost'] = accuracy_score(y_test, y_pred_xgb) * 100
    return results

# Prepare data for models
X_breast = breast.drop(['id', 'diagnosis'], axis=1)
y_breast = breast['diagnosis']
X_heart = heart.drop('target', axis=1)
y_heart = heart['target']
X_diabetes = diabetes.drop('Outcome', axis=1)
y_diabetes = diabetes['Outcome']

with st.spinner("Training models..."):
    acc_breast = run_models(X_breast, y_breast)
    acc_heart = run_models(X_heart, y_heart)
    acc_diabetes = run_models(X_diabetes, y_diabetes)

# --- Results Table ---
st.subheader("Model Accuracy Comparison (Test Set)")
acc_data = {
    "Model": ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
    "Breast Cancer": [acc_breast[m] for m in acc_breast],
    "Heart Disease": [acc_heart[m] for m in acc_heart],
    "Diabetes": [acc_diabetes[m] for m in acc_diabetes]
}
df_acc = pd.DataFrame(acc_data).set_index("Model")
st.dataframe(df_acc.style.format("{:.2f}"))

# --- Bar Chart ---
st.subheader("Accuracy Comparison Bar Chart")
fig, ax = plt.subplots(figsize=(10, 6))
sns.set(style="whitegrid")
df_acc.plot(kind="bar", ax=ax, colormap="tab10")
plt.title("Model Accuracy Comparison Across Medical Datasets")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=0)
plt.ylim(65, 100)
plt.tight_layout()
plt.legend(title="Dataset")
plt.grid(True, axis="y")
st.pyplot(fig)

# Optional: Show sample rows
with st.expander("Show Example Rows from Datasets"):
    st.write("**Breast Cancer:**")
    st.dataframe(breast.head())
    st.write("**Heart Disease:**")
    st.dataframe(heart.head())
    st.write("**Diabetes:**")
    st.dataframe(diabetes.head())