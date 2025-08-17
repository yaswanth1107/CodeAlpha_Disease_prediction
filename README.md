# Disease Prediction using Machine Learning

This project focuses on predicting **Breast Cancer**, **Heart Disease**, and **Diabetes** using multiple machine learning models.  
We compare the performance of **Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost**.

---

## 📂 Datasets Used
1. **Breast Cancer (wdbc.data)**
   - Dataset source: UCI Machine Learning Repository
   - Features: 30 features extracted from digitized images of a breast mass
   - Target: `M` = Malignant (1), `B` = Benign (0)

2. **Heart Disease (processed.cleveland.data)**
   - Dataset source: UCI Heart Disease dataset
   - Features: age, sex, chest pain type, cholesterol, blood pressure, etc.
   - Target: Presence (1) or absence (0) of heart disease

3. **Diabetes (diabetes.csv)**
   - Dataset source: Pima Indians Diabetes dataset
   - Features: glucose, insulin, BMI, age, pregnancies, etc.
   - Target: `1` = Positive, `0` = Negative

---

## ⚙️ Preprocessing Steps
- Handled missing values (for heart dataset).
- Standardized numerical features using `StandardScaler`.
- Converted categorical values (like diagnosis `M` / `B`) into numeric.

---

## 🧠 Models Used
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **XGBoost**

---

## 📊 Results (Accuracy %)

| Model               | Breast Cancer | Heart Disease | Diabetes |
|----------------------|--------------|---------------|----------|
| Logistic Regression | 97.36        | 86.66         | 75.32    |
| SVM                 | 95.61        | 88.33         | 75.97    |
| Random Forest       | 96.49        | 88.33         | 72.72    |
| XGBoost             | 95.61        | 83.33         | 72.08    |

---

## 📈 Visualization
- A bar graph is plotted comparing model accuracies across all three datasets.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python file:
   ```bash
   python disease_prediction.py
   ```

---

## 🛠️ Requirements
All requirements are listed in `requirements.txt`

---

## 📌 Author
- **Puthi Yaswanth**  
Data Science Student | ML & AI Enthusiast  
