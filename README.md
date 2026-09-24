# 🫀 Heart Attack Risk Prediction & Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.14%2B-4c72b0.svg)](https://seaborn.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-1.1%2B-3b7a57.svg)](https://joblib.readthedocs.io/)

A Machine Learning classification project designed to predict a patient's risk of a heart attack based on clinical and medical records.

---

## 📌 Project Overview

Heart disease is one of the causes of global mortality. Early treatment and risk prediction can improve clinical decision-making and patient outcomes a lot.

In this project, I implemented and compared four fundamental classification algorithms:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**

The primary goal is to evaluate model performance using comprehensive metrics (**Accuracy**, **Precision**, **Recall**, and **F1-Score**) and export the best model with the feature scaler for future deployment.

---

## 📊 Dataset Description (`heart.csv`)

The dataset contains medical observation records for 303 patients with 13 features and 1 target variable (`output`).

| Feature                    | Name in Dataset | Description                                    | Values / Units                                                                     |
| :------------------------- | :-------------- | :--------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Age**                    | `age`           | Age of the patient                             | Years                                                                              |
| **Sex**                    | `sex`           | Gender of the patient                          | `1` = Male, `0` = Female                                                           |
| **Chest Pain Type**        | `cp`            | Type of chest pain experienced                 | `0` = Typical Angina, `1` = Atypical Angina, `2` = Non-anginal, `3` = Asymptomatic |
| **Resting Blood Pressure** | `trtbps`        | Resting blood pressure                         | mm Hg                                                                              |
| **Cholesterol**            | `chol`          | Serum cholesterol                              | mg/dl                                                                              |
| **Fasting Blood Sugar**    | `fbs`           | Fasting blood sugar > 120 mg/dl                | `1` = True, `0` = False                                                            |
| **Resting ECG**            | `restecg`       | Resting electrocardiographic results           | `0` = Normal, `1` = ST-T wave abnormality, `2` = Left ventricular hypertrophy      |
| **Max Heart Rate**         | `thalachh`      | Maximum heart rate achieved                    | bpm                                                                                |
| **Exercise Angina**        | `exng`          | Exercise-induced angina                        | `1` = Yes, `0` = No                                                                |
| **ST Depression**          | `oldpeak`       | ST depression induced by exercise              | Numeric float                                                                      |
| **ST Slope**               | `slp`           | Slope of peak exercise ST segment              | `0`, `1`, `2`                                                                      |
| **Major Vessels**          | `caa`           | Number of major vessels colored by fluoroscopy | `0` to `3`                                                                         |
| **Thalassemia**            | `thall`         | Thalassemia status                             | `0` to `3`                                                                         |
| **Target Output**          | `output`        | **Target Class**: Risk of heart attack         | **`0` = Low Chance, `1` = High Chance**                                            |

---

## 🛠️ Machine Learning Pipeline

1. **Exploratory Data Analysis & Cleaning:**
   - Evaluated data shape, data types, missing values, and duplicate rows.
   - Handled duplicate entries to prevent model overfitting.

2. **Data Preprocessing & Feature Scaling:**
   - Separated independent features ($X$) from target variable ($y$).
   - Split dataset into **80% Training** and **20% Testing** sets with stratification to preserve target distribution.
   - Scaled numerical features using `StandardScaler` to normalize feature magnitude.

3. **Model Training & Hyperparameter Setup:**
   - Built and trained Logistic Regression, KNN, Decision Tree, and Support Vector Machine models.

4. **Evaluation & Visualization:**
   - Evaluated model predictions on the unseen test set using Confusion Matrices, Accuracy, Precision, Recall, and F1-Score.

5. **Artifact Export:**
   - Serialized the top-performing model (`best_heart_disease_model.pkl`) and the pre-fitted scaler (`scaler.pkl`) using `joblib`.

---

## 📈 Model Performance Comparison

Summary of model metrics evaluated on the test dataset:

| Model                            |  Accuracy  | Precision  |   Recall   |  F1-Score  |        Status         |
| :------------------------------- | :--------: | :--------: | :--------: | :--------: | :-------------------: |
| **Support Vector Machine (SVM)** | **83.61%** | **79.49%** | **93.94%** | **86.11%** | 🏆 **Best Performer** |
| **K-Nearest Neighbors (KNN)**    |   80.33%   |   76.92%   |   90.91%   |   83.33%   |     🥈 Runner-up      |
| **Logistic Regression**          |   78.69%   |   76.32%   |   87.88%   |   81.69%   |     🥉 3rd Place      |
| **Decision Tree**                |   72.13%   |   71.05%   |   81.82%   |   76.06%   |     👎 4th Place      |

![Comparison Chart](./Plots/model%20comparison%20char.jpg)

> 📌 **Support Vector Machine (SVM)** achieved the highest performance across all evaluation metrics, excelling in **Recall (93.94%)** and **F1-Score (86.11%)**, making it the most reliable model for medical risk detection where minimizing false negatives is critical.

---

## 📂 Repository Structure

```text
.
├── heart.csv                      # Clinical Heart Disease Dataset
├── heart_attack_prediction.ipynb  # Main Jupyter Notebook
├── best_heart_disease_model.pkl   # Serialized best trained ML model (SVM)
├── scaler.pkl                     # Serialized StandardScaler object
├── Plots                          # Plots & Charts
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

\- [Tony White](https://github.com/itstonywhite) ✍️
