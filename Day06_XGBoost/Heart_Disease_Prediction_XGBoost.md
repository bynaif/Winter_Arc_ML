# 🫀 Heart Disease Prediction using XGBoost

## Project Overview

This project implements a **machine learning pipeline** to predict heart disease using the **Cleveland Heart Disease Dataset**. The model achieves **86.7% accuracy** using XGBoost with advanced techniques including hyperparameter tuning, SMOTE for class imbalance, and comprehensive model evaluation.

---

## 🎯 Objective

Build a robust binary classification model to predict the presence of heart disease based on patient medical attributes, enabling early detection and preventive healthcare interventions.

---

## 📊 Dataset

**Source:** Cleveland Heart Disease Dataset  
**Target Variable:** `condition` (0 = No disease, 1 = Disease present)

### Features:

**Categorical Features (8):**
- `sex` - Gender
- `cp` - Chest pain type
- `fbs` - Fasting blood sugar
- `restecg` - Resting electrocardiographic results
- `exang` - Exercise-induced angina
- `slope` - Slope of peak exercise ST segment
- `ca` - Number of major vessels colored by fluoroscopy
- `thal` - Thalassemia

**Numerical Features (5):**
- `age` - Age in years
- `trestbps` - Resting blood pressure
- `chol` - Serum cholesterol
- `thalach` - Maximum heart rate achieved
- `oldpeak` - ST depression induced by exercise

---

## 🛠️ Technical Implementation

### 1. **Data Preprocessing Pipeline**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessing = ColumnTransformer([
    ("cat_col", OneHotEncoder(handle_unknown="ignore"), cat_coln),
    ("num_col", "passthrough", num_coln)
])
```

- **Categorical encoding:** OneHotEncoder with unknown value handling
- **Numerical features:** Passthrough (no scaling required for tree-based models)
- Integrated into scikit-learn pipeline for seamless workflow

### 2. **Model Selection: XGBoost Classifier**

```python
from xgboost import XGBClassifier

XGBClassifier(
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist"
)
```

**Why XGBoost?**
- Superior performance on tabular data
- Handles missing values
- Built-in regularization prevents overfitting
- Fast training with histogram-based algorithm

### 3. **Hyperparameter Optimization**

Used **RandomizedSearchCV** with **StratifiedKFold (5-fold)** cross-validation:

```python
param_distribution = {
    "learning_rate": [0.1, 0.05, 0.01],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 2, 3, 4, 5]
}
```

**Best Parameters Found:**
- `learning_rate`: 0.05
- `max_depth`: 7
- `n_estimators`: 200
- `colsample_bytree`: 0.6
- `subsample`: 0.6
- `min_child_weight`: 4

**Search Configuration:**
- 50 iterations
- F1-score optimization
- Parallel processing (`n_jobs=-1`)

### 4. **Handling Class Imbalance with SMOTE**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Original Distribution:**
- Class 0: 128 samples
- Class 1: 109 samples

**After SMOTE:**
- Class 0: 128 samples
- Class 1: 128 samples (↑ 17.4% increase)

---

## 📈 Results & Performance Metrics

### Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 87.76% |
| **Test Accuracy** | 86.67% |
| **Precision (Class 0)** | 0.83 |
| **Precision (Class 1)** | 0.92 |
| **Recall (Class 0)** | 0.94 |
| **Recall (Class 1)** | 0.79 |
| **F1-Score (Class 0)** | 0.88 |
| **F1-Score (Class 1)** | 0.85 |

### Confusion Matrix

```
              Predicted
              0      1
Actual  0    30      2
        1     6     22
```

**Key Insights:**
- ✅ **Low False Positives (2):** Minimal unnecessary alarms
- ✅ **Good True Positive Rate:** 22/28 disease cases detected
- ⚠️ **6 False Negatives:** Area for improvement in recall

### ROC-AUC Analysis

The model demonstrates strong discriminative ability with ROC curve analysis showing excellent separation between classes.

---

## 🔍 Key Takeaways

1. **Pipeline Architecture:** End-to-end scikit-learn pipeline ensures reproducibility and deployment readiness
2. **Hyperparameter Tuning:** RandomizedSearchCV efficiently explored 50 configurations, optimizing for F1-score
3. **Class Imbalance Handling:** SMOTE improved minority class representation
4. **Generalization:** Minimal gap between train (87.76%) and test (86.67%) accuracy indicates good generalization
5. **Production Ready:** Model serialized using pickle for deployment

---

## 💻 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)

**Libraries:**
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** pickle

---

## 🚀 Future Enhancements

- [ ] Feature engineering (interaction terms, polynomial features)
- [ ] Ensemble methods (stacking with other algorithms)
- [ ] SHAP values for model interpretability
- [ ] Deployment as REST API using Flask/FastAPI
- [ ] Real-time prediction dashboard with Streamlit

---

## 📝 Model Deployment

The trained model is saved using pickle for easy deployment:

```python
import pickle

with open("xgboost_heart_disease_model.pkl", "wb") as f:
    pickle.dump(best_model_resampled, f)
```

---

## 🤝 Connect & Collaborate

If you found this project helpful, feel free to:
- ⭐ Star the repository
- 💬 Share your feedback
- 🔗 Connect on LinkedIn

---

## 📚 References

- Cleveland Heart Disease Dataset - UCI Machine Learning Repository
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Imbalanced-learn SMOTE: https://imbalanced-learn.org/

---

**Tags:** #MachineLearning #DataScience #XGBoost #HealthcareAI #PredictiveAnalytics #Python #HeartDisease

---

*Developed as part of Winter Arc ML Learning Journey* 🚀
