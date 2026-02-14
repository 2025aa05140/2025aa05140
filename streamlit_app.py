import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Breast Camcer Prediction App")

# -------------------------------------------------
# a. Test Dataset download Option
# -------------------------------------------------
sample_url = "https://github.com/2025aa05140/2025aa05140/blob/main/model/breast-cancer-wisconsin-test-data.csv?raw=true"

st.write("Download Sample Test Data")

response = requests.get(sample_url)
st.download_button(
label="Click to Save File",
 data=response.content,
 file_name="breast-cancer-wisconsin-test-data.csv",
 mime="text/csv"
    )
uploaded_file = st.file_uploader("Upload Test Dataset (CSV only)", type=["csv"])
if uploaded_file is None:
    uploaded_file ="/mount/src/2025aa05140/model/breast-cancer-wisconsin-test-data.csv"

data = pd.read_csv(uploaded_file)

st.write("Uploaded Dataset Preview:")
st.dataframe(data.head())


    #--------------------
    # -------------------------------------------------
    # b. Model Selection Dropdown
    # -------------------------------------------------

model_option = st.selectbox(
        "Select Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # Load selected model
model_paths = {
        "Logistic Regression": "/mount/src/2025aa05140/model/cancer_model_Logistic Regression.pkl",
        "Decision Tree": "/mount/src/2025aa05140/model/cancer_model_Decision Tree.pkl",
        "KNN": "/mount/src/2025aa05140/model/cancer_model_KNN.pkl",
        "Naive Bayes": "/mount/src/2025aa05140/model/cancer_model_Naive Bayes.pkl",
        "Random Forest": "/mount/src/2025aa05140/model/cancer_model_Random Forest (Ensemble).pkl",
        "XGBoost": "/mount/src/2025aa05140/model/cancer_model_XGBoost (Ensemble).pkl"
    }

model = joblib.load(model_paths[model_option])
X = data.drop('diagnosis', axis=1)
X.fillna(X.mean(), inplace=True) 
y_true = data['diagnosis'].map({'M': 1, 'B': 0})  # Convert target to binary

loaded_object = joblib.load(model_paths[model_option])

if isinstance(loaded_object, tuple):
    model, preprocessor = loaded_object
    X_processed = preprocessor.transform(X)
else:
    model = loaded_object
    X_processed = X

# Make predictions
y_pred = model.predict(X_processed)
y_prob = model.predict_proba(X_processed)[:, 1]

    # -------------------------------------------------
    # c. Display Evaluation Metrics
    # -------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
mcc = matthews_corrcoef(y_true, y_pred)

st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col1.metric("AUC Score", f"{auc:.4f}")

col2.metric("Precision", f"{precision:.4f}")
col2.metric("Recall", f"{recall:.4f}")

col3.metric("F1 Score", f"{f1:.4f}")
col3.metric("MCC Score", f"{mcc:.4f}")

    # -------------------------------------------------
    # d. Confusion Matrix + Classification Report
    # -------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)
st.write(cm)

st.subheader("Classification Report")
report = classification_report(y_true, y_pred)
st.text(report)





























