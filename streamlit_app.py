import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

st.title("Bank Marketing Classification App")

# -------------------------------------------------
# a. Dataset Upload Option
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload Test Dataset (CSV only)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview:")
    st.dataframe(data.head())

    # Target column must be included in test data
    if 'y' in data.columns:
        data['y'] = data['y'].map({'yes': 1, 'no': 0})

        X = data.drop('y', axis=1)
        y_true = data['y']
    else:
        st.error("Dataset must contain target column 'y'")
        st.stop()

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
        "Logistic Regression": "bank_model_logistic.pkl",
        "Decision Tree": "bank_model_dt.pkl",
        "KNN": "bank_model_knn.pkl",
        "Naive Bayes": "bank_model_nb.pkl",
        "Random Forest": "bank_model_rf.pkl",
        "XGBoost": "bank_model_xgb.pkl"
    }

    model = joblib.load(model_paths[model_option])

    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

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

else:
    st.info("Please upload test CSV file to proceed.")
