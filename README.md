1. **Problem statement:**
The objective of this project is to develop and evaluate multiple machine learning classification models to automatically predict whether a breast tumor is:
  Malignant (Cancerous)
  Benign (Non-cancerous)

2. DataSet  Description
      Number of Instances: 569
      Number of Features: 30 numerical features
      Target Variable: diagnosis
              M → Malignant (Cancerous)
              B → Benign (Non-cancerous)
      Feature Type: Continuous numerical values    
      Missing Values: None

Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9649	0.996	0.975	0.9286	0.9512	0.924518
Decision Tree	0.9211	0.9127	0.9024	0.881	0.8916	0.82966
KNN	0.9561	0.9828	0.9744	0.9048	0.9383	0.905824
Naive Bayes	0.9211	0.9894	0.9231	0.8571	0.8889	0.829162
Random Forest (Ensemble)	0.9561	0.9917	1	0.881	0.9367	0.907605
XGBoost (Ensemble)	0.9737	0.995	1	0.9286	0.963	0.944155
<img width="561" height="141" alt="image" src="https://github.com/user-attachments/assets/2bcc0ab5-4fd3-4535-9f21-3542bfc92751" />



