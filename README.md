1. **Problem statement:**
The objective of this project is to develop and evaluate multiple machine learning classification models to automatically predict whether a breast tumor is:
  Malignant (Cancerous)
  Benign (Non-cancerous)

3. DataSet  Description
      Number of Instances: 569
      Number of Features: 30 numerical features
      Target Variable: diagnosis
              M → Malignant (Cancerous)
              B → Benign (Non-cancerous)
      Feature Type: Continuous numerical values    
      Missing Values: None

4.
<img width="561" height="141" alt="image" src="https://github.com/user-attachments/assets/2bcc0ab5-4fd3-4535-9f21-3542bfc92751" />
ML Model Name	Observation about model performance
Logistic Regression	Logistic Regression performs very strongly with excellent AUC and balanced precision-recall values. It shows that the dataset likely has a reasonably linear decision boundary. The high MCC indicates strong overall classification quality even if classes are slightly imbalanced. It is a robust baseline model for this dataset.
Decision Tree	Decision Tree has the lowest performance among all models. Although precision is high, recall is comparatively lower, which reduces the F1-score. This suggests the tree may be slightly overfitting or not capturing complex patterns as effectively as ensemble methods.
kNN	KNN performs well with strong AUC and balanced precision-recall. It captures non-linear relationships effectively. However, it is computationally expensive at inference time compared to other models.
Naive Bayes	Naive Bayes shows very high AUC but lower accuracy and F1-score compared to Logistic Regression and ensemble models. This indicates that although probability ranking is strong, the independence assumption limits final classification performance.
Random Forest (Ensemble)	Random Forest improves over Decision Tree significantly due to ensemble averaging. It reduces variance and achieves strong overall performance. Precision is perfect (1.0), meaning it produces almost no false positives
XGBoost (Ensemble)	XGBoost is the best-performing model across nearly all metrics. It achieves the highest accuracy, F1-score, and MCC. This indicates that boosting effectively captures complex patterns and interactions in the dataset.
<img width="2042" height="141" alt="image" src="https://github.com/user-attachments/assets/7390f337-be3b-415c-b21b-b8cece45ee37" />







