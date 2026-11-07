# Machine Learning Assignment – 2
## Bank Marketing – Classification Models

### a. Problem Statement

The objective of this assignment is to develop and evaluate different Machine Learning classification models using the Bank Marketing dataset.

The goal is to predict whether a customer will subscribe to a term deposit (`yes` or `no`) based on customer demographic information, financial details, and previous marketing campaign interactions.

Five classification algorithms are implemented and evaluated using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### b. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains information about customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

**Dataset Details:**

- Total instances: 45,211
- Total columns: 17
- Input features: 15
- Target variable: `y`
- Classification type: Binary Classification
- Numerical features: 6
- Categorical features: 9
- Missing values: 0

**Target Variable:**

The target variable `y` indicates whether the customer subscribed to a term deposit.

- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

**Data Preprocessing:**

- Checked the dataset for missing values.
- Separated input features and target variable.
- Identified numerical and categorical features.
- Applied appropriate preprocessing to numerical and categorical features.
- Split the dataset into 80% training data and 20% testing data.

Training dataset: 36,168 records.

Testing dataset: 9,043 records.

### c. GitHub Repository Link

**GitHub Repository:**

https://github.com/divyams-bits/ML-Assignment-2

The repository contains the following project files:

- `app.py` – Streamlit application.
- `requirements.txt` – Required Python dependencies.
- `README.md` – Project documentation and model evaluation results.
- `test_data.csv` – Test dataset used for model evaluation.
- `model_comparison.csv` – Comparison of evaluation metrics for all five models.
- `ML2.ipynb` – Jupyter Notebook containing the implementation.
- `models/` – Directory containing all five saved trained classification models in Joblib format.

### d. Models Used and Performance Evaluation

The following five classification models were implemented using the Bank Marketing dataset.

#### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8268 | 0.7465 | 0.3490 | 0.5548 | 0.4285 | 0.3450 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8361 | 0.7913 | 0.3753 | 0.6030 | **0.4627** | **0.3867** |

#### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved a recall of 0.6238, indicating that it identified a relatively high proportion of customers who subscribed to term deposits. However, its precision and F1 score were comparatively low. |
| Decision Tree | Achieved a balanced performance with an accuracy of 0.8268 and F1 score of 0.4285. It captured customer patterns effectively but performed below SVM in overall F1 score. |
| KNN | Achieved the highest accuracy of 0.8874 and precision of 0.5529. However, its recall was only 0.1975, indicating that it missed many customers who actually subscribed to term deposits. |
| Naive Bayes | Achieved an F1 score of 0.4123 and AUC of 0.7514. It provided reasonable classification performance but was outperformed by SVM. |
| Support Vector Machine | Achieved the highest AUC of 0.7913, F1 score of 0.4627, and MCC of 0.3867. It demonstrated the best overall performance among the five classification models based on these evaluation metrics. |

#### Overall Winner for the Dataset

**Support Vector Machine (SVM)**

The Support Vector Machine achieved the highest F1 score of 0.4627, AUC score of 0.7913, and MCC score of 0.3867 among the five classification models.

Although KNN achieved the highest accuracy of 0.8874, its recall was considerably lower. Since the dataset is imbalanced, accuracy alone is not sufficient to determine the best-performing classifier.

Therefore, Support Vector Machine is selected as the overall winner based on its F1 score, AUC score, and MCC.

### Conclusion

In this assignment, five Machine Learning classification algorithms were implemented and evaluated using the Bank Marketing dataset.

The results demonstrate that different classification algorithms perform differently on the same dataset. Support Vector Machine achieved the best overall performance based on F1 score, AUC, and MCC.

The trained models were saved using Joblib and integrated into an interactive Streamlit application. The application allows users to upload test data, select classification models, and view their evaluation metrics and confusion matrices without retraining the models.
