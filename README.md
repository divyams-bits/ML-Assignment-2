# Machine Learning Assignment 2 – Bank Marketing Classification

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a customer will subscribe to a term deposit based on their demographic information and previous marketing campaign interactions.

The project implements five classification algorithms and compares their performance using different evaluation metrics.

An interactive Streamlit web application is developed to allow users to upload test datasets, select classification models, and view their evaluation results.

## 2. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains information about customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

The target variable is `y`, which indicates whether a customer subscribed to a term deposit.

### Dataset Details

| Attribute | Description |
|---|---|
| Total instances | 45,211 |
| Original columns | 17 |
| Input features used | 15 |
| Numerical features | 6 |
| Categorical features | 9 |
| Target variable | y |
| Classification type | Binary classification |
| Training samples | 36,168 |
| Testing samples | 9,043 |

The `duration` feature was excluded to avoid data leakage because call duration is only known after the call has ended.

### Preprocessing Techniques

- Checked the dataset for missing values.
- Separated input features and target variable.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Split the dataset into 80% training data and 20% testing data.
- Used stratified sampling to maintain the target class distribution.

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/divyams-bits/ML-Assignment-2

The repository contains:

- `app.py` – Streamlit application source code.
- `requirements.txt` – Required Python dependencies.
- `test_data.csv` – Test dataset used for model evaluation.
- `model_comparison.csv` – Comparison of classification model performance.
- `models/` – Contains all five saved trained classification models.
- `README.md` – Project documentation.

### Saved Models

1. logistic_regression.joblib
2. decision_tree.joblib
3. knn.joblib
4. naive_bayes.joblib
5. support_vector_machine.joblib

## 4. Models Used and Evaluation Metrics

Five machine learning classification algorithms were implemented using the same Bank Marketing dataset.

### 4.1 Model Performance Comparison

The following table presents the evaluation results obtained on the test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.2853 | 0.3732 |
| Decision Tree | 0.8268 | 0.7465 | 0.3490 | 0.5548 | 0.3450 | 0.4285 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2833 | 0.2911 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.3271 | 0.4123 |
| Support Vector Machine | 0.8361 | 0.7913 | 0.3753 | 0.6030 | 0.4627 | 0.3867 |

### 4.2 Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an accuracy of 75.48% and an AUC score of 0.7722. It achieved a recall of 0.6238, indicating that it identified a relatively high proportion of actual subscribers. However, its precision and F1 score were comparatively low.

**Decision Tree**

The Decision Tree classifier achieved an accuracy of 82.68%. It obtained an MCC score of 0.4285, the highest among the five models. It provided a reasonable balance between precision and recall.

**K-Nearest Neighbors (KNN)**

KNN achieved the highest accuracy of 88.74% and precision of 0.5529. However, its recall was only 0.1975, indicating that it missed many actual subscribers. Its F1 score was consequently low.

**Naive Bayes**

Naive Bayes achieved an accuracy of 84.52% and an AUC score of 0.7514. It provided moderate precision and recall, but its overall F1 score was lower than that of the Decision Tree and SVM models.

**Support Vector Machine (SVM)**

SVM achieved the highest F1 score of 0.4627 and the highest AUC score of 0.7913. It also achieved a recall of 0.6030. These results indicate a comparatively better balance between precision and recall.

### 4.3 Overall Winner

**Support Vector Machine (SVM)** is the overall winner based on the F1 score.

It achieved:

- F1 Score: 0.4627
- AUC Score: 0.7913
- Precision: 0.3753
- Recall: 0.6030
- Accuracy: 83.61%
- MCC: 0.3867

Although KNN achieved the highest accuracy, SVM performed better in terms of F1 score and AUC. Therefore, SVM is considered the best-performing model for this classification task based on the F1-score criterion.

## 5. Streamlit Application

The project includes an interactive web application developed using Streamlit and deployed on Streamlit Community Cloud.

**Live Streamlit Application Link:**

https://ml-assignment-2-zeg5j8wxk8urzhodabvaf6.streamlit.app/

### Application Features

1. CSV dataset upload functionality.
2. Dropdown menu to select one of the five trained classification models.
3. Evaluation metrics display, including Accuracy, AUC, Precision, Recall, F1 Score and MCC.
4. Confusion matrix and classification report.
5. Evaluation using previously trained models without retraining on uploaded test data.

The uploaded test dataset must contain all 15 input features and the target column `y`.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- GitHub
- Streamlit Community Cloud
- Google Colab

## 7. Conclusion

This project demonstrates the implementation and evaluation of five machine learning classification algorithms using the Bank Marketing dataset.

The models were compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient.

Among the five classifiers, Support Vector Machine achieved the highest F1 score and AUC score, making it the best-performing model according to these evaluation criteria.

The trained models were saved using Joblib and integrated into an interactive Streamlit application, allowing users to evaluate different classifiers using uploaded test datasets without retraining the models.

This project demonstrates the complete machine learning workflow, including data preprocessing, model training, performance evaluation, model serialization, application development and deployment.
