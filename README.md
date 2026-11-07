# Bank Marketing – Machine Learning Classification Assignment

## a. Problem Statement

The objective of this assignment is to implement and compare five different Machine Learning classification algorithms using the Bank Marketing dataset.

The problem involves predicting whether a customer will subscribe to a term deposit (`yes` or `no`) based on customer information and previous marketing campaign details.

The following five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

The performance of all five models is evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The objective is to identify the best-performing classification model based on the evaluation results.

## b. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains information about customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

The dataset is used to predict whether a customer will subscribe to a term deposit based on various customer characteristics and campaign-related information.

**Dataset Details:**

| Description | Details |
|---|---|
| Total Instances | 45,211 |
| Input Features Used | 15 |
| Numerical Features | 6 |
| Categorical Features | 9 |
| Target Variable | y |
| Classification Type | Binary Classification |
| Target Classes | yes / no |

**Numerical Features:**
- age
- balance
- day
- campaign
- pdays
- previous

**Categorical Features:**
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- poutcome

**Target Variable:**

The target variable `y` indicates whether a customer subscribed to a term deposit.

- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

The dataset contains 39,922 negative instances and 5,289 positive instances, indicating an imbalance between the two target classes.

The dataset was divided into training and testing sets using an 80:20 split.

| Dataset Split | Number of Records |
|---|---:|
| Training Dataset | 36,168 |
| Testing Dataset | 9,043 |
| Total | 45,211 |

## c. GitHub Repository Link

**GitHub Repository:**

https://github.com/divyams-bits/ML-Assignment-2

The GitHub repository contains the project source code, trained machine learning models, requirements.txt, test dataset, model comparison results, and README documentation.

## d. Models Used

Five Machine Learning classification algorithms were implemented using the Bank Marketing dataset.

All five models were evaluated using the same testing dataset and the following six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8309 | 0.6132 | 0.2982 | 0.3289 | 0.3128 | 0.2170 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | **0.4123** | 0.3271 |
| Support Vector Machine | 0.8933 | 0.7184 | 0.6555 | 0.1853 | 0.2889 | 0.3098 |

### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 89.33% and the highest AUC score of 0.7717. It also achieved high precision of 0.6632. However, its recall was low at 0.1786, resulting in an F1 score of 0.2815. |
| Decision Tree | Achieved an accuracy of 83.09% and an F1 score of 0.3128. Its recall was 0.3289, while its AUC score was 0.6132. It provided a reasonable balance between precision and recall compared with some of the other models. |
| KNN | Achieved an accuracy of 88.74% and precision of 0.5529. However, its recall was relatively low at 0.1975, resulting in an F1 score of 0.2911. |
| Naive Bayes | Achieved the highest F1 score of 0.4123 among all five models. It obtained a precision of 0.3708 and recall of 0.4641, demonstrating the best balance between precision and recall among the evaluated models. |
| Support Vector Machine | Achieved an accuracy of 89.33% and precision of 0.6555. Its AUC score was 0.7184. However, its recall was low at 0.1853, resulting in an F1 score of 0.2889. |

### Overall Winner for the Dataset

**Naive Bayes Classifier**

Naive Bayes is selected as the overall winner based on the F1 Score.

It achieved the highest F1 score of **0.4123** among all five classification models. It also demonstrated a better balance between precision (0.3708) and recall (0.4641).

Although Logistic Regression achieved the highest AUC score of 0.7717 and Logistic Regression and Support Vector Machine achieved the highest accuracy of 89.33%, Naive Bayes performed best in terms of F1 score.

Therefore, Naive Bayes is considered the best-performing classification model for the Bank Marketing dataset based on the F1 Score.
