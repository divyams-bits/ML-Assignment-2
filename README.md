# Machine Learning Assignment – 2
## Bank Marketing Classification

### a. Problem Statement

The objective of this project is to develop and evaluate five machine learning classification models to predict whether a customer will subscribe to a term deposit based on customer demographic information and previous marketing campaign interactions.

The project implements five classification algorithms using the same dataset. Their performance is evaluated using six evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to allow users to upload test datasets, select a trained classification model and view its evaluation results without retraining the models.

### b. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains information about customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

The target variable is `y`, which indicates whether a customer subscribed to a term deposit (`yes` or `no`).

**Dataset Details**

| Attribute | Description |
|---|---|
| Total instances | 45,211 |
| Original features/columns | 17 |
| Input features used | 15 |
| Numerical features | 6 |
| Categorical features | 9 |
| Target variable | y |
| Problem type | Binary Classification |
| Training instances | 36,168 |
| Testing instances | 9,043 |

**Data Preprocessing**

- Checked the dataset for missing values.
- Separated input features and target variable.
- Excluded the `duration` feature to avoid data leakage.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Split the dataset into 80% training and 20% testing data.
- Used stratified sampling to preserve the target class distribution.

### c. GitHub Repository Link

**GitHub Repository:**

https://github.com/divyams-bits/ML-Assignment-2

The repository contains the complete source code, trained classification models, requirements file, test dataset and project documentation.

**Repository Structure**

```text
ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

### d. Models Used and Evaluation Metrics

Five machine learning classification algorithms were implemented using the Bank Marketing dataset.

The models were trained using the training dataset and evaluated using the same testing dataset.

**Evaluation Metrics**

1. Accuracy – Measures the proportion of correctly classified observations.
2. AUC Score – Measures the model's ability to distinguish between the two classes.
3. Precision – Measures the proportion of correctly predicted positive observations among all predicted positives.
4. Recall – Measures the proportion of actual positive observations correctly identified.
5. F1 Score – Harmonic mean of precision and recall.
6. Matthews Correlation Coefficient (MCC) – Measures the quality of binary classifications using all four confusion matrix categories.

#### 1. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.2853 | 0.3732 |
| Decision Tree | 0.8268 | 0.7465 | 0.3490 | 0.5548 | 0.3450 | 0.4285 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2833 | 0.2911 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.3271 | 0.4123 |
| Support Vector Machine | 0.8361 | 0.7913 | 0.3753 | 0.6030 | 0.4627 | 0.3867 |

#### 2. Observations on Model Performance

**Logistic Regression**

- Achieved an accuracy of 75.48%.
- Obtained a recall of 0.6238, indicating that it identified a relatively high proportion of actual subscribers.
- However, its precision and F1 score were comparatively low.

**Decision Tree**

- Achieved an accuracy of 82.68%.
- Obtained an MCC score of 0.4285, the highest among the five models.
- Provided a reasonable balance between precision and recall.

**K-Nearest Neighbors (KNN)**

- Achieved the highest accuracy of 88.74%.
- Obtained the highest precision of 0.5529.
- However, its recall was only 0.1975, indicating that it missed many actual subscribers.
- Its F1 score was consequently low.

**Naive Bayes**

- Achieved an accuracy of 84.52%.
- Obtained an AUC score of 0.7514.
- Provided moderate precision and recall, but its F1 score was lower than that of the Decision Tree and SVM models.

**Support Vector Machine (SVM)**

- Achieved the highest F1 score of 0.4627.
- Obtained the highest AUC score of 0.7913.
- Achieved a recall of 0.6030.
- Demonstrated a comparatively better balance between precision and recall.

#### 3. Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall winner based on the F1 Score.

Its evaluation results are:

| Metric | Score |
|---|---:|
| Accuracy | 0.8361 |
| AUC | 0.7913 |
| Precision | 0.3753 |
| Recall | 0.6030 |
| F1 Score | 0.4627 |
| MCC | 0.3867 |

Although KNN achieved the highest accuracy, SVM achieved the highest F1 and AUC scores. Therefore, SVM is selected as the best-performing model based on the F1-score criterion.

### Streamlit Application

**Live Streamlit Application Link:**

https://ml-assignment-2-zeg5j8wxk8urzhodabvaf6.streamlit.app/

The Streamlit application provides the following features:

1. CSV test dataset upload functionality.
2. Dropdown menu to select one of the five trained classification models.
3. Display of Accuracy, AUC, Precision, Recall, F1 Score and MCC.
4. Confusion matrix and classification report.
5. Evaluation using saved trained models without retraining on uploaded test data.

The uploaded CSV must contain the 15 required input features and the target column `y`.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- GitHub
- Google Colab

### Conclusion

This project demonstrates an end-to-end machine learning classification workflow, including dataset selection, preprocessing, model implementation, evaluation, model saving, Streamlit application development and deployment.

Among the five implemented classification algorithms, Support Vector Machine achieved the highest F1 score and AUC score. Hence, it was selected as the best-performing model based on the F1-score evaluation criterion.
