# Bank Marketing Classification – ML Assignment 2

## a. Problem statement
The objective is to predict whether a bank client will subscribe to a term deposit (`y = yes` or `no`) using the UCI Bank Marketing dataset. Five classification algorithms are trained and evaluated on the same held-out test set. A Streamlit application loads the saved models and evaluates an uploaded test CSV without retraining.

## b. Dataset description
- **Dataset:** UCI Bank Marketing (`bank-full.csv`)
- **Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Instances:** 45,211
- **Original columns:** 17 (16 input columns and target `y`)
- **Target:** `y` (`yes` / `no`), binary classification
- **Test file:** `test_data.csv` (9,043 rows; 16 input columns plus target)
- The training notebook's preprocessing definition selects the 15 predictors excluding `duration`; the test CSV retains the `duration` column. The saved sklearn pipeline applies its configured preprocessing to the input dataframe.

## c. GitHub Repository Link
https://github.com/divyams-bits/ML-Assignment-2

The repository includes:
- `app.py` – Streamlit evaluation interface
- `requirements.txt` – deployment dependencies
- `ML2.ipynb` – Jupyter notebook containing the experiment code
- `test_data.csv` – held-out test data
- `model_comparison.csv` – recorded evaluation metrics
- `models/` – five serialized trained pipelines

## d. Models used and evaluation

The following values are copied from the supplied `model_comparison.csv`. Metrics are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8309 | 0.6132 | 0.2982 | 0.3289 | 0.3128 | 0.2170 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine (SVC) | 0.8933 | 0.7184 | 0.6555 | 0.1853 | 0.2889 | 0.3098 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy is tied with SVC (0.8933) and it has the highest AUC (0.7717). Its recall (0.1786) and F1 (0.2815) for the positive class are relatively low. |
| Decision Tree | It identifies more positive cases than Logistic Regression, KNN and SVC (recall 0.3289), but has the lowest AUC (0.6132), accuracy (0.8309) and MCC (0.2170). |
| KNN | Accuracy is 0.8874, but positive-class recall is low (0.1975), resulting in an F1 score of 0.2911. |
| Naive Bayes | It achieves the highest recall (0.4641), F1 score (0.4123) and MCC (0.3271), making it the strongest model on these positive-class balance metrics in this experiment. |
| Support Vector Machine | SVC ties for the highest accuracy (0.8933) and has precision 0.6555, but recall (0.1853) and F1 (0.2889) are lower than Naive Bayes. |

**Overall winner:** Naive Bayes, when selecting by F1 score (0.4123), which is the chosen positive-class-focused measure for this imbalanced dataset. Logistic Regression has the highest AUC (0.7717), so the preferred model depends on the evaluation objective.

## Streamlit application
The app provides:
1. CSV test-data upload.
2. A dropdown to select one of the five saved models.
3. Accuracy, AUC, precision, recall, F1 and MCC.
4. Confusion matrix and classification report.
5. A comparison table from `model_comparison.csv`.

The application loads pre-trained `.joblib` pipelines and does not retrain them on uploaded test data.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app.

## Reproducibility note
The notebook contains the experiment workflow. The saved model files were serialized with scikit-learn 1.6.1; therefore, the deployment dependencies pin scikit-learn 1.6.1. Python and package versions should remain compatible with the environment used to serialize the models.
