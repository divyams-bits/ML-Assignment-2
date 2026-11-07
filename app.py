import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="📊", layout="wide")
st.title("Bank Marketing – Classification Models")
st.write(
    "Upload the held-out test CSV, choose one of the five pre-trained classifiers, "
    "and evaluate it. The app loads saved pipelines; it does not retrain models."
)

MODEL_DIR = "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload test_data.csv to begin. It must include the target column 'y'.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if "y" not in data.columns:
    st.error("The uploaded CSV must contain the target column 'y'.")
    st.stop()
if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X_test = data.drop(columns=["y"])
y_test = data["y"].astype(str)
if not set(y_test.unique()).issubset({"yes", "no"}):
    st.error("Target column 'y' must contain only 'yes' and 'no'.")
    st.stop()

st.subheader("Uploaded dataset")
st.write(f"Rows: **{len(data):,}** · Input columns: **{len(X_test.columns)}**")
st.dataframe(data.head(), use_container_width=True)

model_name = st.selectbox("Select a classification model", list(MODEL_FILES))
model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
if not os.path.isfile(model_path):
    st.error(f"Model file not found: {model_path}. Check that the models folder is committed to GitHub.")
    st.stop()

try:
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    if hasattr(classifier, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    y_true_binary = (y_test == "yes").astype(int)
    y_pred_binary = (pd.Series(y_pred, index=y_test.index) == "yes").astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_true_binary, y_score) if y_score is not None else float("nan"),
        "Precision": precision_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label="yes", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    st.subheader(f"Evaluation results – {model_name}")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        st.code(classification_report(y_test, y_pred, labels=["no", "yes"], zero_division=0))

except Exception as exc:
    st.error("The saved model could not evaluate this CSV.")
    st.exception(exc)
    st.caption(
        "Saved scikit-learn models must be loaded with a compatible scikit-learn version "
        "and the same input feature schema used during training."
    )

with st.expander("View model comparison from the training experiment"):
    if os.path.exists("model_comparison.csv"):
        comparison = pd.read_csv("model_comparison.csv")
        st.dataframe(comparison, use_container_width=True)
