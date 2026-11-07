
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing - Classification Models")
st.write(
    "Upload your test dataset to evaluate five pre-trained "
    "classification models. No model retraining is performed."
)

# Saved model locations
MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/support_vector_machine.joblib"
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# 1. Upload CSV
st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload test_data.csv to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "y" not in df.columns:
    st.error("Target column 'y' is missing.")
    st.stop()

if df.empty or df["y"].isna().any():
    st.error("The uploaded dataset is empty or contains missing target values.")
    st.stop()

if not set(df["y"].unique()).issubset({"yes", "no"}):
    st.error("Target column must contain only yes and no.")
    st.stop()

X_test = df.drop(columns=["y"])
y_test = df["y"]

st.subheader("Uploaded Dataset")
st.write("Dataset shape:", df.shape)
st.dataframe(df.head(), use_container_width=True)


# 2. Load and evaluate all five models
st.header("2. Model Evaluation")

all_results = {}
loaded_models = {}
errors = {}

for model_name, model_path in MODEL_FILES.items():

    if not os.path.exists(model_path):
        errors[model_name] = f"Missing model file: {model_path}"
        continue

    try:
        model = load_model(model_path)

        # Match the original training feature order
        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:
            missing = set(expected_features) - set(X_test.columns)

            if missing:
                raise ValueError(
                    f"Missing input features: {sorted(missing)}"
                )

            model_input = X_test[list(expected_features)]
        else:
            model_input = X_test

        predictions = model.predict(model_input)

        classifier = (
            model.named_steps["classifier"]
            if hasattr(model, "named_steps")
            else model
        )

        if (
            hasattr(classifier, "predict_proba")
            and getattr(classifier, "probability", True)
        ):
            scores = model.predict_proba(model_input)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(model_input)
        else:
            scores = None

        y_binary = (y_test == "yes").astype(int)

        metrics = {
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": (
                roc_auc_score(y_binary, scores)
                if scores is not None
                else float("nan")
            ),
            "Precision": precision_score(
                y_test, predictions,
                pos_label="yes", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions,
                pos_label="yes", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions,
                pos_label="yes", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        }

        all_results[model_name] = {
            "metrics": metrics,
            "predictions": predictions,
            "model": model
        }

        loaded_models[model_name] = model

    except Exception as e:
        errors[model_name] = str(e)


# Display comparison of all models
if all_results:

    st.subheader("Performance Comparison - All Five Models")

    comparison_df = pd.DataFrame([
        result["metrics"]
        for result in all_results.values()
    ])

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # 3. Model selection
    st.header("3. Select Model")

    selected_model = st.selectbox(
        "Choose a model to view detailed evaluation:",
        list(all_results.keys())
    )

    selected = all_results[selected_model]
    y_pred = selected["predictions"]

    st.subheader(f"Metrics - {selected_model}")

    metric_values = selected["metrics"]

    cols = st.columns(6)

    for col, metric in zip(
        cols,
        ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    ):
        value = metric_values[metric]

        col.metric(
            metric,
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )

    # 4. Confusion matrix
    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["no", "yes"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(selected_model)

    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=["no", "yes"],
        zero_division=0
    )

    st.code(report)

else:
    st.error("None of the five models could be evaluated.")

if errors:
    st.subheader("Model loading or evaluation errors")

    for name, error in errors.items():
        st.error(f"{name}: {error}")

st.success("Evaluation completed. No models were retrained.")
