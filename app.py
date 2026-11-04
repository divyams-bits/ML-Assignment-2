
import streamlit as st
import pandas as pd
import joblib
import os

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

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing - Classification Models")
st.write(
    "Upload test data, select a trained classification model, "
    "and evaluate its performance."
)

st.info(
    "This application uses five pre-trained models. "
    "It does not retrain models when you upload a CSV file."
)

# --------------------------------------------------
# 2. MODEL FILE PATHS
# --------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/support_vector_machine.joblib"
}

# --------------------------------------------------
# 3. LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# --------------------------------------------------
# 4. MODEL SELECTION
# --------------------------------------------------

selected_model = st.selectbox(
    "Select a Classification Model",
    list(MODEL_PATHS.keys())
)

# --------------------------------------------------
# 5. CSV FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

st.caption(
    "The CSV must contain the 15 input features and the target column 'y'."
)

# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", data.shape)
        st.dataframe(data.head())

        # Check target column
        if "y" not in data.columns:
            st.error("The uploaded CSV must contain a target column named 'y'.")
            st.stop()

        # Separate input features and actual target
        X_test = data.drop(columns=["y"])
        y_test = data["y"].astype(str).str.lower()

        # Load selected saved model
        model_path = MODEL_PATHS[selected_model]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = load_model(model_path)

        # Check and arrange feature columns
        expected_features = list(model.feature_names_in_)

        if set(X_test.columns) != set(expected_features):
            missing = set(expected_features) - set(X_test.columns)
            extra = set(X_test.columns) - set(expected_features)

            st.error(
                f"Feature columns do not match the training data. "
                f"Missing columns: {sorted(missing)}. "
                f"Extra columns: {sorted(extra)}."
            )
            st.stop()

        X_test = X_test[expected_features]

        # Make predictions using the saved model
        y_pred = model.predict(X_test)

        # Find probability of the positive class ('yes')
        classes = list(model.classes_)

        if "yes" not in classes:
            st.error("The selected model does not contain the positive class 'yes'.")
            st.stop()

        positive_index = classes.index("yes")
        y_prob = model.predict_proba(X_test)[:, positive_index]

        # Convert actual and predicted labels to lowercase
        y_pred = pd.Series(y_pred).astype(str).str.lower()

        if not set(y_test.unique()).issubset({"yes", "no"}):
            st.error("The target column must contain only 'yes' and 'no'.")
            st.stop()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            (y_test == "yes").astype(int),
            y_prob
        )

        precision = precision_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # 7. DISPLAY METRICS
        # --------------------------------------------------

        st.subheader("Model Evaluation Results")
        st.write("Selected Model:", selected_model)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 8. CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["no", "yes"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # 9. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=["no", "yes"],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())

        st.success("Model evaluation completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
