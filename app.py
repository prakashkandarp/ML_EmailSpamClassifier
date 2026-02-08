import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

from config.feature_names import SPAMBASE_FEATURES


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)


# --------------------------------------------------
# Load feature names for preview
# --------------------------------------------------
feature_names = SPAMBASE_FEATURES + ["label"]

if len(feature_names) != 58:
    st.error(
        f"❌ Feature name mismatch: Expected 58 names, found {len(feature_names)}. "
    )
    st.stop()


# --------------------------------------------------
# Load sample data
# --------------------------------------------------
@st.cache_data
def load_sample_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    df = pd.read_csv(url, header=None)

    sample_df = df.groupby(df.iloc[:, -1], group_keys=False).apply(
        lambda x: x.sample(n=25, random_state=42)
    )

    return sample_df.sample(frac=1, random_state=42).reset_index(drop=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (Spambase format)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Please select a model",
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model_descriptions = {
    "Please select a model": "Please select a model.",
    "logistic_regression": "Linear model suitable for high-dimensional data.",
    "decision_tree": "Tree-based model that captures non-linear patterns.",
    "knn": "Distance-based classifier sensitive to feature scaling.",
    "naive_bayes": "Probabilistic model assuming feature independence.",
    "random_forest": "Ensemble of decision trees with strong generalization.",
    "xgboost": "Gradient boosting model with best overall performance."
}

st.sidebar.info(f"ℹ️ **Model Info:**\n\n{model_descriptions[model_name]}")

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("▶️ Run Evaluation", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Download Quick Test Data")

sample_df = load_sample_data()
csv_bytes = sample_df.to_csv(index=False, header=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download Sample Test CSV (50 rows)",
    data=csv_bytes,
    file_name="spambase_test_sample_50.csv",
    mime="text/csv"
)


# --------------------------------------------------
# Main Title
# --------------------------------------------------
st.title("📧 Spam Email Classification System")
st.markdown(
    "This application classifies emails as **Spam** or **Not Spam** "
    "using multiple machine learning models trained on the Spambase dataset."
)


# --------------------------------------------------
# Read uploaded data
# --------------------------------------------------
data = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)

        if data.shape[1] != 58:
            st.error(
                f"❌ Invalid file format: Expected 58 columns (57 features + 1 label), "
                f"but found {data.shape[1]} columns."
            )
            st.stop()

    except Exception:
        st.error("❌ Unable to read the uploaded file. Please upload a valid CSV.")
        st.stop()

st.markdown("---")
show_data = st.button("Show Uploaded Data")

# --------------------------------------------------
# Uploaded Data Preview
# --------------------------------------------------
if show_data:
    if data is None:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        preview_df = data.copy()
        preview_df.columns = feature_names

        st.subheader("📄 Uploaded Test Data Preview (first 20 rows)")
        st.dataframe(preview_df.head(20), use_container_width=True)

        st.caption(
            "**:red[NOTE:]** "
            "*Model evaluation is performed only after clicking **Run Evaluation**.*"
        )


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------
if run_clicked:
    if data is None:
        st.warning("⚠️ Please upload a test CSV file before running evaluation.")

    elif model_name == "Please select a model":
        st.warning("⚠️ Please select a machine learning model before running evaluation.")

    else:
        # Load model & scaler
        model = joblib.load(f"model/{model_name}.pkl")
        scaler = joblib.load("model/scaler.pkl")

        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]

        # Scaling where required
        if model_name in ["logistic_regression", "knn"]:
            X_input = scaler.transform(X)
        else:
            X_input = X

        # Predictions
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        st.subheader("📊 Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col1.metric("AUC", f"{auc:.3f}")

        col2.metric("Precision", f"{prec:.3f}")
        col2.metric("Recall", f"{rec:.3f}")

        col3.metric("F1 Score", f"{f1:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Not Spam", "Actual Spam"],
            columns=["Predicted Not Spam", "Predicted Spam"]
        )

        st.dataframe(cm_df, use_container_width=True)

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------
        st.subheader("📈 ROC Curve")

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend()

        st.pyplot(fig)

        # --------------------------------------------------
        # Feature Importance
        # --------------------------------------------------
        if model_name in ["random_forest", "xgboost"]:
            st.subheader("🌲 Feature Importance")

            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]

            top_features = [SPAMBASE_FEATURES[i] for i in top_idx]

            fig, ax = plt.subplots()
            ax.barh(range(len(top_idx)), importances[top_idx][::-1])
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels(top_features[::-1])
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 10 Important Spam Indicators")

            st.pyplot(fig)

            with st.expander("ℹ️ Feature Contribution Explanation"):
                st.markdown(
                    "- Higher bars indicate stronger influence on predictions.\n"
                    "- Word frequency features capture common spam keywords.\n"
                    "- Capital letter features reflect aggressive formatting patterns."
                )

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        with st.expander("📄 Detailed Classification Report", expanded=True):
            st.text(classification_report(y_true, y_pred))

        # --------------------------------------------------
        # Sample prediction confidence
        # --------------------------------------------------
        st.subheader("🔍 Sample Prediction Confidence")

        preview_pred = pd.DataFrame({
            "Actual Label": y_true.values[:10],
            "Predicted Label": y_pred[:10],
            "Spam Probability": np.round(y_prob[:10], 3)
        })

        st.dataframe(preview_pred, use_container_width=True)

else:
    st.info(
        "👈 Select a model, upload a test CSV file, and click **Run Evaluation** "
        "to view performance results."
    )
