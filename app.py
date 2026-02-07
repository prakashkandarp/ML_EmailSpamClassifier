import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

@st.cache_data
def load_sample_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    df = pd.read_csv(url, header=None)

    sample_df = df.groupby(df.iloc[:, -1], group_keys=False).apply(
        lambda x: x.sample(n=25, random_state=42)
    )

    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return sample_df


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model_descriptions = {
    "logistic_regression": "Linear model suitable for high-dimensional data.",
    "decision_tree": "Tree-based model that captures non-linear patterns.",
    "knn": "Distance-based classifier sensitive to feature scaling.",
    "naive_bayes": "Probabilistic model assuming feature independence.",
    "random_forest": "Ensemble of decision trees with strong generalization.",
    "xgboost": "Gradient boosting model with best overall performance."
}

st.sidebar.info(f"ℹ️ **Model Info:**\n\n{model_descriptions[model_name]}")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (Spambase format)",
    type=["csv"]
)

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
# Load model & scaler
# --------------------------------------------------
model = joblib.load(f"model/{model_name}.pkl")
scaler = joblib.load("model/scaler.pkl")

# --------------------------------------------------
# When file is uploaded
# --------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)

    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]

    # Scaling only where required
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
    col1.metric("AUC Score", f"{auc:.3f}")

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
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # Feature Importance (Tree-based models only)
    # --------------------------------------------------
    if model_name in ["random_forest", "xgboost"]:
        st.subheader("🌲 Feature Importance")

        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[-10:][::-1]

        fig, ax = plt.subplots()
        ax.barh(
            range(len(feature_indices)),
            importances[feature_indices][::-1]
        )
        ax.set_yticks(range(len(feature_indices)))
        ax.set_yticklabels([f"Feature {i}" for i in feature_indices][::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Important Features")

        st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    with st.expander("📄 Detailed Classification Report"):
        st.text(classification_report(y_true, y_pred))

    # --------------------------------------------------
    # Spam probability preview
    # --------------------------------------------------
    st.subheader("🔍 Sample Prediction Confidence")

    preview_df = pd.DataFrame({
        "Actual Label": y_true.values[:10],
        "Predicted Label": y_pred[:10],
        "Spam Probability": np.round(y_prob[:10], 3)
    })

    st.dataframe(preview_df, use_container_width=True)

else:
    st.info("👈 Upload a test CSV file from the sidebar to begin evaluation.")
