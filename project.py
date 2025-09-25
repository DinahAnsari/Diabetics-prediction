import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def replace_zeros_with_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        df[c] = df[c].replace(0, np.nan)
    return df


def impute_with_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def prepare_features(df: pd.DataFrame):
    target_col = "Outcome"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].astype(int).values
    return X, y, feature_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, sensitivity, specificity, cm


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"]) 
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)


def build_model(
    model_name: str,
    svm_c: float,
    svm_kernel: str,
    knn_k: int,
    knn_weights: str,
    ada_estimators: int,
    ada_lr: float,
):
    if model_name == "SVM":
        return SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
    if model_name == "AdaBoost":
        base = DecisionTreeClassifier(max_depth=1, random_state=42)
        return AdaBoostClassifier(estimator=base, n_estimators=ada_estimators, learning_rate=ada_lr, random_state=42)
    raise ValueError("Unknown model")


def main():
    st.set_page_config(page_title="Diabetes Prediction (PIMA)", layout="wide")
    st.title("Diabetes Prediction on PIMA Dataset")
    st.caption("SVM · KNN · AdaBoost | Metrics: Accuracy, Sensitivity, Specificity")

    with st.sidebar:
        st.header("Data & Preprocess")
        data_source = st.radio("Dataset source", ["Included data.csv", "Upload CSV"], index=0)
        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload PIMA-like CSV", type=["csv"]) 

        st.markdown("---")
        st.header("Model")
        model_name = st.selectbox("Choose model", ["SVM", "KNN", "AdaBoost"], index=2)

        svm_c = st.slider("SVM C", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        svm_kernel = st.selectbox("SVM kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)

        knn_k = st.slider("KNN k", min_value=1, max_value=25, value=7, step=1)
        knn_weights = st.selectbox("KNN weights", ["uniform", "distance"], index=0)

        ada_estimators = st.slider("AdaBoost estimators", min_value=10, max_value=500, value=150, step=10)
        ada_lr = st.slider("AdaBoost learning rate", min_value=0.01, max_value=2.0, value=0.5, step=0.01)

        st.markdown("---")
        test_size = st.slider("Test size (%)", min_value=10, max_value=50, value=25, step=5)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        scale_features = st.checkbox("Standardize features (recommended for SVM/KNN)", value=True)
        run_button = st.button("Train & Evaluate", type="primary")

    if data_source == "Included data.csv":
        df = load_data("data.csv")
    else:
        if uploaded is None:
            st.info("Upload a CSV to proceed or switch to the included dataset.")
            return
        df = pd.read_csv(uploaded)

    expected_columns = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    if not set(expected_columns).issubset(set(df.columns)):
        st.error("CSV must contain standard PIMA columns including 'Outcome'.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Shape**")
        st.write(df.shape)
        st.markdown("**Class balance**")
        st.write(df["Outcome"].value_counts())
    with right:
        st.markdown("**Summary stats**")
        st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlations")
    corr = df.drop(columns=["Outcome"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig, clear_figure=True)

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = replace_zeros_with_nan(df, zero_as_missing)
    df_clean = impute_with_median(df_clean)

    X, y, feature_cols = prepare_features(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100.0, stratify=y, random_state=random_state
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = build_model(
        model_name=model_name,
        svm_c=svm_c,
        svm_kernel=svm_kernel,
        knn_k=knn_k,
        knn_weights=knn_weights,
        ada_estimators=ada_estimators,
        ada_lr=ada_lr,
    )

    if run_button:
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy, sensitivity, specificity, cm = compute_metrics(y_test, y_pred)

            st.markdown("---")
            st.subheader("Performance Metrics")
            met1, met2, met3 = st.columns(3)
            met1.metric("Accuracy", f"{accuracy:.3f}")
            met2.metric("Sensitivity (TPR)", f"{sensitivity:.3f}")
            met3.metric("Specificity (TNR)", f"{specificity:.3f}")

            st.subheader("Confusion Matrix")
            plot_confusion_matrix(cm)

            proba_supported = hasattr(model, "predict_proba")
            if proba_supported:
                y_score = model.predict_proba(X_test)[:, 1]
                st.subheader("ROC Curve")
                plot_roc(y_test, y_score)
            else:
                st.info("ROC requires probability estimates; current model settings do not provide them.")

            if model_name == "AdaBoost":
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    st.subheader("Feature Importance (AdaBoost)")
                    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
                    imp_df = imp_df.sort_values("importance", ascending=False)
                    fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.0))
                    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax_imp)
                    ax_imp.set_title("AdaBoost Feature Importance")
                    st.pyplot(fig_imp, clear_figure=True)

            # save trained artifacts for manual prediction
            st.session_state["trained_model"] = model
            st.session_state["trained_scaler"] = scaler
            st.session_state["trained_features"] = feature_cols
            st.session_state["feature_bounds"] = df_clean[feature_cols].describe()

            # --- Model Comparison (SVM, KNN, AdaBoost) ---
            st.subheader("Model Comparison: Accuracy, Sensitivity, Specificity")
            comparison_results = []

            models_to_compare = [
                ("SVM", SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)),
                ("KNN", KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)),
                ("AdaBoost", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                                                n_estimators=ada_estimators,
                                                learning_rate=ada_lr,
                                                random_state=42)),
            ]

            for name, clf in models_to_compare:
                clf.fit(X_train, y_train)
                y_pred_cmp = clf.predict(X_test)
                acc, sen, spe, _ = compute_metrics(y_test, y_pred_cmp)
                comparison_results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Sensitivity": sen,
                    "Specificity": spe,
                })

            cmp_df = pd.DataFrame(comparison_results)
            cmp_df = cmp_df.set_index("Model")["Accuracy Sensitivity Specificity".split()]

            fig_cmp, ax_cmp = plt.subplots(figsize=(7.5, 4.2))
            cmp_df.plot(kind="bar", ax=ax_cmp, rot=0, ylim=(0, 1))
            ax_cmp.set_ylabel("Score")
            ax_cmp.set_title("Model Comparison on Test Set")
            ax_cmp.legend(loc="lower right")
            st.pyplot(fig_cmp, clear_figure=True)

    st.markdown("---")
    st.subheader("Manual Prediction")
    if "trained_model" not in st.session_state:
        st.info("Train a model first, then use this form to predict a single case.")
    else:
        trained_model = st.session_state["trained_model"]
        trained_scaler = st.session_state["trained_scaler"]
        trained_features = st.session_state["trained_features"]
        bounds = st.session_state["feature_bounds"]

        # Build inputs with reasonable defaults and ranges from data
        cols = st.columns(3)
        user_values = []
        for idx, feat in enumerate(trained_features):
            col = cols[idx % 3]
            col_min = float(bounds.loc["min", feat]) if feat in bounds.columns else 0.0
            col_max = float(bounds.loc["max", feat]) if feat in bounds.columns else 1.0
            default = float(bounds.loc["50%", feat]) if feat in bounds.columns else (col_min + col_max) / 2.0

            with col:
                if feat in ["Pregnancies", "SkinThickness", "Insulin", "Age", "BloodPressure", "Glucose"]:
                    val = st.number_input(feat, value=default, min_value=col_min, max_value=col_max, step=1.0)
                else:
                    val = st.number_input(feat, value=default, min_value=col_min, max_value=col_max, step=0.1)
            user_values.append(val)

        predict_btn = st.button("Predict Outcome")
        if predict_btn:
            x = np.array(user_values, dtype=float).reshape(1, -1)
            if trained_scaler is not None:
                x = trained_scaler.transform(x)
            pred = trained_model.predict(x)[0]
            proba = None
            if hasattr(trained_model, "predict_proba"):
                proba = trained_model.predict_proba(x)[0, 1]

            st.success(f"Predicted Outcome: {int(pred)} (0 = Non-diabetic, 1 = Diabetic)")
            if proba is not None:
                st.write(f"Probability of diabetes (class 1): {proba:.3f}")


if __name__ == "__main__":
    main()


