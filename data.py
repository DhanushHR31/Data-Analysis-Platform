import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    precision_recall_curve
)

# Initialize session
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar navigation
st.set_page_config(page_title="Data App", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", [
    "Upload Dataset",
    "Preprocessing",
    "Clustering",
    "Classification",
    "Update Dataset",
    "Dashboard"
])

# ------------------ Upload Dataset ------------------
if page == "Upload Dataset":
    st.title("📤 Upload Dataset")
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("✅ Dataset Uploaded!")
        st.dataframe(st.session_state.df)

# ------------------ Preprocessing ------------------
elif page == "Preprocessing":
    st.title("🧹 Preprocessing")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("Dataset Overview:")
        st.dataframe(df)

        col = st.selectbox("Choose Column to Filter", df.columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            range_vals = st.slider("Select Range", min_val, max_val, (min_val, max_val))
            filtered_df = df[df[col].between(*range_vals)]
        else:
            unique_vals = df[col].unique().tolist()
            selected_vals = st.multiselect("Select values", unique_vals, default=unique_vals)
            filtered_df = df[df[col].isin(selected_vals)]

        st.write("Filtered Data:")
        st.dataframe(filtered_df)

        st.write("🔢 Value Counts")
        st.write(df[col].value_counts())
    else:
        st.warning("⚠️ Upload a dataset first.")

# ------------------ Clustering ------------------
elif page == "Clustering":
    st.title("🧩 Clustering")
    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) < 2:
            st.warning("⚠️ At least 2 numeric features required.")
        else:
            features = st.multiselect("Select Features", num_cols, default=num_cols[:2])
            if len(features) >= 2:
                X = StandardScaler().fit_transform(df[features])
                k = st.slider("Number of Clusters", 2, 10, 3)
                model = KMeans(n_clusters=k)
                labels = model.fit_predict(X)
                df["Cluster"] = labels
                st.session_state.df = df

                st.dataframe(df)

                st.subheader("Scatter Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=labels, palette="Set2", ax=ax)
                st.pyplot(fig)
    else:
        st.warning("⚠️ Upload a dataset first.")

# ------------------ Classification ------------------
elif page == "Classification":
    st.title("🧠 Classification with Evaluation Metrics")

    if st.session_state.df is not None:
        df = st.session_state.df.dropna()
        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target])

        if features and target:
            X = pd.get_dummies(df[features])
            y = df[target]

            if pd.api.types.is_numeric_dtype(y):
                y = pd.cut(y, bins=2, labels=["Low", "High"])

            model_choice = st.selectbox("Select Classifier", [
                "Random Forest", "Logistic Regression", "SVM", "KNN"
            ])

            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "SVM":
                model = SVC(probability=True)
            elif model_choice == "KNN":
                model = KNeighborsClassifier()

            model.fit(X, y)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            st.subheader("📊 Metrics")

            # Accuracy
            acc = accuracy_score(y, y_pred)
            st.write(f"**Accuracy:** {acc:.2f}")

            # Classification Report
            st.text("Classification Report")
            st.text(classification_report(y, y_pred))

            # Confusion Matrix
            st.write("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

            # Precision, Recall, F1
            precision = precision_score(y, y_pred, average="weighted")
            recall = recall_score(y, y_pred, average="weighted")
            f1 = f1_score(y, y_pred, average="weighted")
            st.write(f"**Precision (Weighted):** {precision:.2f}")
            st.write(f"**Recall (Weighted):** {recall:.2f}")
            st.write(f"**F1-Score (Weighted):** {f1:.2f}")

            # ROC-AUC Score
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y, y_prob)
                    st.write(f"**ROC-AUC Score:** {roc_auc:.2f}")
                except Exception as e:
                    st.warning(f"ROC-AUC error: {e}")

            # Log Loss
            if y_prob is not None:
                try:
                    logloss = log_loss(y, model.predict_proba(X))
                    st.write(f"**Log Loss:** {logloss:.2f}")
                except Exception as e:
                    st.warning(f"Log loss error: {e}")

            # Precision-Recall Curve
            if y_prob is not None:
                try:
                    precision_vals, recall_vals, _ = precision_recall_curve(y.map({label: i for i, label in enumerate(np.unique(y))}), y_prob)
                    fig_pr, ax_pr = plt.subplots()
                    ax_pr.plot(recall_vals, precision_vals, label="PR Curve")
                    ax_pr.set_xlabel("Recall")
                    ax_pr.set_ylabel("Precision")
                    ax_pr.set_title("Precision-Recall Curve")
                    ax_pr.legend()
                    st.pyplot(fig_pr)
                except Exception as e:
                    st.warning(f"PR curve error: {e}")
    else:
        st.warning("⚠️ Please upload a dataset first.")

# ------------------ Update Dataset ------------------
elif page == "Update Dataset":
    st.title("🔄 Update or Edit Dataset")
    upload = st.file_uploader("Upload New CSV", type=["csv"], key="update")
    if upload:
        st.session_state.df = pd.read_csv(upload)
        st.success("✅ Dataset Replaced")
        st.dataframe(st.session_state.df)

    if st.session_state.df is not None:
        edited_df = st.data_editor(st.session_state.df, num_rows="dynamic")
        if st.button("💾 Save Changes"):
            st.session_state.df = edited_df
            st.success("✅ Dataset Saved")
            st.dataframe(edited_df)
    else:
        st.warning("⚠️ Upload a dataset first.")
        
# ------------------ Dashboard ------------------
elif page == "Dashboard":
    st.title("📈 Dashboard")

    if st.session_state.df is not None:
        df = st.session_state.df
        column = st.selectbox("Select Column for Visualization", df.columns)

        # Drop NaN for accurate plots
        column_data = df[column].dropna()

        st.subheader("Bar Chart")
        if column_data.dtype == object or column_data.dtype == 'category':
            st.bar_chart(column_data.value_counts())
        else:
            st.bar_chart(column_data)

        st.subheader("Pie Chart")
        if column_data.dtype == object or column_data.dtype == 'category':
            pie_data = column_data.value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)
        else:
            st.info("Pie chart is only applicable to categorical columns.")

        st.subheader("Histogram")
        if pd.api.types.is_numeric_dtype(column_data):
            fig, ax = plt.subplots()
            sns.histplot(column_data, kde=True, bins=20, ax=ax, color='skyblue')
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("Histogram is only applicable to numeric columns.")
    else:
        st.warning("⚠️ Upload a dataset first.")

