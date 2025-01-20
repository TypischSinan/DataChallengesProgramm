import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sidebar import configure_sidebar
import numpy as np
from imblearn.over_sampling import SMOTE


@st.cache_data
def load_data():
    data = pd.read_parquet('./data/dataset.parquet')

    # Entferne unerwünschte Spalten direkt nach dem Laden
    excluded_columns = ['Filename', 'Patient Number', 'Participant ID_x',
                        'Adult BMI', 'Child Weight', 'Child Height', 'Participant ID_y']
    data = data.drop(
        columns=[col for col in excluded_columns if col in data.columns])

    return data


data = load_data()

# Sidebar configuration
apply_imputation, imputation_method = configure_sidebar()
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Target column selection
st.title("Classification Analysis")
st.subheader("Select Target Column")

target_column = st.selectbox(
    "Choose the target column for classification:", options=data.columns)

# Features selection
st.subheader("Select Features")
manuel_or_all = st.checkbox("Use all columns as features", value=False)
if manuel_or_all:
    feature_columns = numeric_data.columns.tolist()
else:
    feature_columns = st.multiselect(
        "Select columns for features:",
        options=numeric_data.columns.tolist(),
        default=[col for col in numeric_data.columns if col != target_column]
    )

if target_column not in data.columns or not feature_columns:
    st.error("Please select valid target and feature columns.")
else:
    # Data preprocessing
    if apply_imputation:
        if imputation_method == "Mean":
            imputer = SimpleImputer(strategy="mean")
        elif imputation_method == "Median":
            imputer = SimpleImputer(strategy="median")
        elif imputation_method == "Most Frequent":
            imputer = SimpleImputer(strategy="most_frequent")

        data[numeric_data.columns] = imputer.fit_transform(numeric_data)
    else:
        # If no imputation is applied, drop rows with missing values
        data = data.dropna()

    # Encoding target variable if categorical
    if data[target_column].dtype == 'object':
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
    else:
        label_encoder = None

    # Scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, data[target_column], test_size=0.2, random_state=42
    )

    # Choose Sampling Method
    st.subheader("Choose Sampling Method")
    sampling_method = st.radio(
        "Select a sampling method to address data imbalance:",
        ["None", "Undersampling", "Oversampling", "SMOTE"]
    )

    if sampling_method == "Undersampling":
        undersampler = RandomUnderSampler(random_state=32)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    elif sampling_method == "Oversampling":
        oversampler = RandomOverSampler(random_state=32)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    elif sampling_method == "SMOTE":
        # Adjust the number of neighbors for SMOTE based on the number of samples in the minority class
        minority_class_count = min(np.bincount(y_train))
        n_neighbors = min(5, minority_class_count - 1)
        oversampler = SMOTE(random_state=32, k_neighbors=n_neighbors)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # Classification algorithm selection
    st.subheader("Select Classification Algorithm")
    algorithm = st.radio(
        "Choose a classification algorithm:",
        ["K-Nearest Neighbors"]
    )

    if st.button("Run Classification"):
        if algorithm == "K-Nearest Neighbors":
            classifier = KNeighborsClassifier(n_neighbors=5)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.2f}")

        # Extract Precision and F1-Score
        if label_encoder:
            target_names = [str(i) for i in label_encoder.classes_]
        else:
            target_names = [str(i) for i in np.unique(data[target_column])]

        class_report = classification_report(
            y_test, y_pred, target_names=target_names,
            output_dict=True, zero_division=1
        )

        # Summarize and display metrics
        precision = class_report["weighted avg"]["precision"]
        f1_score = class_report["weighted avg"]["f1-score"]

        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**F1-Score:** {f1_score:.2f}")

        # t-SNE Visualization
        st.subheader("2D-Visualization of Classification using t-SNE")
        tsne_results = TSNE(n_components=2, perplexity=30, learning_rate=200,
                            random_state=42).fit_transform(X_train)

        tsne_df = pd.DataFrame(tsne_results, columns=["TSNE-1", "TSNE-2"])
        tsne_df["Class"] = y_train

        # Map numeric class labels to class names
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = np.unique(data[target_column])

        class_mapping = {i: str(class_names[i])
                         for i in range(len(class_names))}
        tsne_df["Class_Name"] = tsne_df["Class"].map(class_mapping)

        # Plot the t-SNE results
        fig_tsne = px.scatter(
            tsne_df, x="TSNE-1", y="TSNE-2", color="Class_Name",
            title="t-SNE Visualization after Sampling",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig_tsne)
