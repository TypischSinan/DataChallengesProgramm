import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st
from sidebar import configure_sidebar


@st.cache_data
def load_data():
    data = pd.read_parquet('./data/dataset.parquet')
    return data


def colored_text(text, color):
    return f"<span style='color:{color}'>{text}</span>"


data = load_data()
apply_imputation, imputation_method = configure_sidebar()
numeric_data = data.select_dtypes(include=['float64', 'int64'])
columns_to_drop = ["Adult BMI", "Child Weight", "Child Height", "Age"]
numeric_data = numeric_data.drop(columns=columns_to_drop, errors='ignore')

st.title("Clustering Analysis")
st.subheader("Clustering Settings")

clustering_algorithm = st.radio(
    "Choose a clustering algorithm",
    ["KMeans", "DBSCAN", "OPTICS"]
)

manuel_or_all = st.checkbox("All columns for clustering", value=False)
if manuel_or_all:
    columns_for_clustering = numeric_data.columns.tolist()
else:
    columns_for_clustering = st.multiselect(
        "Select columns for clustering:",
        options=numeric_data.columns.tolist(),
        default=["Patient Number"]
    )

columns_for_clustering_df = numeric_data[columns_for_clustering]

if apply_imputation:
    if imputation_method == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation_method == "Median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "Most Frequent":
        imputer = SimpleImputer(strategy="most_frequent")

    imputed_data = imputer.fit_transform(columns_for_clustering_df)
    columns_for_clustering_df = pd.DataFrame(
        imputed_data, columns=columns_for_clustering_df.columns)

columns_for_clustering_df = columns_for_clustering_df.select_dtypes(include=[
                                                                    'float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(columns_for_clustering_df)

if clustering_algorithm == "KMeans":
    n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

if clustering_algorithm == "DBSCAN":
    eps = st.slider("Eps (Neighborhood radius)", 0.1, 5.0, 0.5)
    min_samples = st.slider("Minimum samples", 1, 20, 5)

if clustering_algorithm == "OPTICS":
    min_samples = st.slider("Minimum samples", 1, 20, 5)

dimensionality_reduction = st.checkbox("Reduce dimensions (PCA)")
if dimensionality_reduction:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
else:
    reduced_data = scaled_data

if st.button("Run clustering"):
    if clustering_algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif clustering_algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_algorithm == "OPTICS":
        model = OPTICS(min_samples=min_samples)

    cluster_labels = model.fit_predict(reduced_data)
    st.subheader("Clustering Results")
    st.write(f"Number of identified clusters: {len(set(cluster_labels))}")

    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        st.write(f"Silhouette coefficient: {silhouette_avg:.2f}")
        if silhouette_avg > 0.7:
            st.markdown(colored_text(
                "**Sehr gut:** Eine hohe Silhouette (über 0.7) deutet auf klar definierte Cluster hin.", "green"), unsafe_allow_html=True)
        elif 0.5 <= silhouette_avg <= 0.7:
            st.markdown(colored_text(
                "**Gut:** Die Cluster sind mäßig gut definiert.", "blue"), unsafe_allow_html=True)
        elif 0.25 <= silhouette_avg < 0.5:
            st.markdown(colored_text(
                "**Mäßig:** Die Cluster könnten sich überschneiden.", "orange"), unsafe_allow_html=True)
        else:
            st.markdown(colored_text(
                "**Schlecht:** Cluster sind schwer unterscheidbar.", "red"), unsafe_allow_html=True)
    else:
        st.write("Silhouette coefficient cannot be calculated.")

    if len(set(cluster_labels)) > 1:
        dbi = davies_bouldin_score(reduced_data, cluster_labels)
        st.write(f"Davies-Bouldin Index: {dbi:.2f}")
        if dbi < 0.5:
            st.markdown(colored_text(
                "**Sehr gut:** Ein sehr niedriger Wert deutet auf gut getrennte Cluster hin.", "green"), unsafe_allow_html=True)
        elif 0.5 <= dbi < 1.5:
            st.markdown(colored_text(
                "**Gut:** Cluster sind relativ gut getrennt.", "blue"), unsafe_allow_html=True)
        elif 1.5 <= dbi <= 2.5:
            st.markdown(colored_text(
                "**Mäßig:** Cluster sind nicht optimal getrennt.", "orange"), unsafe_allow_html=True)
        else:
            st.markdown(colored_text(
                "**Schlecht:** Cluster sind stark überlappend.", "red"), unsafe_allow_html=True)
    else:
        st.write("Davies-Bouldin Index cannot be calculated.")

    if len(set(cluster_labels)) > 1:
        chs = calinski_harabasz_score(reduced_data, cluster_labels)
        st.write(f"Calinski-Harabasz Index: {chs:.2f}")
        if chs > 3000:
            st.markdown(colored_text(
                "**Sehr gut:** Ein hoher Wert deutet auf gut definierte Cluster hin.", "green"), unsafe_allow_html=True)
        elif 1500 <= chs <= 3000:
            st.markdown(colored_text(
                "**Gut:** Die Cluster sind einigermaßen gut definiert.", "blue"), unsafe_allow_html=True)
        elif 500 <= chs < 1500:
            st.markdown(colored_text(
                "**Mäßig:** Die Clusterdefinition ist verbesserungswürdig.", "orange"), unsafe_allow_html=True)
        else:
            st.markdown(colored_text(
                "**Schlecht:** Cluster sind schlecht definiert.", "red"), unsafe_allow_html=True)
    else:
        st.write("Calinski-Harabasz Index cannot be calculated.")

    if reduced_data.shape[1] == 2:
        df_cluster = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
        df_cluster['Cluster'] = cluster_labels
        fig = px.scatter(df_cluster, x="PC1", y="PC2", color="Cluster",
                         title="Cluster Visualization",
                         color_continuous_scale='viridis')
        st.plotly_chart(fig)
    else:
        st.write("Reduce dimensions to 2 for visualization.")
