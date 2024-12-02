import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st
from sidebar import configure_sidebar
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


@st.cache_data
def load_data():
    data = pd.read_parquet('./data/dataset.parquet')
    return data


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
manuel_or_all = st.toggle("All columns for clustering", False)
if manuel_or_all:
    columns_for_clustering = numeric_data.columns.tolist()
else:
    columns_for_clustering = st.multiselect(
        "Select columns for clustering:",
        options=numeric_data.columns.tolist(),
        default=["Patient Number"]
    )

if apply_imputation:
    if imputation_method == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation_method == "Median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "Most Frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    numeric_only = columns_for_clustering.select_dtypes(
        include=['float64', 'int64'])
    imputed_data = imputer.fit_transform(numeric_only)
    columns_for_clustering = pd.DataFrame(
        imputed_data, columns=numeric_only.columns)

numeric_columns = numeric_data[numeric_data.columns.intersection(
    columns_for_clustering)]
columns_for_clustering = numeric_columns.copy()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(columns_for_clustering)

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
    else:
        st.write("Silhouette coefficient cannot be calculated.")
    if len(set(cluster_labels)) > 1:
        dbi = davies_bouldin_score(reduced_data, cluster_labels)
        st.write(f"Davies-Bouldin-Index: {dbi:.2f}")
    else:
        st.write("Davies-Bouldin-Index cannot be calculated.")
    if len(set(cluster_labels)) > 1:
        chs = calinski_harabasz_score(reduced_data, cluster_labels)
        st.write(f"Calinski-Harabasz-Index: {chs:.2f}")
    else:
        st.write("Calinski-Harabasz-Index cannot be calculated.")
    if reduced_data.shape[1] == 2:
        df_cluster = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
        df_cluster['Cluster'] = cluster_labels
        fig = px.scatter(df_cluster, x="PC1", y="PC2", color="Cluster",
                         title="Cluster Visualization",
                         color_continuous_scale='viridis')
        st.plotly_chart(fig)
    else:
        st.write("Reduce dimensions to 2 for visualization.")
