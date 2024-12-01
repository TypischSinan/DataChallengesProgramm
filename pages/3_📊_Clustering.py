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


# Load data
data = load_data()

# Sidebar configuration
apply_imputation, imputation_method = configure_sidebar()

# Extract only numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Remove specific columns from numeric_data
columns_to_drop = ["Adult BMI", "Child Weight", "Child Height", "Age"]
numeric_data = numeric_data.drop(columns=columns_to_drop, errors='ignore')

# Main content of the app
st.title("Clustering Analysis")

# Clustering settings
st.subheader("Clustering Settings")

# Select the clustering algorithm
clustering_algorithm = st.radio(
    "Choose a clustering algorithm",
    ["KMeans", "DBSCAN", "OPTICS"]
)

# Select columns for clustering
manuel_or_all = st.toggle("All columns for clustering", False)
if manuel_or_all:
    columns_for_clustering = numeric_data.columns.tolist()
else:
    columns_for_clustering = st.multiselect(
        "Select columns for clustering:",
        options=numeric_data.columns.tolist(),
        default=["Patient Number"]  # No columns selected by default
    )

# Filter data based on column selection
selected_data = numeric_data[columns_for_clustering]

# Apply imputation if enabled
if apply_imputation:
    if imputation_method == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation_method == "Median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "Most Frequent":
        imputer = SimpleImputer(strategy="most_frequent")

    selected_data = pd.DataFrame(imputer.fit_transform(
        selected_data), columns=selected_data.columns)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# Parameter inputs for KMeans
if clustering_algorithm == "KMeans":
    n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

# Parameter inputs for DBSCAN
if clustering_algorithm == "DBSCAN":
    eps = st.slider("Eps (Neighborhood radius)", 0.1, 5.0, 0.5)
    min_samples = st.slider("Minimum samples", 1, 20, 5)

# Parameter inputs for OPTICS
if clustering_algorithm == "OPTICS":
    min_samples = st.slider("Minimum samples", 1, 20, 5)

# Reduce dimensions
dimensionality_reduction = st.checkbox("Reduce dimensions (PCA)")
if dimensionality_reduction:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
else:
    reduced_data = scaled_data

# Apply clustering
if st.button("Run clustering"):
    if clustering_algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif clustering_algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_algorithm == "OPTICS":
        model = OPTICS(min_samples=min_samples)

    # Compute cluster labels
    cluster_labels = model.fit_predict(reduced_data)

    # Display results
    st.subheader("Clustering Results")
    st.write(f"Number of identified clusters: {len(set(cluster_labels))}")

    # Quality measure: Silhouette coefficient
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        st.write(f"Silhouette coefficient: {silhouette_avg:.2f}")
    else:
        st.write(
            "Silhouette coefficient cannot be calculated.")

       # Quality measure: Davies-Bouldin-Index
    if len(set(cluster_labels)) > 1:
        dbi = davies_bouldin_score(reduced_data, cluster_labels)
        st.write(f"Davies-Bouldin-Index: {dbi:.2f}")
    else:
        st.write(
            "Davies-Bouldin-Index cannot be calculated.")

    # Quality measure: Calinski-Harabasz-Index
    if len(set(cluster_labels)) > 1:
        chs = calinski_harabasz_score(reduced_data, cluster_labels)
        st.write(f"Calinski-Harabasz-Index: {chs:.2f}")
    else:
        st.write(
            "Calinski-Harabasz-Index cannot be calculated.")

    # Visualize clusters
    if reduced_data.shape[1] == 2:  # 2D visualization
        # Plotly visualization
        df_cluster = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
        df_cluster['Cluster'] = cluster_labels
        fig = px.scatter(df_cluster, x="PC1", y="PC2", color="Cluster",
                         title="Cluster Visualization",
                         color_continuous_scale='viridis')
        st.plotly_chart(fig)
    else:
        st.write("Reduce dimensions to 2 for visualization.")
