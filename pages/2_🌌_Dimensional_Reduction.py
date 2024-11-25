import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sidebar import configure_sidebar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


@st.cache_data
def load_data():
    data = pd.read_parquet('./data/dataset.parquet')
    return data


data = load_data()

apply_imputation, imputation_method = configure_sidebar()

if apply_imputation:
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if imputation_method == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation_method == "Median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "Most Frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    data[numeric_data.columns] = imputer.fit_transform(numeric_data)


def preprocess_data_for_reduction(data, selected_columns, group_by_column):
    columns_to_select = selected_columns.copy()
    if group_by_column not in columns_to_select:
        columns_to_select.append(group_by_column)
    selected_data = data[columns_to_select]
    selected_data = selected_data.dropna()
    group_by_data = selected_data[group_by_column]
    processed_data = selected_data.drop(columns=[group_by_column])
    return processed_data, group_by_data


def run_tsne(data, n_components=2, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate)
    tsne_results = tsne.fit_transform(data)
    return tsne_results


def encode_columns(df):
    # Handle categorical columns using LabelEncoder
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


# Encode categorical columns in the dataset
data = encode_columns(data)


def show_page(data):
    st.title("t-SNE Dimensionality Reduction")
    st.write("Apply t-SNE to reduce the dataset dimensions.")

    selected_columns = st.multiselect(
        'Select Columns for Dimensionality Reduction', data.columns)

    group_by_column = st.selectbox(
        'Color By Column', data.columns)

    if len(selected_columns) > 0:
        processed_data, group_by_data = preprocess_data_for_reduction(
            data, selected_columns, group_by_column)
        tsne_results = run_tsne(processed_data)

        # Check if group_by_data is numeric or categorical
        if pd.api.types.is_numeric_dtype(group_by_data):
            # Use continuous colorscale
            fig = px.scatter(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                color=group_by_data,
                color_continuous_scale='RdYlGn_r',  # Green to red
                title="t-SNE Dimensionality Reduction",
                labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'}
            )
        else:
            # For 'Male' and 'Female', assign specific colors
            categories = group_by_data.unique()
            color_discrete_map = None
            if set(categories) == set(['Male', 'Female']) or set(categories) == set(['Female', 'Male']):
                color_discrete_map = {'Male': 'blue', 'Female': 'pink'}
            else:
                # Assign a default color mapping
                default_colors = px.colors.qualitative.Plotly
                color_discrete_map = {category: default_colors[i % len(
                    default_colors)] for i, category in enumerate(categories)}
            fig = px.scatter(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                color=group_by_data,
                color_discrete_map=color_discrete_map,
                title="t-SNE Dimensionality Reduction",
                labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'}
            )

        fig.update_traces(marker=dict(opacity=0.6, size=5))
        st.plotly_chart(fig)
        st.subheader('t-SNE Results')
        st.write(tsne_results)


show_page(data)
