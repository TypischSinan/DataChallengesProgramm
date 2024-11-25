import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sidebar import configure_sidebar
from sklearn.impute import SimpleImputer


def show_page(data):
    st.title("ICBHI 2017 Challenge Dataset")

    st.markdown("""
    **The ICBHI 2017 Challenge Dataset** is an open-access database for research in respiratory sound classification. The key points are:
    
    - **Purpose**: To evaluate and develop algorithms for classifying respiratory sounds.
    - **Content**: The database contains respiratory sounds recorded from patients with various lung and respiratory diseases.
    - **Demographic Data**: Each record includes information such as participant age, sex, BMI (for adults), or weight and height (for children).
    - **Source**: The data was collected in multiple clinics and is available in various formats.
    - **Target Audience**: Researchers and developers interested in respiratory sound analysis and classification.
    
    The dataset is freely available on the ICBHI Challenge website and is described in Rocha et al. (2019).
    """)

    st.title("Medical Dataset Overview")
    st.write("Explore vital signs and lab results over ICU stay")

    st.subheader("Dataset Overview")
    st.write(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.header("Visualization")
    df_selected = data[['Patient number', 'Diagnosis']]
    df_selected.drop_duplicates(subset='Patient number')
    plotly_chart = px.bar(df_selected, x='Patient number', y='Diagnosis')
    st.plotly_chart(plotly_chart)

    df_selected = data[['Patient number', 'Sex', 'Diagnosis']]
    df_selected = df_selected.drop_duplicates(subset='Patient number')

    df_counts = df_selected.groupby(
        ['Diagnosis', 'Sex']).size().reset_index(name='Count')

    color_map = {'M': 'blue', 'F': 'pink'}

    plotly_chart = px.bar(
        df_counts,
        x='Diagnosis',
        y='Count',
        color='Sex',
        color_discrete_map=color_map,
        barmode='group'
    )

    st.plotly_chart(plotly_chart)


def load_data():
    data = pd.read_parquet('./data/dataset.parquet')

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

    return data


data = load_data()
show_page(data)
