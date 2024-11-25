# sidebar.py

import streamlit as st
from sklearn.impute import SimpleImputer

# Funktion zur Erstellung der Sidebar


def configure_sidebar():
    st.sidebar.header("Imputation Options")

    # Checkbox zur Aktivierung der Imputation
    apply_imputation = st.sidebar.checkbox("Apply Imputation")

    # Wenn Imputation aktiviert ist, Auswahl der Methode ermöglichen
    if apply_imputation:
        imputation_method = st.sidebar.selectbox(
            "Select Imputation Method",
            options=["Mean", "Median", "Most Frequent"]
        )
        return apply_imputation, imputation_method
    return apply_imputation, None
