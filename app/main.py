import streamlit as st
import scanpy as sc
import pandas as pd
import tempfile

st.set_page_config(page_title="scRNA-seq App", layout="wide")

# Keep uploaded file in session
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Single cell RNA", "Differential Expression Analysis", "About Us"])

# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("Upload your `.h5ad` file", type=["h5ad"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        adata = sc.read(tmp_path)
        st.session_state.adata = adata

        st.subheader("Batch Value Counts")
        if "batch" in adata.obs:
            st.write(adata.obs["batch"].value_counts())
        else:
            st.warning("'batch' column not found in `adata.obs`.")