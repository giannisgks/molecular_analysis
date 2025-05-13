import streamlit as st
import scanpy as sc
import pandas as pd
import scipy

st.set_page_config(page_title="scRNA-seq App", layout="wide")

# Keep uploaded file in session
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Single cell RNA", "Differential Expression Analysis", "About Us"])


if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Upload Data"

# --------- Tab 1: Upload Data ---------
if st.session_state.selected_tab == "Upload Data":
    with tabs[0]:
        st.header("Upload Your Data")

        uploaded_file = st.file_uploader("Upload your `.h5ad` file", type=["h5ad"])

        if uploaded_file:
            adata = sc.read_h5ad(uploaded_file)

            st.session_state.adata = adata

            if "batch" in adata.obs:
                st.success("✅ File successfully uploaded and loaded.")

                st.subheader("Shape of expression matrix:")
                st.write(f"{adata.shape[0]} cells x {adata.shape[1]} genes")

                st.subheader("Dataset columns:")
                st.markdown("<br>".join(adata.obs.columns), unsafe_allow_html=True)

                st.subheader("UMAP Available:")
                if 'X_umap' in adata.obsm:
                    st.write("Yes, it is.")
                else:
                    st.warning("❌ UMAP is not available. You need to preprocess the data before visualization.")

                st.subheader("Is the matrix sparse?")
                if (scipy.sparse.issparse(adata.X)):
                    st.success("✅ The matrix is sparse.")
                else:
                    st.warning("❌ The matrix is not sparse. You need to preprocess the data before visualization.")
                
                st.subheader("Number of genes to display")
                selected_gene_count = st.slider("",5, 100, 10)
                st.write(adata.var_names[:selected_gene_count])

                if st.button("Proceed to Preprocessing"):
                    st.session_state.selected_tab = "Preprocessing"
                    st.rerun()

            else:
                st.warning("'batch' column not found in `adata.obs`.")