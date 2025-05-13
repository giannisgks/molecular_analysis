import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

st.set_page_config(page_title="scRNA-seq App", layout="wide")

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Single cell RNA", "Differential Expression Analysis", "About Us"])

# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("Upload your `.h5ad` file", type=["h5ad"])

    if uploaded_file is not None:
        adata = sc.read_h5ad(uploaded_file)
        st.session_state.adata = adata
        st.success("✅ File successfully uploaded and loaded.")

        if "batch" in adata.obs:
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

        else:
            st.warning("'batch' column not found in `adata.obs`.")


with tabs[1]:
    st.header("Preprocessing")

    if "adata" in st.session_state:
        adata = st.session_state.adata

        st.subheader("QC Metrics Before Filtering")
        st.write("""
**QC (Quality Control) metrics** in single-cell RNA-seq help identify low-quality cells or artifacts.  
Common ones include:

- **n_genes_by_counts**: Number of genes detected per cell  
- **pct_counts_mt**: Percentage of reads mapping to mitochondrial genes (high % indicates stressed or dying cells)
""")

        adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')  # mitochondrial genes
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(adata.obs['n_genes_by_counts'], bins=50, ax=axes[0])
        axes[0].set_title("Genes per cell")
        sns.histplot(adata.obs['pct_counts_mt'], bins=50, ax=axes[1])
        axes[1].set_title("Mitochondrial gene % per cell")
        st.pyplot(fig)

        if 'X_umap' in adata.obsm:
            st.subheader("UMAP Plot")
            sc.pl.umap(adata, color=['batch'], show=False)
            st.pyplot(plt.gcf())

            option = st.selectbox("Color UMAP by:", adata.obs.columns)
            sc.pl.umap(adata, color=option, show=False)
            st.pyplot(plt.gcf())
