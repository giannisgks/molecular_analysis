import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

from pipeline import run_sin_cell_rna_seq_preprocessing

st.set_page_config(page_title="scRNA-seq App", layout="wide")

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Single cell RNA", "Differential Expression Analysis", "About Us"])

# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("", type=["h5ad", "csv", "xlsx", "xls"])


    # If checkBox selected, we can try to adata.obs or batch otherwise we consider the data in iris format.
    sc_rna_seq_selected = st.checkbox("Single cell RNA sequencing", value=True)

    if uploaded_file is None:
        

        # Type of the file that is uploaded
        # ftype = uploaded_file.name.split('.')[-1]
        ftype = 'h5ad'

        if ftype == 'h5ad':
            # We consider data in AnnData object format.
            sc_rna_seq_selected = True
            import os
            adata = sc.read_h5ad(os.path.join(os.getcwd(), 'pancreas_data.h5ad'))
            # adata = sc.read_h5ad(uploaded_file)
        elif ftype == "csv":
            adata = pd.read_csv(uploaded_file)
        elif ftype in ["xlsx", "xls"]:
            pd.read_excel(uploaded_file)
        else:   
            st.error(f'File type "${ftype}" is not supported')

        st.session_state.adata = adata
        st.success("✅ File successfully uploaded and loaded.")

        # If biological file(file is in AnnData object format):
        if ftype == "h5ad" and "batch" in adata.obs:
            st.subheader("Shape of expression matrix:")
            st.write(f"{adata.shape[0]} cells x {adata.shape[1]} genes")

            st.subheader("Dataset columns:")
            st.markdown("<br>".join(adata.obs.columns), unsafe_allow_html=True)

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
    # ||----------Preprocessing for single cell RNA sequencing----------||
    st.header("Preprocessing single cell RNA sequencing")

    if "adata" in st.session_state:
        adata = st.session_state.adata

        # ----------Gathering-parameters-of-preprocessing-from-user----------
        col1, col2 = st.columns(2)
        col1.subheader("Minimum genes per cell")
        min_genes = col1.number_input("", min_value=0, value=100, step=50)
        col2.subheader("Minimum cells per gene")
        min_cells = col2.number_input("", min_value=0, value=3, step=1)


        st.subheader("Remove genes with prefixes:")
        # All prefixes to show as checkboxes
        all_prefixes = ['ERCC', 'MT-', 'mt-', 'RPS', 'RPL', 'HB', 'HSP', 'IG']

        # Default selected prefixes
        default_selected = ['ERCC', 'MT-', 'mt-']

        # Create 8 columns
        cols = st.columns(len(all_prefixes))

        # Render checkboxes in each column
        selected_prefixes = []
        for i, prefix in enumerate(all_prefixes):
            with cols[i]:
                checked = st.checkbox(prefix, value=(prefix in default_selected))
                if checked:
                    selected_prefixes.append(prefix)


        st.write("preprocessing...")
        run_sin_cell_rna_seq_preprocessing(adata, min_genes, min_cells, selected_prefixes)
        st.write("preprocess has completed")









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