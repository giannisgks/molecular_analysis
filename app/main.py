import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

from pipeline import run_sin_cell_rna_seq_preprocessing

st.set_page_config(page_title="scRNA-seq App", layout="wide")

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Algorithms", "About Us"])
# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("", type=["h5ad", "csv", "xlsx", "xls"])
    if uploaded_file is not None:
        

        # Type of the file that is uploaded
        ftype = uploaded_file.name.split('.')[-1]
        

        if ftype == 'h5ad':
            # We consider data in AnnData object format.
            sc_rna_seq_selected = True
            import os
            adata = sc.read_h5ad(uploaded_file)
        elif ftype == "csv":
            adata = pd.read_csv(uploaded_file)
        elif ftype in ["xlsx", "xls"]:
            adata = pd.read_excel(uploaded_file)
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
            if (ftype in ["xlsx", "xls"]):
                adata = pd.read_excel(uploaded_file)
            st.subheader("Shape of the dataset:")
            st.write(f"{adata.shape[0]} rows × {adata.shape[1]} columns")

            st.subheader("First 5 rows:")
            st.dataframe(adata.head())

            st.subheader("Column Data Types:")
            st.write(adata.dtypes)

            st.subheader("Missing Values per Column:")
            st.write(adata.isnull().sum())

            # Optional: show value counts for object-type columns
            categorical_cols = adata.select_dtypes(include="object").columns
            if len(categorical_cols) > 0:
                st.subheader("Value Counts (First Categorical Column):")
                st.write(adata[categorical_cols[0]].value_counts())

with tabs[1]:
    # ||----------Preprocessing for single cell RNA sequencing----------||
    st.header("Preprocessing of the data")

    if "adata" in st.session_state and ftype == 'h5ad':
        
        # ----------Gathering-parameters-of-preprocessing-from-user----------
        col1, col2 = st.columns(2)
        
        col1.subheader("Minimum genes per cell")
        col1.write("*Filters out low-quality cells that express very few genes (often empty droplets or debris).*")
        min_genes = col1.number_input("", min_value=0, value=100, step=50)

        col2.subheader("Minimum cells per gene")
        col2.write("*Removes rarely expressed genes with low analytical value.*")
        min_cells = col2.number_input("", min_value=0, value=3, step=1)

        st.subheader("Remove genes with prefixes:")
        st.caption("*Removes genes such as mitochondrial or ribosomal that may skew downstream analysis.*")

        # All prefixes to show as checkboxes
        all_prefixes = ['ERCC', 'MT-', 'mt-', 'RPS', 'RPL', 'HB', 'HSP', 'IG']

        # Default selected prefixes
        default_selected = ['ERCC', 'MT-', 'mt-']

        cols = st.columns(len(all_prefixes))

        # Render checkboxes in each column
        selected_prefixes = []
        for i, prefix in enumerate(all_prefixes):
            with cols[i]:
                checked = st.checkbox(prefix, value=(prefix in default_selected))
                if checked:
                    selected_prefixes.append(prefix)

        # Optional batch correction
        if "batch" in adata.obs.columns:
            st.subheader("Batch Correction")
            st.caption("*Corrects for technical variation between different experimental batches.*")
            
            batch_keys = ['No filtering'] + list(adata.obs.columns)
            batch_key = st.selectbox("Select Batch Key for Correction", options=batch_keys, index=batch_keys.index('batch') if 'batch' in batch_keys else 0)
        else:
            batch_key = None
            st.subheader("Batch Correction")
            st.caption("*No batch information available. Skipping batch correction.*")

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

    elif ("adata" in st.session_state and ftype != 'h5ad'): 
        normalization_method = st.selectbox("Choose normalization method:", ("None","Min-Max Scaling", "Standard Scaling"))
    
        numeric_cols = adata.select_dtypes(include=[np.number]).columns
        adata_normalized = adata.copy()

        if normalization_method == "None":
            col1, col2 = st.columns(2)
            col1.subheader("Before Normalization:")
            col1.write(adata.head())

            col2.subheader("No Normalization Applied:")

        else:
            # Apply normalization
            if normalization_method == "Standard Scaling":
                scaler = StandardScaler()
            elif normalization_method == "Min-Max Scaling":
                scaler = MinMaxScaler()

            adata_normalized[numeric_cols] = scaler.fit_transform(adata[numeric_cols])

            # Display side-by-side
            col1, col2 = st.columns(2)
            col1.subheader("Before Normalization:")
            col1.write(adata.head())

            col2.subheader(f"After Normalization using {normalization_method}:")
            col2.write(adata_normalized.head())

            adata = adata_normalized

        if st.checkbox("Drop rows with missing values"):
            original_rows = adata.shape[0]
            adata.dropna(inplace=True)
            new_rows = adata.shape[0]
            rows_dropped = original_rows - new_rows
            st.info(f"Dropped {rows_dropped} row(s) with missing values.")

        if st.checkbox("Remove outliers (Z-score > 3)"):
            original_rows = adata.shape[0]

            # Compute z-scores only for numeric columns
            numeric_data = adata.select_dtypes(include=[np.number])
            z_scores = np.abs(zscore(numeric_data, nan_policy='omit'))

            # Create a boolean mask: True for rows without extreme outliers
            mask = (z_scores < 3).all(axis=1)
            adata = adata[mask]

            new_rows = adata.shape[0]
            rows_dropped = original_rows - new_rows
            st.info(f"Removed {rows_dropped} row(s) as outliers (Z-score > 3).")

        # --loads too slow--
        #if st.checkbox("Show correlation heatmap"):
            #st.subheader("Correlation Heatmap")
            #corr = adata.select_dtypes(include=[np.number]).corr()
            #fig, ax = plt.subplots(figsize=(10, 8))
            #sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            #st.pyplot(fig)