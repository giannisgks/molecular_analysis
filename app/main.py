import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

st.set_page_config(page_title="scRNA-seq App", layout="wide")

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Visualisation (before vs after)","Algorithms", "About Us"])
# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("", type=["h5ad"])
    if uploaded_file is not None:
        

        # Type of the file that is uploaded
        ftype = uploaded_file.name.split('.')[-1]
        

        if ftype == 'h5ad':
            # We consider data in AnnData object format.
            sc_rna_seq_selected = True
            import os
            adata = sc.read_h5ad(uploaded_file)
        else:   
            st.error(f'File type "${ftype}" is not supported')

        st.session_state.adata = adata
        st.success("✅ File successfully uploaded and loaded.")

        if ftype == "h5ad" and "batch" in adata.obs:
            st.subheader("Dataset Overview")

            # Use columns for matrix shape and sparsity info
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Shape of expression matrix:**")
                st.info(f"{adata.shape[0]} cells × {adata.shape[1]} genes")

            with col2:
                st.markdown("**Is the matrix sparse?**")
                if scipy.sparse.issparse(adata.X):
                    st.success("✅ The matrix is sparse.")
                else:
                    st.error("❌ The matrix is not sparse. You need to preprocess the data before visualization.")

            st.markdown("---")

            # Dataset columns
            st.subheader("**Dataset metadata columns (`adata.obs`)**")
            # Show as scrollable dataframe with max height
            st.dataframe(adata.obs.columns.to_list(), height=180)

            st.markdown("---")

            st.subheader("**Genes preview**")
            # Number of genes to display slider with better label
            selected_gene_count = st.slider(
                label="Select number of genes to preview",
                min_value=5,
                max_value=100,
                value=10,
                step=5,
            )

            st.markdown(f"**Previewing first {selected_gene_count} genes:**")
            # Show genes as a nicely formatted table in 1 column
            genes_df = pd.DataFrame(adata.var_names[:selected_gene_count], columns=["Gene Names"])
            st.dataframe(genes_df, height=200)
            
            # --- Extra: Metadata preview ---
            st.subheader("Data from `adata.obs`:")
            st.dataframe(adata.obs.head())

            # --- Genes info preview (var) ---
            if adata.var is not None:
                st.subheader("Information from `adata.var`:")
                st.dataframe(adata.var.head())

            # --- Calculate percentage of mitochondrial genes ---
            if not "pct_mito" in adata.obs.columns:
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

            # --- Quality Control Plots ---
            st.subheader("Quality Control Plots")
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))

            sns.histplot(adata.obs["n_genes_by_counts"], bins=50, ax=axs[0], kde=True)
            axs[0].set_title("n_genes_by_counts")

            sns.histplot(adata.obs["total_counts"], bins=50, ax=axs[1], kde=True)
            axs[1].set_title("total_counts")

            sns.histplot(adata.obs["pct_counts_mt"], bins=50, ax=axs[2], kde=True)
            axs[2].set_title("pct_counts_mt")

            st.pyplot(fig)

            # --- Select & Display gene expression ---
            st.subheader("View expression of a gene")
            gene_name = st.selectbox("Select gene", adata.var_names)

            col1, col2 = st.columns(2)
            if gene_name in adata.var_names:
                with col1:
                    st.write(f"Expression of gene: {gene_name}")
                    gene_exp = adata[:, gene_name].X.toarray().flatten() if scipy.sparse.issparse(adata.X) else adata[:, gene_name].X.flatten()
                    fig2, ax2 = plt.subplots()
                    sns.histplot(gene_exp, bins=50, kde=True, ax=ax2)
                    ax2.set_xlabel("Expression")
                    ax2.set_title(f"{gene_name} Expression Distribution")
                    st.pyplot(fig2)

with tabs[1]:
    st.header("Preprocessing of the data")

    if "adata" in st.session_state and ftype == 'h5ad':
        adata = st.session_state.adata  # load AnnData from session

        # === User Parameters ===
        min_genes = st.number_input("Minimum genes per cell", min_value=0, value=100, step=50, key="min_genes_preproc")
        min_cells = st.number_input("Minimum cells per gene", min_value=0, value=3, step=1, key="min_cells_preproc")
        remove_prefixes = st.multiselect(
            "Remove genes with prefixes:",
            options=['ERCC', 'MT-', 'mt-', 'RPS', 'RPL', 'HB', 'HSP', 'IG'],
            default=['ERCC', 'MT-', 'mt-'],
            key="remove_prefixes"
        )

        batch_key = None
        if "batch" in adata.obs.columns:
            batch_key = st.selectbox("Batch key for batch correction", options=["None"] + list(adata.obs.columns), index=1, key="batch_key_select")
            if batch_key == "None":
                batch_key = None

        # === Optional Steps ===
        st.markdown("### Optional Preprocessing Steps")
        do_normalize = st.checkbox("Normalize Total Counts", value=True)
        do_log = st.checkbox("Log1p Transform", value=True)
        do_scaling = st.checkbox("Scale Data", value=True)
        do_hvg = st.checkbox("Select Highly Variable Genes", value=True)
        do_pca = st.checkbox("Run PCA and UMAP", value=True)

        # === Run Button ===
        if st.button("Run Preprocessing"):
            with st.spinner("Running preprocessing..."):
                # Save raw copy
                st.session_state.adata_raw = adata.copy()
                sc.pp.calculate_qc_metrics(st.session_state.adata_raw, qc_vars=["mt"], inplace=True)

                # Step 1: Filter
                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)

                # Step 2: Remove genes by prefix
                adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(remove_prefixes))]]

                # Step 3: Normalization
                if do_normalize:
                    sc.pp.normalize_total(adata, target_sum=1e4)

                # Step 4: Log1p
                if do_log:
                    sc.pp.log1p(adata)

                # Step 5: HVGs
                if do_hvg:
                    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                    adata.raw = adata.copy()
                    adata = adata[:, adata.var.highly_variable]
                else:
                    adata.raw = adata.copy()  # Still preserve the raw version

                # Step 6: Scaling
                if do_scaling:
                    sc.pp.scale(adata, max_value=10)

                # Step 7: Dimensionality Reduction
                if do_pca:
                    sc.pp.pca(adata)
                    sc.pp.neighbors(adata)
                    sc.tl.umap(adata)

                # Step 8: Batch Correction
                if batch_key and batch_key in adata.obs.columns:
                    sc.pp.combat(adata, key=batch_key)

                # Step 9: QC metrics after all changes
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

                # Save back
                st.session_state.adata = adata.copy()

                st.success("✅ Preprocessing completed!")

                st.markdown(f"**Procced to the next tab to see the changes visually!**")

with tabs[2]:
    st.header("Visualization")
    st.markdown("### Compare Before and After Preprocessing")

        # Compute UMAP for raw data if missing
    if "adata_raw" in st.session_state:
        adata_raw = st.session_state.adata_raw
        if "X_umap" not in adata_raw.obsm.keys():
            try:
                sc.pp.neighbors(adata_raw)  # neighbors required before UMAP
                sc.tl.umap(adata_raw)
            except Exception as e:
                st.warning(f"Could not compute UMAP for raw data: {e}")

    # Compute UMAP for processed data if missing
    if "adata" in st.session_state:
        adata = st.session_state.adata
        if "X_umap" not in adata.obsm.keys():
            try:
                sc.pp.neighbors(adata)
                sc.tl.umap(adata)
            except Exception as e:
                st.warning(f"Could not compute UMAP for filtered data: {e}")

    if "adata_raw" in st.session_state and "adata" in st.session_state:
        adata_raw = st.session_state.adata_raw  # Unfiltered data
        adata = st.session_state.adata        # Filtered data

        col1, col2 = st.columns(2)

        # === Gene Count Per Cell ===
        col1.subheader("Before Preprocessing: Genes per Cell")
        fig1, ax1 = plt.subplots()
        sns.histplot(adata_raw.obs["n_genes_by_counts"], bins=50, ax=ax1)
        ax1.set_title("n_genes_by_counts (Raw)")
        col1.pyplot(fig1)

        col2.subheader("After Preprocessing: Genes per Cell")
        fig2, ax2 = plt.subplots()
        sns.histplot(adata.obs["n_genes_by_counts"], bins=50, ax=ax2)
        ax2.set_title("n_genes_by_counts (Filtered)")
        col2.pyplot(fig2)

        # === Total Counts Per Cell ===
        col1.subheader("Before: Total Counts per Cell")
        fig_tc_raw, ax_tc_raw = plt.subplots()
        sns.histplot(adata_raw.obs["total_counts"], bins=50, ax=ax_tc_raw)
        ax_tc_raw.set_title("total_counts (Raw)")
        col1.pyplot(fig_tc_raw)

        col2.subheader("After: Total Counts per Cell")
        fig_tc_filtered, ax_tc_filtered = plt.subplots()
        sns.histplot(adata.obs["total_counts"], bins=50, ax=ax_tc_filtered)
        ax_tc_filtered.set_title("total_counts (Filtered)")
        col2.pyplot(fig_tc_filtered)

        # === Mitochondrial Gene Percentage ===
        col1.subheader("Before: % Mitochondrial Genes")
        fig3, ax3 = plt.subplots()
        sns.histplot(adata_raw.obs["pct_counts_mt"], bins=50, ax=ax3)
        ax3.set_title("pct_counts_mt (Raw)")
        col1.pyplot(fig3)

        col2.subheader("After: % Mitochondrial Genes")
        fig4, ax4 = plt.subplots()
        sns.histplot(adata.obs["pct_counts_mt"], bins=50, ax=ax4)
        ax4.set_title("pct_counts_mt (Filtered)")
        col2.pyplot(fig4)

        # === Scatter: n_genes_by_counts vs total_counts ===
        col1.subheader("Before: Genes vs Total Counts")
        fig_scatter_raw, ax_scatter_raw = plt.subplots()
        ax_scatter_raw.scatter(
            adata_raw.obs["total_counts"], adata_raw.obs["n_genes_by_counts"], alpha=0.3, s=5)
        ax_scatter_raw.set_xlabel("total_counts (Raw)")
        ax_scatter_raw.set_ylabel("n_genes_by_counts (Raw)")
        col1.pyplot(fig_scatter_raw)

        col2.subheader("After: Genes vs Total Counts")
        fig_scatter_filt, ax_scatter_filt = plt.subplots()
        ax_scatter_filt.scatter(
            adata.obs["total_counts"], adata.obs["n_genes_by_counts"], alpha=0.3, s=5)
        ax_scatter_filt.set_xlabel("total_counts (Filtered)")
        ax_scatter_filt.set_ylabel("n_genes_by_counts (Filtered)")
        col2.pyplot(fig_scatter_filt)

        col1.subheader("UMAP Before Preprocessing")
        fig_umap_raw, ax_umap_raw = plt.subplots(figsize=(6,6))
        if "X_umap" in adata_raw.obsm.keys():
            sc.pl.umap(adata_raw, ax=ax_umap_raw, show=False)
        else:
            ax_umap_raw.text(0.5, 0.5, "UMAP not computed", ha='center')
        col1.pyplot(fig_umap_raw)

        col2.subheader("UMAP After Preprocessing")
        fig_umap_filt, ax_umap_filt = plt.subplots(figsize=(6,6))
        if "X_umap" in adata.obsm.keys():
            sc.pl.umap(adata, ax=ax_umap_filt, show=False)
        else:
            ax_umap_filt.text(0.5, 0.5, "UMAP not computed", ha='center')
        col2.pyplot(fig_umap_filt)

        # === Gene Expression Distribution (Selectable Gene) ===
        col1.subheader("Gene Expression (Raw)")
        selected_gene = col1.selectbox("Select a gene", adata_raw.var_names, key="gene_exp_select")

        if selected_gene in adata_raw.var_names:
            gene_exp_raw = adata_raw[:, selected_gene].X
            gene_exp_raw = gene_exp_raw.toarray().flatten() if scipy.sparse.issparse(gene_exp_raw) else gene_exp_raw.flatten()
            fig5, ax5 = plt.subplots()
            sns.histplot(gene_exp_raw, bins=50, kde=True, ax=ax5)
            ax5.set_title(f"{selected_gene} Expression (Raw)")
            col1.pyplot(fig5)

        if selected_gene in adata.var_names:
            col2.subheader("Gene Expression (Filtered)")
            gene_exp_filtered = adata[:, selected_gene].X
            gene_exp_filtered = gene_exp_filtered.toarray().flatten() if scipy.sparse.issparse(gene_exp_filtered) else gene_exp_filtered.flatten()
            fig6, ax6 = plt.subplots()
            sns.histplot(gene_exp_filtered, bins=50, kde=True, ax=ax6)
            ax6.set_title(f"{selected_gene} Expression (Filtered)")
            col2.pyplot(fig6)

    else:
        st.warning("Please upload and preprocess data first.")