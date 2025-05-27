import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def show():
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

        st.divider()
        col1, col2 = st.columns(2)
        # === Gene Count Per Cell ===
        col1.subheader("Before Preprocessing: Genes per Cell")
        fig1, ax1 = plt.subplots()
        sns.histplot(adata_raw.obs["n_genes_by_counts"], bins=50, ax=ax1)
        ax1.set_title("n_genes_by_counts (Raw)")
        col1.pyplot(fig1)
        col1.markdown('<p style="color: grey;">Distribution of number of genes detected per cell before preprocessing.</p>', unsafe_allow_html=True)

        col2.subheader("After Preprocessing: Genes per Cell")
        fig2, ax2 = plt.subplots()
        sns.histplot(adata.obs["n_genes_by_counts"], bins=50, ax=ax2)
        ax2.set_title("n_genes_by_counts (Filtered)")
        col2.pyplot(fig2)
        col2.markdown('<p style="color: grey;">Distribution of number of genes detected per cell after preprocessing.</p>', unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        # === Total Counts Per Cell ===
        col1.subheader("Before: Total Counts per Cell")
        fig_tc_raw, ax_tc_raw = plt.subplots()
        sns.histplot(adata_raw.obs["total_counts"], bins=50, ax=ax_tc_raw)
        ax_tc_raw.set_title("total_counts (Raw)")
        col1.pyplot(fig_tc_raw)
        col1.markdown('<p style="color: grey;">Distribution of total counts per cell before preprocessing.</p>', unsafe_allow_html=True)

        col2.subheader("After: Total Counts per Cell")
        fig_tc_filtered, ax_tc_filtered = plt.subplots()
        sns.histplot(adata.obs["total_counts"], bins=50, ax=ax_tc_filtered)
        ax_tc_filtered.set_title("total_counts (Filtered)")
        col2.pyplot(fig_tc_filtered)
        col2.markdown('<p style="color: grey;">Distribution of total counts per cell after preprocessing.</p>', unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        # === Mitochondrial Gene Percentage ===
        col1.subheader("Before: % Mitochondrial Genes")
        fig3, ax3 = plt.subplots()
        sns.histplot(adata_raw.obs["pct_counts_mt"], bins=50, ax=ax3)
        ax3.set_title("pct_counts_mt (Raw)")
        col1.pyplot(fig3)
        col1.markdown('<p style="color: grey;">Percentage of mitochondrial gene counts per cell before preprocessing.</p>', unsafe_allow_html=True)

        col2.subheader("After: % Mitochondrial Genes")
        fig4, ax4 = plt.subplots()
        sns.histplot(adata.obs["pct_counts_mt"], bins=50, ax=ax4)
        ax4.set_title("pct_counts_mt (Filtered)")
        col2.pyplot(fig4)
        col2.markdown('<p style="color: grey;">Percentage of mitochondrial gene counts per cell after preprocessing.</p>', unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        # === Scatter: n_genes_by_counts vs total_counts ===
        col1.subheader("Before: Genes vs Total Counts")
        fig_scatter_raw, ax_scatter_raw = plt.subplots()
        ax_scatter_raw.scatter(
            adata_raw.obs["total_counts"], adata_raw.obs["n_genes_by_counts"], alpha=0.3, s=5)
        ax_scatter_raw.set_xlabel("total_counts (Raw)")
        ax_scatter_raw.set_ylabel("n_genes_by_counts (Raw)")
        col1.pyplot(fig_scatter_raw)
        col1.markdown('<p style="color: grey;">Scatter plot of total counts vs genes detected before preprocessing.</p>', unsafe_allow_html=True)

        col2.subheader("After: Genes vs Total Counts")
        fig_scatter_filt, ax_scatter_filt = plt.subplots()
        ax_scatter_filt.scatter(
            adata.obs["total_counts"], adata.obs["n_genes_by_counts"], alpha=0.3, s=5)
        ax_scatter_filt.set_xlabel("total_counts (Filtered)")
        ax_scatter_filt.set_ylabel("n_genes_by_counts (Filtered)")
        col2.pyplot(fig_scatter_filt)
        col2.markdown('<p style="color: grey;">Scatter plot of total counts vs genes detected after preprocessing.</p>', unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        col1.subheader("UMAP Before Preprocessing")
        fig_umap_raw, ax_umap_raw = plt.subplots(figsize=(6,6))
        if "X_umap" in adata_raw.obsm.keys():
            sc.pl.umap(adata_raw, ax=ax_umap_raw, show=False)
        else:
            ax_umap_raw.text(0.5, 0.5, "UMAP not computed", ha='center')
        col1.pyplot(fig_umap_raw)
        col1.markdown('<p style="color: grey;">UMAP plot showing cell clustering before preprocessing.</p>', unsafe_allow_html=True)

        col2.subheader("UMAP After Preprocessing")
        fig_umap_filt, ax_umap_filt = plt.subplots(figsize=(6,6))
        if "X_umap" in adata.obsm.keys():
            sc.pl.umap(adata, ax=ax_umap_filt, show=False)
        else:
            ax_umap_filt.text(0.5, 0.5, "UMAP not computed", ha='center')
        col2.pyplot(fig_umap_filt)
        col2.markdown('<p style="color: grey;">UMAP plot showing cell clustering after preprocessing.</p>', unsafe_allow_html=True)

        st.markdown("---")

        # This ensures the selectbox does NOT affect column alignment
        selected_gene = st.selectbox("Select a gene", adata_raw.var_names, key="gene_exp_select")

        # Now start the side-by-side layout
        col1, col2 = st.columns(2)

        # Raw Expression Plot
        col1.subheader("Gene Expression (Raw)")
        if selected_gene in adata_raw.var_names:
            gene_exp_raw = adata_raw[:, selected_gene].X
            gene_exp_raw = gene_exp_raw.toarray().flatten() if scipy.sparse.issparse(gene_exp_raw) else gene_exp_raw.flatten()
            fig5, ax5 = plt.subplots()
            sns.histplot(gene_exp_raw, bins=50, kde=True, ax=ax5)
            ax5.set_title(f"{selected_gene} Expression (Raw)")
            col1.pyplot(fig5)
            col1.markdown(
                '<p style="color: grey;">Distribution of expression levels for the selected gene in raw data.</p>',
                unsafe_allow_html=True
            )

        # Filtered Expression Plot
        col2.subheader("Gene Expression (Filtered)")
        if selected_gene in adata.var_names:
            gene_exp_filtered = adata[:, selected_gene].X
            gene_exp_filtered = gene_exp_filtered.toarray().flatten() if scipy.sparse.issparse(gene_exp_filtered) else gene_exp_filtered.flatten()
            fig6, ax6 = plt.subplots()
            sns.histplot(gene_exp_filtered, bins=50, kde=True, ax=ax6)
            ax6.set_title(f"{selected_gene} Expression (Fwith tabs[3]:iltered)")
            col2.pyplot(fig6)
            col2.markdown(
                '<p style="color: grey;">Distribution of expression levels for the selected gene in filtered data.</p>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please upload and preprocess data first.")
