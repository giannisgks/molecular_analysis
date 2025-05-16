import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import scipy
import bbknn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

st.set_page_config(page_title="scRNA-seq App", layout="wide")

with open("./app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Visualisation (before vs after)","Algorithms", "About Us"])
# --------- Tab 1: Upload Data ---------
with tabs[0]:
    st.header("Upload Your Data")
    st.markdown('<p style="color: grey;">This text should be grey</p>', unsafe_allow_html=True)
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
            st.markdown("---")
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
            st.markdown("---")
            st.subheader("Data from `adata.obs`:")
            st.dataframe(adata.obs.head())

            # --- Genes info preview (var) ---
            if adata.var is not None:
                st.subheader("Information from `adata.var`:")
                st.dataframe(adata.var)

            # --- Calculate percentage of mitochondrial genes ---
            if not "pct_mito" in adata.obs.columns:
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

            # --- Quality Control Plots ---
            st.markdown("---")
            st.subheader("Quality Control Plots")
            st.markdown('<p style="color: grey;">cells</p>', unsafe_allow_html=True)
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
            st.markdown('<p style="color: grey;">View the distribution of expression for a specific gene across all cells.</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                gene_name1 = st.selectbox("Select gene (left)", adata.var_names, key="gene_selectbox_1")
                st.write(f"Expression of gene: {gene_name1}")
                gene_exp = adata[:, gene_name1].X.toarray().flatten() if scipy.sparse.issparse(adata.X) else adata[:, gene_name1].X.flatten()
                fig2, ax2 = plt.subplots()
                sns.histplot(gene_exp, bins=50, kde=True, ax=ax2)
                ax2.set_xlabel("Expression")
                ax2.set_title(f"{gene_name1} Expression Distribution")
                st.pyplot(fig2)

            with col2:
                gene_name2 = st.selectbox("Select gene (right)", adata.var_names, key="gene_selectbox_2")
                st.write(f"Expression of gene: {gene_name2}")
                gene_exp = adata[:, gene_name2].X.toarray().flatten() if scipy.sparse.issparse(adata.X) else adata[:, gene_name2].X.flatten()
                fig3, ax3 = plt.subplots()
                sns.histplot(gene_exp, bins=50, kde=True, ax=ax3)
                ax3.set_xlabel("Expression")
                ax3.set_title(f"{gene_name2} Expression Distribution")
                st.pyplot(fig3)


with tabs[1]:
    st.header("Preprocessing of the data")

    if "adata" in st.session_state and ftype == 'h5ad':
        adata = st.session_state.adata  # load AnnData from session

        # === User Parameters ===
        st.markdown('<p style="color: grey;">Filter out cells with too few genes and genes expressed in too few cells to remove noise and low-quality entries.</p>', unsafe_allow_html=True)
        min_genes = st.number_input("Minimum genes per cell", min_value=0, value=100, step=50, key="min_genes_preproc")
        min_cells = st.number_input("Minimum cells per gene", min_value=0, value=3, step=1, key="min_cells_preproc")
       
        st.markdown('<p style="color: grey;">Some gene prefixes (e.g., mitochondrial or ribosomal genes) can introduce bias; optionally remove them.</p>', unsafe_allow_html=True)
        remove_prefixes = st.multiselect(
            "Remove genes with prefixes:",
            options=['ERCC', 'MT-', 'mt-', 'RPS', 'RPL', 'HB', 'HSP', 'IG'],
            default=['ERCC', 'MT-', 'mt-'],
            key="remove_prefixes"
        )

        # === Batch Correction Option ===
        st.markdown('<p style="color: grey;">Batch correction accounts for technical variation across different experimental runs.</p>', unsafe_allow_html=True)
        batch_key = None
        if "batch" in adata.obs.columns:
            batch_key = st.selectbox("Batch key for batch correction", options=["None"] + list(adata.obs.columns), index=1, key="batch_key_select")
            if batch_key == "None":
                batch_key = None

        # === Optional Steps ===
        st.markdown('<p style="color: grey;">Normalization and log transformation are standard steps to make expression values comparable across cells.</p>', unsafe_allow_html=True)
        do_normalize = st.checkbox("Normalize Total Counts", value=True)
        do_log = st.checkbox("Log1p Transform", value=True)

        st.markdown('<p style="color: grey;">Scaling ensures that all genes contribute equally to downstream analyses like PCA.</p>', unsafe_allow_html=True)
        do_scaling = st.checkbox("Scale Data", value=True)

        st.markdown('<p style="color: grey;">Highly Variable Genes (HVGs) capture the most informative features and reduce dimensionality.</p>', unsafe_allow_html=True)
        do_hvg = st.checkbox("Select Highly Variable Genes", value=True)

        st.markdown('<p style="color: grey;">Dimensionality reduction helps visualize high-dimensional data in 2D space.</p>', unsafe_allow_html=True)
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
                st.markdown('<p style="color: grey;">Recalculating quality control metrics to reflect filtered and normalized data.</p>', unsafe_allow_html=True)
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

                # Save back
                st.session_state.adata = adata.copy()

                st.success("✅ Preprocessing completed!")

                st.markdown(f"**Proceed to the next tab to see the changes visually!**")

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
            ax6.set_title(f"{selected_gene} Expression (Filtered)")
            col2.pyplot(fig6)
            col2.markdown(
                '<p style="color: grey;">Distribution of expression levels for the selected gene in filtered data.</p>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please upload and preprocess data first.")


with tabs[3]:
    st.header("🧠 Algorithms & DEG Analysis")

    if "adata" in st.session_state:
        adata = st.session_state.adata

        dim_red_method = st.selectbox("Dimensionality Reduction Method", ["UMAP", "t-SNE"])
        use_3d = st.checkbox("Use 3D projection")

        # Make sure PCA is done first
        if "X_pca" not in adata.obsm:
            with st.spinner("Running PCA..."):
                sc.tl.pca(adata)

        # Run chosen dimensionality reduction automatically
        with st.spinner(f"Running {dim_red_method}..."):
            if dim_red_method == "UMAP":
                if "neighbors" not in adata.uns:
                    sc.pp.neighbors(adata)
                sc.tl.umap(adata, n_components=3 if use_3d else 2)
                coords = adata.obsm["X_umap"]

            elif dim_red_method == "t-SNE":
                if "X_tsne" not in adata.obsm or adata.obsm["X_tsne"].shape[1] != (3 if use_3d else 2):
                    tsne = TSNE(n_components=3 if use_3d else 2, random_state=0)
                    X_pca = adata.obsm['X_pca']
                    tsne_result = tsne.fit_transform(X_pca)
                    adata.obsm['X_tsne'] = tsne_result
                coords = adata.obsm["X_tsne"]

        # Plotting
        if coords is not None:
            fig = plt.figure()
            if use_3d and coords.shape[1] == 3:
                df = pd.DataFrame(coords, columns=["Component 1", "Component 2", "Component 3"])
                fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", opacity=0.25)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',   # Transparent outer background
                    plot_bgcolor='rgba(0,0,0,0)',    # Transparent plotting area
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                        yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                        zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
                    )
                )
                fig.update_layout(title=f"{dim_red_method} (3D)")
                st.plotly_chart(fig)
            else:
                plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.6)
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.title(f"{dim_red_method} (2D)")
                st.pyplot(plt.gcf())
        else:
            st.info(f"{dim_red_method} embedding not found. Please run dimensionality reduction.")




        # BATCH CORRECTION
        st.subheader("🔁 Batch Correction")
        batch_methods = st.multiselect(
            "Select batch correction method(s):",
            ["Harmony", "BBKNN"]
        )
        if "Harmony" in batch_methods:
            st.spinner("Applying Harmony...")
            import harmonypy as hm
            ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, ['batch'])
            adata.obsm['X_pca_harmony'] = ho.Z_corr.T

        if "BBKNN" in batch_methods:
            st.spinner("Applying BBKNN...")
            bbknn.bbknn(adata, batch_key='batch')

        # DEG ANALYSIS
        st.subheader("📊 DEG Analysis")
        groupby = st.selectbox("Group by:", options=[ "celltype", "disease", "donor", "batch", "protocol"])
        groups = adata.obs[groupby].unique().tolist()
        ref_group = st.selectbox("Reference Group:", options=groups)
        comp_group = st.selectbox("Comparison Group:", options=[g for g in groups if g != ref_group])
        
        if st.button("Run DEG Analysis"):
            with st.spinner("Running differential expression..."):
                sc.tl.rank_genes_groups(adata, groupby=groupby, groups=[comp_group], reference=ref_group, method="wilcoxon")
                result = adata.uns['rank_genes_groups']
                degs_df = sc.get.rank_genes_groups_df(adata, group=comp_group)

                # Classify DEG
                degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
                degs_df["diffexpressed"] = "NS"
                degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
                degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"

                # Volcano Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=degs_df,
                    x="logfoldchanges",
                    y="neg_log10_pval",
                    hue="diffexpressed",
                    palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
                    alpha=0.7,
                    edgecolor=None,
                    ax=ax
                )
                ax.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
                ax.axvline(x=-1, color='gray', linestyle='dashed')
                ax.axvline(x=1, color='gray', linestyle='dashed')
                ax.set_xlim(-11, 11)
                ax.set_ylim(0, degs_df["neg_log10_pval"].max() + 10)
                ax.set_xlabel("log2 Fold Change", fontsize=14)
                ax.set_ylabel("-log10 p-value", fontsize=14)
                ax.set_title(f"Volcano Plot: {comp_group} vs {ref_group}", fontsize=16)
                ax.legend(title="Expression", loc="upper right")
                st.pyplot(fig)

                # Show top genes
                st.subheader("🔬 Top DEGs")
                top_up = degs_df[degs_df["diffexpressed"] == "UP"].nlargest(10, "logfoldchanges")
                top_down = degs_df[degs_df["diffexpressed"] == "DOWN"].nsmallest(10, "logfoldchanges")
                st.write("**Top Up-regulated Genes:**")
                st.dataframe(top_up[["names", "logfoldchanges", "pvals"]])
                st.write("**Top Down-regulated Genes:**")
                st.dataframe(top_down[["names", "logfoldchanges", "pvals"]])

    else:
        st.warning("Please upload and preprocess data before using this tab.")