import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show():
    st.header("Algorithms & DEG Analysis")

    if "adata" not in st.session_state:
        st.warning("Please upload and preprocess data before using this tab.")
        st.stop()

    else:
        adata = st.session_state.adata
        col1, col2 = st.columns(2)
        with col1:
            # Dimensionality Reduction controls
            dim_red_method = st.selectbox("Dimensionality Reduction Method", ["UMAP", "t-SNE"])
            st.markdown(
                "<span style='color: grey;'>Reduces high-dimensional gene expression data to 2D or 3D for visualization and pattern discovery.</span>",
                unsafe_allow_html=True
            )
        with col2:

            # Let user choose coloring variable if available
            color_options = []
            if "batch" in adata.obs.columns:
                color_options.append("batch")
            if "celltype" in adata.obs.columns:
                color_options.append("celltype")

            if color_options:
                color_by = st.selectbox("Color by:", options=color_options)
            else:
                color_by = None
        
        use_3d = st.checkbox("Use 3D projection")

        # Button to trigger dimensionality reduction and plotting
        if st.button("Run Dimensionality Reduction and Plot", key='rotate3d'):

            # Run PCA if needed
            if "X_pca" not in adata.obsm:
                with st.spinner("Running PCA..."):
                    sc.tl.pca(adata)

            coords = None
            with st.spinner(f"Running {dim_red_method}..."):
                if dim_red_method == "UMAP":
                    if "neighbors" not in adata.uns:
                        sc.pp.neighbors(adata)
                    sc.tl.umap(adata, n_components=3 if use_3d else 2)
                    coords = adata.obsm["X_umap"]

                elif dim_red_method == "t-SNE":
                    from sklearn.manifold import TSNE
                    if "X_tsne" not in adata.obsm or adata.obsm["X_tsne"].shape[1] != (3 if use_3d else 2):
                        tsne = TSNE(n_components=3 if use_3d else 2, random_state=0)
                        tsne_result = tsne.fit_transform(adata.obsm['X_pca'])
                        adata.obsm['X_tsne'] = tsne_result
                    coords = adata.obsm["X_tsne"]

            # Plot embedding
            if coords is not None:
                if use_3d and coords.shape[1] == 3:
                    df = pd.DataFrame(coords, columns=["Component 1", "Component 2", "Component 3"])
                    import plotly.express as px
                    fig = px.scatter_3d(
                        df,
                        x="Component 1", y="Component 2", z="Component 3",
                        color=adata.obs[color_by] if color_by else None,
                        opacity=0.7,
                    )
                    fig.update_layout(title=f"{dim_red_method} (3D)")
                    st.plotly_chart(fig)
                else:
                    fig, ax = plt.subplots()
                    if color_by and color_by in adata.obs.columns:
                        if adata.obs[color_by].dtype.name == "category":
                            colors = adata.obs[color_by].cat.codes
                        else:
                            # fallback if not category dtype
                            colors = pd.factorize(adata.obs[color_by])[0]
                    else:
                        colors = 'b'

                    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7, c=colors)
                    ax.set_facecolor('none')          # Make axes background transparent
                    fig.patch.set_alpha(0)            # Make figure background fully transparent

                    # Optionally remove axes spines and ticks for cleaner look
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.set_title(f"{dim_red_method} (2D)", color="white")
                    st.pyplot(fig)

                st.caption(
                    f"The axes for {dim_red_method} are abstract and non-interpretable — they don't represent specific biological features, "
                    "but rather summarize high-dimensional variation in the data."
                )
            else:
                st.info(f"{dim_red_method} embedding not found. Please run dimensionality reduction.")

        # Batch Correction Section
        st.subheader("Batch Correction")
        batch_methods = st.multiselect(
            "Select batch correction method(s):",
            ["Harmony", "BBKNN"]
        )
        st.markdown(
            "<span style='color: grey;'>Batch correction fixes differences in the data caused by technical artifacts.</span>",
            unsafe_allow_html=True
        )

        if st.button("Apply Batch Correction", key='io-button'):
            if "batch" not in adata.obs.columns:
                st.error("No 'batch' column found in metadata for batch correction.")
            else:
                if "Harmony" in batch_methods:
                    import harmonypy as hm
                    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, ['batch'])
                    adata.obsm['X_pca_harmony'] = ho.Z_corr.T
                    st.success("Harmony batch correction applied.")

                if "BBKNN" in batch_methods:
                    import bbknn
                    bbknn.bbknn(adata, batch_key='batch')
                    st.success("BBKNN batch correction applied.")

# DEG ANALYSIS
        valid_groupby_options = [
            col for col in adata.obs.columns 
            if adata.obs[col].nunique() > 1 and adata.obs[col].dtype.name in ["category", "object"]
]

        if not valid_groupby_options:
            st.error("❌ No suitable categorical columns found for DEG analysis.")
            st.stop()

        groupby = st.selectbox("Group by:", options=valid_groupby_options)
        st.subheader("📊 DEG Analysis")
        groupby = st.selectbox("Group by:", options=["celltype", "disease", "donor", "batch", "protocol"])

        # Check column existence
        if groupby not in adata.obs.columns:
            st.error(f"❌ The selected column '{groupby}' is not found in the dataset. Please choose a different one.")
            st.stop()

        groups = adata.obs[groupby].unique().tolist()
        ref_group = st.selectbox("Reference Group:", options=groups)
        comp_group = st.selectbox("Comparison Group:", options=[g for g in groups if g != ref_group])

        st.markdown("<span style='color: grey;'>Select a group to compare against the reference group using statistical tests.</span>", unsafe_allow_html=True)

        if st.button("Run DEG Analysis", key='glow-on-hover'):
            with st.spinner("Running differential expression..."):
                sc.tl.rank_genes_groups(adata, groupby=groupby, groups=[comp_group], reference=ref_group, method="wilcoxon")
                result = adata.uns['rank_genes_groups']
                degs_df = sc.get.rank_genes_groups_df(adata, group=comp_group)
                if "pvals" in degs_df.columns and "logfoldchanges" in degs_df.columns:
                    degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
                    degs_df["diffexpressed"] = "NS"
                    degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
                    degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"
                else:
                    st.error("DEG results missing required columns. Please check input data.")
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
                st.markdown("<span style='color: grey;'>Identifies genes significantly up- or down-regulated between groups, useful for biomarker discovery.</span>", unsafe_allow_html=True)
                st.subheader("Top DEGs")
                top_up = degs_df[degs_df["diffexpressed"] == "UP"].nlargest(10, "logfoldchanges")
                top_down = degs_df[degs_df["diffexpressed"] == "DOWN"].nsmallest(10, "logfoldchanges")
                st.write("**Top Up-regulated Genes:**")
                st.dataframe(top_up[["names", "logfoldchanges", "pvals"]])
                st.write("**Top Down-regulated Genes:**")
                st.dataframe(top_down[["names", "logfoldchanges", "pvals"]])
