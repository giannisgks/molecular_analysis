import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def show():
    
    st.header("Upload Your Data")

    uploaded_file = st.file_uploader("", type=["h5ad"], key='pulse')    

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
