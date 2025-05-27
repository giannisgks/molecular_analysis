import streamlit as st
import scanpy as sc
import numpy as np

def show():
    st.header("Preprocessing of the data")

    if "adata" in st.session_state:
        adata = st.session_state.adata  # Load AnnData from session

        st.markdown('<p style="color: grey;">Set preprocessing parameters below.</p>', unsafe_allow_html=True)

        # --- First Row: Filtering & Batch ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_genes = st.number_input("Minimum genes per cell", min_value=0, value=100, step=50, key="min_genes_preproc")

        with col2:
            min_cells = st.number_input("Minimum cells per gene", min_value=0, value=3, step=1, key="min_cells_preproc")

        with col3:
            remove_prefixes = st.multiselect(
                "Remove genes with prefixes:",
                options=['ERCC', 'MT-', 'mt-', 'RPS', 'RPL', 'HB', 'HSP', 'IG'],
                default=['ERCC', 'MT-', 'mt-'],
                key="remove_prefixes"
            )

        with col4:
            batch_key = None
            if "batch" in adata.obs.columns:
                batch_key = st.selectbox(
                    "Batch key for correction",
                    options=["None"] + list(adata.obs.columns),
                    index=1,
                    key="batch_key_select"
                )
                if batch_key == "None":
                    batch_key = None

        st.markdown('<hr>', unsafe_allow_html=True)

        # --- Second Row: Preprocessing Checkboxes ---
        col1, col2, col3, col4, col5, col6= st.columns(6)

        with col1:
            do_normalize = st.checkbox("Normalize Total Counts", value=True)
        with col2:
            do_log = st.checkbox("Log1p Transform", value=True)
        with col3:
            do_scaling = st.checkbox("Scale Data", value=True)
        with col4:
            do_hvg = st.checkbox("Select HVGs", value=True)
        with col5:
            do_pca = st.checkbox("Run PCA & UMAP", value=True)
        with col6:
            do_leiden = st.checkbox("Run Leiden Clustering", value=False)

        st.markdown('<hr>', unsafe_allow_html=True)

        # --- Functions ---
        summary_metrics = {}

        def log_step_diff(prev_adata, curr_adata, step_label):
            cells_removed = prev_adata.n_obs - curr_adata.n_obs
            genes_removed = prev_adata.n_vars - curr_adata.n_vars

            msg_parts = []
            if cells_removed > 0:
                msg_parts.append(f"**{cells_removed} cells removed**")
            if genes_removed > 0:
                msg_parts.append(f"**{genes_removed} genes removed**")

            if msg_parts:
                st.markdown(f"**{step_label}:** " + ", ".join(msg_parts))

        # --- Run Button ---
        if st.button("run preprocessing", key="glitch"):
            loader = st.empty()

            with loader.container():
                loader.markdown("""
                <div class="pl">
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__dot"></div>
                <div class="pl__text">Loading…</div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("Calculating initial QC metrics..."):
                st.session_state.adata_raw = adata.copy()
                sc.pp.calculate_qc_metrics(st.session_state.adata_raw, qc_vars=["mt"], inplace=True)

            # Step 1: Filter
            with st.spinner("Filtering cells and genes..."):
                prev = adata.copy()
                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)
                log_step_diff(prev, adata, "Filtering")

            # Step 2: Remove gene prefixes
            with st.spinner("Removing gene prefixes..."):
                prev = adata.copy()
                adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(remove_prefixes))]]
                log_step_diff(prev, adata, "Prefix Removal")

            # Step 3: Normalize
            if do_normalize:
                with st.spinner("Normalizing total counts..."):
                    sc.pp.normalize_total(adata, target_sum=1e4)

            # Step 4: Log transform
            if do_log:
                with st.spinner("Applying log1p transform..."):
                    sc.pp.log1p(adata)

            # Step 5: HVG
            if do_hvg:
                with st.spinner("Selecting highly variable genes..."):
                    prev = adata.copy()
                    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                    adata.raw = adata.copy()
                    adata = adata[:, adata.var.highly_variable]

                    # ✅ Recalculate obs metrics based on subsetted genes
                    if isinstance(adata.X, np.ndarray):
                        adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1)
                        adata.obs["total_counts"] = adata.X.sum(axis=1)
                    else:
                        adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1).A1
                        adata.obs["total_counts"] = adata.X.sum(axis=1).A1

                    log_step_diff(prev, adata, "HVG Selection")
            else:
                adata.raw = adata.copy()

            # Step 6: Scaling
            if do_scaling:
                with st.spinner("Scaling the data..."):
                    sc.pp.scale(adata, max_value=10)

            # Step 7: PCA & UMAP
            if do_pca:
                with st.spinner("Running PCA and UMAP..."):
                    sc.pp.pca(adata)
                    sc.pp.neighbors(adata)
                    sc.tl.umap(adata)

            # Run Leiden clustering if checked
            if do_leiden:
                with st.spinner("Running Leiden clustering..."):
                    sc.tl.leiden(adata, resolution=1.0)
                    st.markdown(f"Leiden clustering done. Found **{adata.obs['leiden'].nunique()}** clusters.")

            # Step 8: Batch Correction
            if batch_key and batch_key in adata.obs.columns:
                with st.spinner("Applying batch correction (ComBat)..."):
                    sc.pp.combat(adata, key=batch_key)

                                    # ✅ Recalculate per-cell metrics
                if adata.raw is not None:
                    if isinstance(adata.X, np.ndarray):
                        adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1)
                        adata.obs["total_counts"] = adata.X.sum(axis=1)
                    else:
                        adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1).A1
                        adata.obs["total_counts"] = adata.X.sum(axis=1).A1

            # Save processed data
            st.session_state.adata = adata.copy()
            loader.empty()
            st.success("✅ Preprocessing completed!")

            st.markdown("**Proceed to the next tab to see the changes visually.**")
    else:
        st.markdown("No data availabe")