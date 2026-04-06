Main API
============

Reference-based imputation
---------------------------

.. code-block:: python

   expTransImp(
             df_ref: pd.DataFrame, 
             df_tgt: pd.DataFrame, 
             train_gene: list, 
             test_gene: list, 
             classes: list=None, 
             ct_list: list=None,
             autocorr_method: str='moranI', 
             signature_mode: str='cluster',
             mapping_mode: str='full',
             mapping_lowdim: int=256,
             spa_adj: sparse.coo_array=None,
             lr: float=1e-2, 
             weight_decay: float=1e-2, 
             n_epochs: int=2000,
             clip_max: int=10,
             wt_spa: float=1.0,
             locations: np.array=None,
             n_simulation: int=None,
             convert_uncertainty_score: bool=True,
             device: torch.device=None,
             seed: int=None)
    """Main function for transimp

    Args:
        df_ref (pd.DataFrame): Dataframe of reference single cell
        df_tgt (pd.DataFrame): Dataframe of ST for training
        train_gene (list): Training genes
        test_gene (list):  Genes for ST prediction, should be in df_ref
        classes (list, optional): Single-cell type annotations. Defaults to None.
        ct_list (list, optional): List of cell type labels. Defaults to None.
        autocorr_method (str, optional): Autocorrelation method. Defaults to 'moranI'.
        signature_mode (str, optional): Mode for creating compressed signature. Defaults to 'cluster'.
        mapping_mode (str, optional): 'lowrank' or 'full' mapping matrix. Defaults to 'full'.
        mapping_lowdim (int, optional): Defaults to 256.
        spa_adj (sparse.coo_array, optional): Spatial adjacency matrix. Defaults to None.
        lr (float, optional): Defaults to 1e-2.
        weight_decay (float, optional): Defaults to 1e-2.
        n_epochs (int, optional): Defaults to 1000.
        clip_max (int, optional): Defaults to 10.
        wt_spa (float, optional): Defaults to 1.0.
        locations (np.array, optional): Spatial coordinates of the ST dataset. Defaults to None.
        n_simulation (int, optional): Indicater & the number of local bootstraps for performance uncertainty estimation. Defaults to None.
        convert_uncertainty_score (bool, optional): whether or not to convert uncertainty score to certainty score with $sigmoid(-pred.var.)$, 
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        list: results
    """

TransImpLR:

.. code-block:: python

        expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            signature_mode='cell',
            mapping_mode='lowrank',
            n_epochs=2000,
            seed=seed,
            device=device
         )

TransImpLR (with Uncertainty Score):

.. code-block:: python

        expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            signature_mode='cell',
            mapping_mode='lowrank',
            n_epochs=2000,
            n_simulation=200,
            classes=classes,
            seed=seed,
            device=device
         )

TransImpCls:

.. code-block:: python

         expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            ct_list=ct_list,
            classes=classes,
            n_epochs=2000,
            signature_mode='cluster',
            mapping_mode='full',
            seed=seed,
            device=device
        )

TransImpSpa:

.. code-block:: python

         expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            signature_mode='cell',
            mapping_mode='lowrank',
            n_epochs=2000,
            spa_adj=spa_adata.obsp['spatial_connectivities'].tocoo(),
            seed=seed,
            device=device
        )

TransImpClsSpa:

.. code-block:: python

         expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            ct_list=ct_list,
            classes=classes,
            spa_adj=spa_adata.obsp['spatial_connectivities'].tocoo(),
            signature_mode='cluster',
            mapping_mode='full',
            wt_spa=0.1,
            n_epochs=2000,
            seed=seed,
            device=device
        )



Cell-type deconvolution
---------------------------

.. code-block:: python

    expDeconv(adata_ref: sc.AnnData=None,
              adata_tgt: sc.AnnData=None,
              label_key: str='Class',
              df_ref: pd.DataFrame=None,
              df_tgt: pd.DataFrame=None,
              classes: np.array=None,
              ct_list: np.array=None,
              lr: float=1e-2,
              weight_decay: float=1e-3,
              tau: float=None,
              n_epochs: int=8000,
              n_top_genes: int=2000,
              topk: int=50,
              wt_spa: float=1.0,
              wt_l1: float=5.0,
              wt_abd: float=0.5,
              wt_l2_G: float=2.0,
              wt_l2_S: float=2.0,
              wt_js: float=2.0,
              autocorr_method: str='moranI',
              spa_adj: sparse.coo_array=None,
              spa_adata: sc.AnnData=None,
              calibrate: float=0.0,
              gene_mask: pd.DataFrame=None,
              normalize_sig: bool=False,
              raw_counts: bool=None,
              smart_markers: bool=False,
              spatial_markers: bool=False,
              score_init: bool=True,
              cluster_mapping: bool=True,
              cosine_lr: bool=True,
              device: torch.device=None,
              seed: int=None)
    """Cell type deconvolution.

    Fits a linear translation model from reference cell-type gene signatures
    to spatial gene profiles, and returns predicted per-spot cell-type weights.

    Data can be provided in two ways:
      - AnnData mode: pass adata_ref and adata_tgt. The function extracts
        df_ref, df_tgt, classes, and ct_list automatically using label_key.
      - DataFrame mode (legacy): pass df_ref, df_tgt, classes, and ct_list
        explicitly.

    Args:
        adata_ref (sc.AnnData, optional): Reference scRNA-seq AnnData.
        adata_tgt (sc.AnnData, optional): Spatial transcriptomics AnnData.
        label_key (str): Column in adata_ref.obs for cell-type labels.
        df_ref, df_tgt, classes, ct_list: Legacy DataFrame-mode inputs.
        lr, weight_decay: Optimiser hyperparameters.
        tau (float, optional): Softmax temperature.
        n_epochs (int): Training epochs.
        n_top_genes (int): Variable genes (ignored when topk is set).
        topk (int): Marker genes per cell type via DE test.
        wt_spa (float): Spatial regularization weight.
        wt_l1 (float): L1 regularization weight on translation matrix.
        wt_abd (float): Abundance signature loss weight.
        wt_l2_G (float): L2 regularization weight on gene scalers.
        wt_l2_S (float): L2 regularization weight on spot scalers.
        wt_js (float): Jensen-Shannon divergence loss weight.
        autocorr_method (str): 'moranI' or 'gearyC'.
        spa_adj: Spatial adjacency matrix.
        spa_adata: Spatial AnnData for Leiden clustering.
        calibrate (float): Post-hoc calibration strength.
        gene_mask: Boolean mask [n_types x n_genes].
        normalize_sig (bool): L2-normalize cell-type signatures.
        raw_counts (bool, optional): Auto-detected if None.
        smart_markers (bool): Abundance-aware marker selection.
        spatial_markers (bool): Augment with spatial-derived markers.
        score_init (bool): sc.tl.score_genes warm-start.
        cluster_mapping (bool): Cluster-to-celltype mapping regularization.
        cosine_lr (bool): Use cosine annealing learning rate schedule.
        device: Torch device.
        seed: Random seed.

    Returns:
        np.array, np.ndarray: predicted ST expression, weight matrix
    """

Recommended configuration (TransDeconvV2):

.. code-block:: python

        expDeconv(
            adata_ref=adata_ref,
            adata_tgt=adata_tgt,
            label_key='Class', # change to the cell type key in reference adata.obs
            score_init=True,
            cluster_mapping=True,
            topk=50,
            cosine_lr=True,
            weight_decay=1e-3,
            wt_abd=0.5,
            wt_js=2.0,
            n_epochs=8000,
            seed=seed,
            device=device
        )

Legacy DataFrame mode:

.. code-block:: python

        expDeconv(
            df_ref=df_ref,
            df_tgt=df_tgt,
            classes=classes,
            ct_list=ct_list,
            n_epochs=8000,
            seed=seed,
            device=device
        )

ST Velocity estimation
---------------------------

.. code-block:: python

    expVeloImp(df_ref: pd.DataFrame, 
               df_tgt: pd.DataFrame,
               S: np.array, 
               U: np.array, 
               V: np.array, 
               train_gene: list, 
               test_gene: list, 
               classes: list=None, 
               ct_list: list=None,
               autocorr_method: str='moranI', 
               signature_mode: str='cell',
               mapping_mode: str='lowrank',
               mapping_lowdim: int=256,
               spa_adj: sparse.coo_array=None,
               lr: float=1e-2, 
               weight_decay: float=1e-2, 
               n_epochs: int=1000,
               clip_max: int=10,
               wt_spa: float=1.0,
               locations: np.array=None,
               n_simulation: int=None,
               device: torch.device=None,
               seed: int=None)
    """ST Velocity estimation

    Args:
        df_ref (pd.DataFrame): Dataframe of reference single cell
        df_tgt (pd.DataFrame): Dataframe of ST for training
        S (np.array): Spliced expression matrix
        U (np.array): Unspliced expression matrix
        V (np.array): SC velocity matrix
        train_gene (list): Training genes
        test_gene (list):  Genes for ST prediction, should be in df_ref
        classes (list, optional): Single-cell type annotations. Defaults to None.
        ct_list (list, optional): List of cell type labels. Defaults to None.
        autocorr_method (str, optional): Autocorrelation method. Defaults to 'moranI'.
        signature_mode (str, optional): Mode for creating compressed signature. Defaults to 'cell'.
        mapping_mode (str, optional): 'lowrank' or 'full' mapping matrix. Defaults to 'lowrank'.
        mapping_lowdim (int, optional): Defaults to 256.
        spa_adj (sparse.coo_array, optional): Spatial adjacency matrix. Defaults to None.
        lr (float, optional): Defaults to 1e-2.
        weight_decay (float, optional): Defaults to 1e-2.
        n_epochs (int, optional): Defaults to 1000.
        clip_max (int, optional): Defaults to 10.
        wt_spa (float, optional): Defaults to 1.0.
        locations (np.array, optional): Spatial coordinates of the ST dataset. Defaults to None.
        n_simulation (int, optional): Indicater & the number of local bootstraps for performance uncertainty estimation. Defaults to None.
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        tuple(np.array): ST results
    """

example:

.. code-block:: python

    expVeloImp(
        df_ref=raw_scrna_df,
        df_tgt=raw_spatial_df,
        S=RNA.layers['spliced'],
        U=RNA.layers['unspliced'],
        V=RNA.layers['spliced'],
        train_gene=raw_shared_gene,
        test_gene=RNA.var_names,
        signature_mode='cell',
        mapping_mode='lowrank',
        classes='celltype_prediction',
        n_epochs=1000,
        seed=seed,
        device=device
    )

