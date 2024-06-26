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



Cell-type deconvoluation
---------------------------

.. code-block:: python

    expDeconv(df_ref: pd.DataFrame, 
              df_tgt: pd.DataFrame, 
              classes: np.array, 
              ct_list: np.array,
              lr: float=1e-2, 
              weight_decay: float=1e-2, 
              n_epochs: int=1000,
              n_top_genes: int=2000,
              device: torch.device=None,
              seed: int=None)
    """

    Args:
        df_ref (pd.DataFrame): Single cell reference dataframe
        df_tgt (pd.DataFrame): ST dataframe
        classes (np.array): cell type annotations for single cell
        ct_list (np.array): cell type label list
        lr (float, optional): Defaults to 1e-3.
        weight_decay (float, optional): Defaults to 1e-2.
        n_epochs (int, optional): Number of epochs for fitting. Defaults to 1000.
        n_top_genes (int, optional): Number of top variable genes. Defaults to 2000.
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        np.array, np.ndarray: predicted ST cell type, alignment matrix
    """

example:

.. code-block:: python

        expDeconv(
            df_ref=df_ref,
            df_tgt=df_tgt,
            classes=classes,
            ct_list=ct_list,
            n_epochs=8000,
            lr=1e-2,
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

