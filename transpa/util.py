import torch

import numpy as np
import scanpy as sc
import squidpy as sq
import pandas as pd

from scipy import sparse
from scipy.special import expit
from torchmetrics.functional.regression import cosine_similarity
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .model import TransDeconv, TransImp, SpaAutoCorr, SparkX

def plot_genes(genes, spa_adata, df_corr=None, is_I=False, n_cols=5, dpi=380, figsize=(20, 20)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams.update({"font.size":30, 'axes.titlesize':30})
    exprs = [spa_adata[:, gene].X.toarray().flatten() for gene in genes]
    # titles = [f'\n {gene} M.I. {spa_adata.uns["moranI"].loc[gene].I:.3f}; S.R. {df_corr.loc[gene, "cos_sim"]:.3f}' for gene in genes]
    if is_I:
        titles = [f'{gene}\n M.I. {spa_adata.uns["moranI"].loc[gene].I:.3f}' for gene in genes]
    else:
        titles = [f'{gene}\n C.S. {df_corr.loc[gene, "cos_sim"]:.3f}' for gene in genes]
    # sc.pl.spatial(spa_adata, color=[gene], spot_size=0.1, title='Truth')
    print(spa_adata.uns['moranI'].loc[genes])
    
    tmp_adata = sc.AnnData(np.array(exprs).T)
    tmp_adata.var_names = titles
    tmp_adata.obsm['spatial'] = spa_adata.obsm['spatial']
    sc.pl.spatial(tmp_adata, color=titles, spot_size=0.1, title=titles, color_map='OrRd', legend_fontsize=10, hspace=0.5, wspace=0.0001, ncols=n_cols)

def compute_autocorr(spa_adata: sc.AnnData, 
                     df: pd.DataFrame, 
                     n_jobs: int=10, 
                     mode: str='moran'):
    """Compute spatial autocorrelation

    Args:
        spa_adata (sc.AnnData): the ST AnnData
        df (pd.DataFrame): table of predicted expressions 
        n_jobs (int, optional): Number of jobs. Defaults to 10.
        mode (str, optional): spatial autocorrelation mode. Defaults to 'moran'.

    Returns:
        sc.AnndData: adata with spatial autocorrelation statistics
    """
    imputed_adata = spa_adata.copy()
    imputed_adata.X = df[imputed_adata.var_names].values
    sq.gr.spatial_autocorr(
        imputed_adata,
        genes=imputed_adata.var_names,
        n_jobs=n_jobs,
        mode=mode
    )
    return imputed_adata

def leiden_cluster(adata: sc.AnnData, 
                   normalize: bool=True):
    """Clustering with Leiden method

    Args:
        adata (sc.AnnData): Adata object
        normalize (bool, optional): Whether or not normalize matrix. Defaults to True.

    Returns:
        tuple[(predictions, labels)]: 
    """
    adata_cp = adata.copy()
    if normalize:
        sc.pp.normalize_total(adata_cp)
        sc.pp.log1p(adata_cp)
        sc.pp.highly_variable_genes(adata_cp)
        adata_cp = adata_cp[:, adata_cp.var.highly_variable]
    sc.pp.scale(adata_cp, max_value=10)
    sc.tl.pca(adata_cp)
    sc.pp.neighbors(adata_cp)
    sc.tl.leiden(adata_cp, resolution = 0.5)
    return adata_cp.obs.leiden, np.unique(adata_cp.obs.leiden)

def sparse_vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    return a.power(2).mean(axis) - np.square(a.mean(axis))


def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    if sparse.issparse(data_mtx):
        var = np.asarray(sparse_vars(data_mtx, axis=0)).flatten()
    else:
        var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind


def tensify(X, 
            device: str=None, 
            is_dense: bool=True):
    """Tensify input matrix

    Args:
        X (np.array|sparse matrix):
        device (str, optional): Defaults to None.
        is_dense (bool, optional): Defaults to True.

    Returns:
        Tensor
    """
    # X is dense or a sparse matrix
    if is_dense:
        return torch.FloatTensor(X).to(device)
    else:
        X = X.tocoo()
        return torch.sparse_coo_tensor(
                                indices=np.array([X.row, X.col]),
                                values=X.data,
                                size=X.shape).float().to(device)


def signature(classes: np.array, 
              ct_list: np.array, 
              expr_mtx: np.array):
    """Generate gene signatures by aggregation expression matrix
       based on cell types.

    Args:
        classes (np.array): cell type annotation
        ct_list (np.array): available cell type labels
        expr_mtx (np.array): expression matrix

    Returns:
        tensor, tensor: gene signature, class expression abundence
    """

    g_cls_sig = np.vstack([np.sum(expr_mtx[classes == cls], axis=0, keepdims=True) for cls in ct_list])
    cls_abd_sig = np.array([(classes == cls).sum() for cls in ct_list]).reshape(-1, 1)
    return g_cls_sig, cls_abd_sig

def train_deconv_step(optimizer, model, X, Y, cls_abd_sig, wt_spa=1.0,
                   truth_autocorr=None, method_autocorr='moranI'):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(X, Y, cls_abd_sig, 
                      truth_autocorr=truth_autocorr, 
                      wt_spa=wt_spa,
                      method_autocorr=method_autocorr
                      )
    loss.backward()
    optimizer.step()
    info = f'loss: {loss.item():.6f}'
    return info

def get_spa_laplacian(locations, n_nbs, rbf_gamma, device=None):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csgraph

    nnb = NearestNeighbors(n_neighbors=n_nbs)
    nnb.fit(locations)
    dists, indices = nnb.kneighbors(locations)
    weights = np.exp(-rbf_gamma * np.square(dists))
    kernel = sparse.csr_matrix((weights.flatten(), 
                                (np.arange(weights.shape[0]).repeat(weights.shape[1]), 
                                 indices.flatten())
                                ), 
                               shape=(locations.shape[0], locations.shape[0]))
    L = csgraph.laplacian(kernel).tocsr()
    rowids, colids = L.nonzero()
    values = L[(rowids, colids)]
    values = values.toarray().flatten() if sparse.issparse(values) else np.asarray(values).flatten()
    coords = np.array([rowids, colids])
    L = torch.sparse_coo_tensor(indices=torch.LongTensor(coords), 
                                values=torch.FloatTensor(values),
                                size=L.shape,
                                device=device
                                )
    return L

def fit_deconv(
            df_ref: pd.DataFrame, 
            df_tgt: pd.DataFrame, 
            lr: float, 
            weight_decay: float, 
            n_epochs: int,
            classes: np.array,
            ct_list: np.array,
            n_top_genes: int,
            wt_spa: float,
            tau: float=0.5,
            autocorr_method: str='moranI',
            spa_adj: sparse.coo_array=None,
            device: torch.device=None,
            seed: int=None):
    indices = select_top_variable_genes(df_ref.values, n_top_genes)
    X = df_ref.values[:, indices]
    Y = df_tgt.values[:, indices]

    # is class by gene
    g_cls_sig, cls_abd_sig = signature(classes, ct_list, X)
    # g_cls_sig = np.vstack([g_cls_sig, np.zeros((1, g_cls_sig.shape[1]))])
    X, Y = tensify(g_cls_sig, device), tensify(Y, device)
    cls_abd_sig = tensify(cls_abd_sig, device)
    
    if not spa_adj is None:
        spa_adj = torch.sparse_coo_tensor(indices=np.array([spa_adj.row, spa_adj.col]),
                                                values=spa_adj.data,
                                                size=spa_adj.shape).to(device).float()

    model = TransDeconv(
                 dim_tgt_outputs=Y.shape[0],
                 n_feats=len(indices),
                 dim_ref_inputs=X.shape[0],
                 tau = tau,
                 spa_autocorr=None if spa_adj is None else SpaAutoCorr(spa_adj),
                 device=device,
                 seed=seed).to(device)

    if not spa_adj is None:
        with torch.no_grad():
            truth_autocorr = model.spa_autocorr.cal_spa_stats(Y, autocorr_method)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)        
    pbar = tqdm(range(n_epochs))

    for ith_epoch in pbar:
        info  = train_deconv_step(optimizer, model, X, Y, cls_abd_sig, wt_spa,
                                  None if spa_adj is None else truth_autocorr,
                                  autocorr_method
                                  )
        pbar.set_description(f"[LinTrans] Epoch: {ith_epoch+1}/{n_epochs}, {info}")    
    return model, X, Y

def expDeconv(df_ref: pd.DataFrame, 
              df_tgt: pd.DataFrame, 
              classes: np.array, 
              ct_list: np.array,
              lr: float=1e-3, 
              weight_decay: float=1e-2, 
              tau: float=1.0,
              n_epochs: int=1000,
              n_top_genes: int=2000,
              wt_spa: float=1.0,
              autocorr_method: str='moranI',
              spa_adj:sparse.coo_array=None,
              device: torch.device=None,
              seed: int=None):
    """Cell type deconvalution

    Args:
        df_ref (pd.DataFrame): Single cell reference dataframe
        df_tgt (pd.DataFrame): ST dataframe
        classes (np.array): cell type annotations for single cell
        ct_list (np.array): cell type label list
        lr (float, optional): Defaults to 1e-3.
        weight_decay (float, optional): Defaults to 1e-2.
        tau (float, optional): Defaults to 1.0, softmax temperature for mapping matrix, 
                               1.0 no effect,
                               < 1.0, the smaller the sharper the softmaxed distribution becomes
                               > 1.0, the larger the more even the softmaxed distribution becomes
                               None: turn-off, use the old method
        n_epochs (int, optional): Number of epochs for fitting. Defaults to 1000.
        n_top_genes (int, optional): Number of top variable genes. Defaults to 2000.
        wt_spa (float, optional): Weight of spatial regularization. Defaults to 1.0.
        autocorr_method (str, optional): Defaults to 'moranI'.
        spa_adj (sparse.coo_array, optional): Spatial adjacency matrix. Defaults to None.
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        np.array, np.ndarray: predicted ST cell type, alignment matrix
    """
    if not n_top_genes is None and n_top_genes > 0:
        n_top_genes = min(n_top_genes, min(df_ref.shape[1], df_tgt.shape[1]))
            
    model, X, Y = fit_deconv(
                            df_ref, df_tgt,
                            lr, weight_decay, n_epochs,
                            classes,
                            ct_list,
                            n_top_genes,
                            tau=tau,
                            wt_spa=wt_spa,
                            autocorr_method=autocorr_method,
                            spa_adj=spa_adj,                            
                            device=device,
                            seed=seed) 
    with torch.no_grad():
        model.eval()
        preds, weights = model.predict(X, return_cluster=True)
    return preds, weights

def train_imp_step(optimizer, model, X, Y, wt_spa=0.1, wt_l1norm=1e-2, wt_l2norm=1e-2,
                   truth_spa_stats=None):
    model.train()
    optimizer.zero_grad()
    loss, imp_loss, spa_reg = model.loss(X, Y, truth_spa_stats=truth_spa_stats, 
                            wt_l2norm=wt_l2norm, wt_l1norm=wt_l1norm, wt_spa=wt_spa)
    loss.backward()
    optimizer.step()
    info = f'loss: {loss.item():.6f}, (IMP) {imp_loss:.6f}' 
    if not model.spa_inst is None:
        info += f', (SPA) {wt_spa} x {spa_reg:.6f}'
    return info


def fit_transImp(
            df_ref: pd.DataFrame, 
            df_tgt: pd.DataFrame, 
            train_gene: list, 
            test_gene: list,
            lr: float, 
            weight_decay: float, 
            n_epochs: int,
            classes: list,
            ct_list: list,
            autocorr_method: SpaAutoCorr,
            mapping_mode: str,
            mapping_lowdim: int,
            spa_adj: sparse.coo_array,
            clip_max: int=10,
            signature_mode: str='cluster',
            wt_spa: float=1e-1,
            wt_l1norm: float=None,
            wt_l2norm: float=None,
            locations: np.array=None,
            device: torch.device=None,
            seed: int=None):
        
    X = df_ref[train_gene].values
    Y = df_tgt[train_gene].values
    # is class by gene
    Y = tensify(Y, device)
    if signature_mode == 'cluster':
        g_cls_sig, _ = signature(classes, ct_list, X)
    
        X = tensify(g_cls_sig, device)
    else:
        X = tensify(X, device)
        
    spa_inst = None
    if not locations is None:
        locations = tensify(locations, device)
        spa_inst = SparkX(locations)

    if not spa_adj is None:
        spa_adj = torch.sparse_coo_tensor(indices=np.array([spa_adj.row, spa_adj.col]),
                                          values=spa_adj.data,
                                          size=spa_adj.shape).to(device).float()
        spa_inst = SpaAutoCorr(Y, spa_adj, method=autocorr_method)
        
    model = TransImp(
                dim_tgt_outputs=Y.shape[0],
                dim_ref_inputs=X.shape[0],
                spa_inst=spa_inst,
                mapping_mode=mapping_mode,
                dim_hid=mapping_lowdim,
                clip_max=clip_max,
                device=device,
                seed=seed).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)        
    pbar = tqdm(range(n_epochs))

    for ith_epoch in pbar:
        info  = train_imp_step(optimizer, model, X, Y, wt_spa, wt_l1norm, wt_l2norm)
        pbar.set_description(f"[TransImp] Epoch: {ith_epoch+1}/{n_epochs}, {info}") 

    if signature_mode == 'cluster':
        test_X, _ = signature(classes, ct_list, df_ref[test_gene].values)
        test_X = tensify(test_X, device)
    else:
        test_X = tensify(df_ref[test_gene].values, device)

    return model, X, Y, test_X

def infer_prediction_variance(features, train_y, n_jobs=10):
    from sklearn.preprocessing import MinMaxScaler
    st0 = np.random.get_state()
    np.random.seed()
    train_end = train_y.shape[0]
    features = MinMaxScaler().fit_transform(features)
    model = LinearRegression(n_jobs=n_jobs, fit_intercept=False)
    model = model.fit(features[:train_end], train_y)
    preds = model.predict(features)
    np.random.set_state(st0)    
    return preds[:train_end], preds[train_end:]

def estimate_uncertainty_local(model, X, classes, n_simulations=100):
    st0 = np.random.get_state()
    np.random.seed()
    sim_res = []
    classes = np.array(classes)
    for i in range(n_simulations):
        sim_X = torch.empty_like(X)
        for cls in np.unique(classes):
            cls_indices = np.argwhere(classes == cls).flatten()
            sim_indices = np.random.choice(cls_indices, cls_indices.shape[0], replace=True)
            sim_X[cls_indices] = X[sim_indices]
        preds = model.predict(sim_X)
        sim_res.append(preds)
    np.random.set_state(st0)    
    return sim_res

def estimate_performance_uncertainty(model, 
                                     train_X, 
                                     train_y, 
                                     test_X, 
                                     classes, 
                                     n_simulation, 
                                     convert_uncertainty_score, 
                                     device=None):
    X = torch.cat([train_X, test_X], dim=1)
    y = model(X)
    sim_res_lc = estimate_uncertainty_local(model,  X, classes, n_simulations=n_simulation)
                                
    train_score_var = np.var(
        np.array(
            [np.nan_to_num(
                cosine_similarity(tensify(_y[:, :train_X.shape[1]], device).t(), train_y.t(), 'none').cpu().numpy(), 
                posinf=0, 
                neginf=0) for _y in sim_res_lc]
    ), axis=0)    
    
    features = np.hstack([
        (X == 0).float().mean(dim=0).view(-1, 1).cpu().numpy(),
        torch.var(y, dim=0).view(-1, 1).cpu().numpy(),
        torch.mean(y, dim=0).view(-1, 1).cpu().numpy(),
    ])            

    hat_train_score_var, hat_test_score_var = infer_prediction_variance(features, train_score_var)
    if convert_uncertainty_score:
        hat_train_score_var, hat_test_score_var = expit(hat_train_score_var), expit(hat_test_score_var)
    return hat_train_score_var, hat_test_score_var
    
def expTransImp(
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
             n_epochs: int=1000,
             clip_max: int=10,
             wt_spa: float=1.0,
             wt_l1norm: float=None,
             wt_l2norm: float=None,
             locations: np.array=None,
             n_simulation: int=None,
             convert_uncertainty_score: bool=True,
             device: torch.device=None,
             seed: int=None):
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
        wt_l1norm (float, optional): Defaults to None.
        wt_l2norm (float, optional): Defaults to None.
        locations (np.array, optional): Spatial coordinates of the ST dataset. Defaults to None.
        n_simulation (int, optional): Indicater & the number of local bootstraps for performance uncertainty estimation. Defaults to None.
        convert_uncertainty_score (bool, optional): whether or not to convert uncertainty score to certainty score with $sigmoid(-pred.var.)$, 
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        list: results
    """
    model, train_X, train_y, test_X = fit_transImp(
                                        df_ref, df_tgt,
                                        train_gene, test_gene,
                                        lr, weight_decay, n_epochs,
                                        classes,
                                        ct_list,
                                        autocorr_method, 
                                        mapping_mode,
                                        mapping_lowdim,
                                        spa_adj,
                                        clip_max=clip_max,
                                        signature_mode=signature_mode,
                                        wt_spa=wt_spa,
                                        wt_l1norm=wt_l1norm,
                                        wt_l2norm=wt_l2norm,
                                        locations=locations,
                                        device=device,
                                        seed=seed)
    with torch.no_grad():
        model.eval()
        preds = model.predict(test_X)
        if not n_simulation is None and not classes is None:
            _, hat_test_score_var = estimate_performance_uncertainty(model, 
                                                                     train_X, 
                                                                     train_y, 
                                                                     test_X, 
                                                                     classes, 
                                                                     n_simulation, 
                                                                     convert_uncertainty_score,
                                                                     device)
            return preds, hat_test_score_var
    return preds

def expVeloImp(df_ref: pd.DataFrame, 
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
               wt_l1norm: float=None,
               wt_l2norm: float=None,
               locations: np.array=None,
               n_simulation: int=None,
               device: torch.device=None,
               seed: int=None):
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
        wt_l1norm (float, optional): Defaults to None.
        wt_l2norm (float, optional): Defaults to None.
        locations (np.array, optional): Spatial coordinates of the ST dataset. Defaults to None.
        n_simulation (int, optional): Indicater & the number of local bootstraps for performance uncertainty estimation. Defaults to None.
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.

    Returns:
        tuple(np.array): ST results
    """
    
    model, train_X, train_y, test_X = fit_transImp(
                                            df_ref, df_tgt,
                                            train_gene, test_gene,
                                            lr, weight_decay, n_epochs,
                                            classes,
                                            ct_list,
                                            autocorr_method, 
                                            mapping_mode,
                                            mapping_lowdim,
                                            spa_adj,
                                            clip_max=clip_max,
                                            signature_mode=signature_mode,
                                            wt_spa=wt_spa,
                                            wt_l1norm=wt_l1norm,
                                            wt_l2norm=wt_l2norm,
                                            locations=locations,
                                            device=device,
                                            seed=seed) 
    with torch.no_grad():
        model.eval()
        _U = model.predict(tensify(U, device, not sparse.issparse(U)))
        _S = model.predict(tensify(S, device, not sparse.issparse(U)))
        _V = model.predict(tensify(V, device, not sparse.issparse(V)) + tensify(S, device, not sparse.issparse(U))) - _S
        _X  = model.predict(test_X)
        if not n_simulation is None and not classes is None:
            _, hat_test_score_var = estimate_performance_uncertainty(model, train_X, train_y, test_X, classes, n_simulation, device)
            return _S, _U, _V, _X, hat_test_score_var
    return _S, _U, _V, _X