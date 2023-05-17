import numpy as np
import pandas as pd
import torch
import scanpy as sc

from torch import nn
import squidpy as sq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel

from tqdm import tqdm

from .model import TransDeconv, TransImp, SpaAutoCorr, SparkX, SpaReg
from .eval_util import spark_stat


def compute_autocorr(spa_adata, df):
    imputed_adata = spa_adata.copy()
    imputed_adata.X = df[imputed_adata.var_names].values
    sq.gr.spatial_autocorr(
        imputed_adata,
        genes=imputed_adata.var_names,
        n_jobs=10,
        mode='moran'
    )
    # sq.gr.spatial_autocorr(
    #     imputed_adata,
    #     n_jobs=10,
    #     mode='geary',
    # )    
    # imputed_adata.var['sparkx'] = spark_stat(imputed_adata.obsm['spatial'], imputed_adata.X)
    return imputed_adata

def leiden_cluster(adata, normalize=True):
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


def tensify(X, device=None, is_dense=True):
    # X is dense or a sparse matrix
    if is_dense:
        return torch.FloatTensor(X).to(device)
    else:
        X = X.tocoo()
        return torch.sparse_coo_tensor(
                                indices=np.array([X.row, X.col]),
                                values=X.data,
                                size=X.shape).float().to(device)


def signature(classes, ct_list, expr_mtx):
    g_cls_sig = np.vstack([np.sum(expr_mtx[classes == cls], axis=0, keepdims=True) for cls in ct_list])
    cls_abd_sig = np.array([(classes == cls).sum() for cls in ct_list]).reshape(-1, 1)
    return g_cls_sig, cls_abd_sig

def expImpute(df_ref, df_tgt, train_genes, 
             test_genes,
             ae_wt, kld_wt=1.0,
             lr=1e-3, weight_decay=1e-2, n_epochs=1000,
             dim_hid_AE=512, 
             dim_hid_RG=None,
             pca_low_dim=2000, 
             n_top_genes=2000,
             device=None,
             seed=None):
    if pca_low_dim is not None:
        pca_low_dim = min(pca_low_dim, df_ref.shape[0]-1)
    if not n_top_genes is None and n_top_genes > 0:
        n_top_genes = min(n_top_genes, df_ref.shape[1])
    model, test_X = fit_base(df_ref, df_tgt, train_genes, 
                            test_genes,
                            ae_wt, kld_wt,
                            lr, weight_decay, n_epochs,
                            dim_hid_AE, 
                            dim_hid_RG,
                            pca_low_dim, 
                            n_top_genes,
                            device=device,
                            seed=seed) 
    with torch.no_grad():
        model.eval()
        pred_Y = model.predict(test_X)
    return pred_Y


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
            df_ref, df_tgt, lr, weight_decay, 
            n_epochs,
            classes,
            ct_list,
            n_top_genes,
            wt_spa,
            autocorr_method='moranI',
            spa_adj=None,
            device=None,
            seed=None):
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

def expDeconv(df_ref, df_tgt, classes, ct_list,
             lr=1e-3, weight_decay=1e-2, n_epochs=1000,
             n_top_genes=2000,
             wt_spa=1.0,
             autocorr_method='moranI',
             spa_adj=None,
             device=None,
             seed=None):
    if not n_top_genes is None and n_top_genes > 0:
        n_top_genes = min(n_top_genes, min(df_ref.shape[1], df_tgt.shape[1]))
            
    model, X, Y = fit_deconv(
                            df_ref, df_tgt,
                            lr, weight_decay, n_epochs,
                            classes,
                            ct_list,
                            n_top_genes,
                            wt_spa=wt_spa,
                            autocorr_method=autocorr_method,
                            spa_adj=spa_adj,                            
                            device=device,
                            seed=seed) 
    with torch.no_grad():
        model.eval()
        preds, weights = model.predict(X, return_cluster=True)
    return preds, weights
    

    ############# expTransImp ###############

def train_imp_step(optimizer, model, X, Y, wt_spa=0.1, wt_l1norm=1e-2, wt_l2norm=1e-2,
                   truth_spa_stats=None):
    model.train()
    optimizer.zero_grad()
    loss, imp_loss, spa_reg = model.loss(X, Y, truth_spa_stats=truth_spa_stats, 
                            wt_l2norm=wt_l2norm, wt_l1norm=wt_l1norm, wt_spa=wt_spa)
    loss.backward()
    optimizer.step()
    info = f'loss: {loss.item():.6f}, (IMP) {imp_loss:.6f}' 
    info += f', (SPA) {wt_spa} x {spa_reg:.6f}'
    return info


def fit_transImp(
            df_ref, df_tgt, 
            train_gene, test_gene,
            lr, weight_decay, 
            n_epochs,
            classes,
            ct_list,
            autocorr_method,
            mapping_mode,
            mapping_lowdim,
            spa_adj,
            clip_max=10,
            signature_mode='cluster',
            wt_spa=1e-1,
            wt_l1norm=None,
            wt_l2norm=None,
            locations=None,
            rank_margin=0,
            device=None,
            seed=None):
        
    X = df_ref[train_gene].values
    Y = df_tgt[train_gene].values
    # is class by gene
    Y = tensify(Y, device)
    if signature_mode == 'cluster':
        g_cls_sig, _ = signature(classes, ct_list, X)
    # g_cls_sig = X
    # g_cls_sig = np.vstack([g_cls_sig, np.zeros((1, g_cls_sig.shape[1]))])
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
        # spa_inst = SpaReg(spa_adj, spa_adj @ Y, margin=rank_margin)
        # spa_inst = LocSpaAutoCorr(Y, spa_adj, method=autocorr_method)

    model = TransImp(
                dim_tgt_outputs=Y.shape[0],
                dim_ref_inputs=X.shape[0],
                spa_inst=spa_inst,
                mapping_mode=mapping_mode,
                dim_hid=mapping_lowdim,
                clip_max=clip_max,
                device=device,
                seed=seed).to(device)
    
    # if not spa_inst is None:
    #     with torch.no_grad():
    #         # tru_spa_stats = model.spa_inst.cal_spa_stats(Y)
    #         model.spa_inst.set_truth_stats(model.spa_inst.cal_spa_stats(Y))


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)        
    pbar = tqdm(range(n_epochs))

    for ith_epoch in pbar:
        # info  = train_imp_step(optimizer, model, X, Y, wt_spa, wt_l1norm, wt_l2norm, None if spa_inst is None else tru_spa_stats)
        info  = train_imp_step(optimizer, model, X, Y, wt_spa, wt_l1norm, wt_l2norm)
        pbar.set_description(f"[TransImp] Epoch: {ith_epoch+1}/{n_epochs}, {info}") 

    if signature_mode == 'cluster':
        test_X, _ = signature(classes, ct_list, df_ref[test_gene].values)
        # test_X = df_ref[test_gene].values
        test_X = tensify(test_X, device)
    else:
        test_X = tensify(df_ref[test_gene].values, device)

    return model, test_X

def expTransImp(df_ref, df_tgt, train_gene, test_gene, classes=None, ct_list=None,
             autocorr_method='moranI', signature_mode='cluster',
             mapping_mode='full',
             mapping_lowdim=256,
             spa_adj=None,
             lr=1e-2, weight_decay=1e-2, n_epochs=1000,
             clip_max=10,
             wt_spa=1.0,
             wt_l1norm=None,
             wt_l2norm=None,
             locations=None,
             n_simulation=None,
             device=None,
             seed=None):
    model, test_X = fit_transImp(
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
            sim_res = estimate_uncertainty(model, test_X, classes, n_simulations=n_simulation)
            return preds, sim_res
    return preds

def estimate_uncertainty(model, test_X, classes, n_simulations=100):
    sim_res = []
    classes = np.array(classes)
    for i in range(n_simulations):
        sim_test_X = torch.empty_like(test_X)
        for cls in np.unique(classes):
            cls_indices = np.argwhere(classes == cls).flatten()
            sim_indices = np.random.choice(cls_indices, cls_indices.shape[0], replace=True)
            sim_test_X[cls_indices] = test_X[sim_indices]
        preds = model.predict(sim_test_X)
        sim_res.append(preds)
    return sim_res

def estimate_uncertainty_local(model, test_X, classes, n_simulations=100):
    sim_res = {}
    classes = np.array(classes)
                  
    for cls in np.unique(classes):
        res = []
        for i in range(n_simulations):
            sim_test_X = test_X.clone().detach()
            cls_indices = np.argwhere(classes == cls).flatten()
            sim_indices = np.random.choice(cls_indices, cls_indices.shape[0], replace=True)
            sim_test_X[cls_indices] = test_X[sim_indices]
            preds = model.predict(sim_test_X)
            res.append(preds)
        sim_res[cls] = res
    return sim_res    



def expVeloImp(df_ref, df_tgt, S, U, V, train_gene, test_gene, classes=None, ct_list=None,
             autocorr_method='moranI', signature_mode='full',
             mapping_mode='nonelinear',
             mapping_lowdim=256,
             spa_adj=None,
             lr=1e-2, weight_decay=1e-2, n_epochs=1000,
             clip_max=10,
             wt_spa=1.0,
             wt_l1norm=None,
             wt_l2norm=None,
             locations=None,
             device=None,
             seed=None):
    model, test_X = fit_transImp(
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
        X  = model.predict(test_X)
    return _S, _U, _V, X