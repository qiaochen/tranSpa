import numpy as np
import pandas as pd
import torch
import scanpy as sc

from torch import nn
import squidpy as sq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
from chi2comb import chi2comb_cdf, ChiSquared

from tqdm import tqdm

from .model import TransDeconv, TransImpute, TransImp, SpaAutoCorr, SparkX, SpaReg
from .eval_util import spark_stat


def compute_autocorr(spa_adata, df):
    imputed_adata = spa_adata.copy()
    imputed_adata.X = df[imputed_adata.var_names].values
    sq.gr.spatial_autocorr(
        imputed_adata,
        n_jobs=10,
        mode='moran'
    )
    sq.gr.spatial_autocorr(
        imputed_adata,
        n_jobs=10,
        mode='geary',
    )    
    imputed_adata.var['sparkx'] = spark_stat(imputed_adata.obsm['spatial'], imputed_adata.X)
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

def decompose_raw_gene_mtx(data_mtx, low_dim):
    scrna_pca = TruncatedSVD(n_components=low_dim)
    scrna_g_rep = scrna_pca.fit_transform(data_mtx.T).T
    return scrna_g_rep, scrna_pca.explained_variance_ratio_


def prepare_ref_gene_rep(expr_mat, low_dim=2000, preserved_gene_ids=[], n_genes=None):
    ind = np.array(preserved_gene_ids) # range(expr_mat.shape[1])
    if not n_genes is None and n_genes > 0:
        ind = select_top_variable_genes(expr_mat, n_genes)
        ind = np.union1d(preserved_gene_ids, ind)
    if not low_dim is None:
        rep, weights = decompose_raw_gene_mtx(expr_mat[:, ind], low_dim)
    else:
        rep, weights = expr_mat[:, ind] if not sparse.issparse(expr_mat) else expr_mat.toarray()[:, ind], 0
    return rep, weights, ind


def auto_RG_dim(n_genes):
    if n_genes > 512:
        return int(np.sqrt(n_genes)) + 1
    elif n_genes > 128:
    # return max(int(np.sqrt(n_genes)) * 2, 16) 
        return int(np.power(n_genes, 0.6)) + 1
    elif n_genes > 32:
        return int(np.power(n_genes, 0.7)) + 1
    else:
        return int(np.power(n_genes, 0.8)) + 1

def prepare_data_lite(df_ref, df_tgt, train_genes, test_genes, pca_low_dim):
    full_ref_df = df_ref[np.concatenate([train_genes, test_genes])]
    expr_mat = full_ref_df.sparse.to_coo().tocsc() if pd.api.types.is_sparse(full_ref_df) else full_ref_df.values
    X, _ = decompose_raw_gene_mtx(expr_mat, pca_low_dim)
    tgt_Y = df_tgt[train_genes].values
    return X, tgt_Y

def prepare_data(df_ref, df_tgt, train_genes, test_genes, pca_low_dim, n_genes):
    shared_genes = np.intersect1d(df_ref.columns, df_tgt.columns)
    preserved_genes = np.union1d(shared_genes, test_genes)
    preserved_gene_ids = [df_ref.columns.get_loc(name) for name in preserved_genes]
    ref_X, _, sel_ids = prepare_ref_gene_rep(
                                   df_ref.sparse.to_coo().tocsc() if pd.api.types.is_sparse(df_ref) else df_ref.values, 
                                   low_dim=pca_low_dim,
                                   preserved_gene_ids=preserved_gene_ids, 
                                   n_genes=n_genes)

    bridge_gene_ids = [np.argwhere(sel_ids == df_ref.columns.get_loc(g)).flatten()[0]  for g in np.setdiff1d(shared_genes, test_genes)]
    test_gene_ids = [np.argwhere(sel_ids == df_ref.columns.get_loc(g)).flatten()[0]  for g in test_genes]
    tgt_Y = df_tgt[train_genes].values
    return ref_X, tgt_Y, bridge_gene_ids, test_gene_ids, sel_ids

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


def train_nntrans_step(optimizer, model, X, Y, reg_dims, 
                      ae_wt, kld_wt=1.0):
    model.train()
    optimizer.zero_grad()
    tt_loss, trans_loss, ae_loss, kld_loss = model.loss_vae_trans(X, Y, reg_dims, ae_wt, kld_wt)
    tt_loss.backward()
    optimizer.step()
    info = f'loss: {tt_loss.item():.6f}, (Trans): {trans_loss.item():.6f}, (AE) {ae_loss.item():.6f}, (KLD) {kld_loss.item():.6f}'
    return info


def fit_base(df_ref, df_tgt, train_genes, 
             test_genes,
             ae_wt, kld_wt,
             lr, weight_decay, n_epochs,
             dim_hid_AE, 
             dim_hid_RG,
             pca_low_dim, 
             n_top_genes,
             device=None,
             seed=None):
    X, Y, reg_dims, test_dims, all_dims = prepare_data(
                                            df_ref, df_tgt, train_genes, test_genes, 
                                            pca_low_dim=pca_low_dim,
                                            n_genes=n_top_genes
                                        )
    assert(X.shape[1] == len(all_dims))
    
    if dim_hid_RG is None:
        dim_hid_RG = auto_RG_dim(Y.shape[1])  

    if ae_wt is None:
        n_shared_genes = (len(reg_dims) + len(test_dims)) 
        n_spots = df_tgt.shape[0]
        ae_wt = n_shared_genes * n_spots / (len(all_dims) - n_shared_genes) / X.shape[0]                                              
    
    X, Y = tensify(X, device), tensify(Y, device)

    model = TransImpute(dim_tgt_outputs=Y.shape[0],
                        dim_ref_inputs=X.shape[0],
                        dim_hid_AE=dim_hid_AE,
                        dim_hid_RG=dim_hid_RG,
                        device=device,
                        seed=seed).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)        
    pbar = tqdm(range(n_epochs))

    for ith_epoch in pbar:
        info  = train_nntrans_step(optimizer, model, X, Y, reg_dims, 
                                    ae_wt=ae_wt, kld_wt=kld_wt)
        pbar.set_description(f"[TranSpa] Epoch: {ith_epoch+1}/{n_epochs}, {info}")
    return model, X[:, test_dims]


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
    return preds


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


def sparkx_stat(loc, expr, STS_inv, return_pval=True, mean_expr=None):
    # cell/spot by gene
    N = expr.shape[0]
    EHL = expr.t() @ loc 
    numerator = torch.sum((EHL @ STS_inv) * EHL, dim=1)
    denominator = torch.sum(torch.square(expr), dim=0)
    test_stats = numerator * N / denominator
    if not return_pval:
        return test_stats
    
    S_eigvals = torch.linalg.eigvals(STS_inv @ loc.t() @ loc).real
    
    # np_S_eigvals = np.linalg.eigvals(loc.t().cpu().numpy() @ loc.cpu().numpy() @ STS_inv.cpu().numpy())
    # print(S_eigvals, S_eigvals.real, (S_eigvals.imag != 0).sum(), S_eigvals.shape[0])
    # print(np_S_eigvals)
    expr_eigvals = 1 -  N * torch.square(mean_expr) / denominator
    coefs = S_eigvals.view(1, -1) * expr_eigvals.view(-1, 1)
    # coefs = (-np.sort(-coefs, axis=-1))[0]
    coefs = torch.sort(coefs, dim=-1, descending=True)[0].cpu().numpy()
    test_stats = test_stats.cpu().numpy()
    pvals = [1-chi2comb_cdf(_stat, [ChiSquared(_coef, 0, 1) for _coef in _coefs], 0, atol=1e-8)[0] for _stat, _coefs in zip(test_stats, coefs)]
    return test_stats, np.array(pvals)

def sparkx_test(expr, loc, device=None):
    expr = torch.FloatTensor(expr).to(device)
    loc  = torch.FloatTensor(loc).to(device)
    mean_expr = torch.mean(expr, dim=0, keepdim=True)
    centered_expr = expr - mean_expr
    centered_loc = loc - torch.mean(loc, dim=0, keepdim=True)
    loc_inv = torch.linalg.solve(centered_loc.t() @ centered_loc, torch.eye(centered_loc.shape[1]).to(device))
    stats, pvals = sparkx_stat(centered_loc, centered_expr, loc_inv,
                                return_pval=True, mean_expr=mean_expr)
    return stats, pvals



