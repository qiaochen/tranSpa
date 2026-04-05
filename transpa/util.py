import torch
import torch.nn.functional as F
import sys
import os

import numpy as np
import scanpy as sc
import pandas as pd

from scipy import sparse
from scipy.special import expit
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .model import TransDeconv, TransImp, SpaAutoCorr, SparkX


def is_count_data(X, n_check=1000):
    """Heuristic to detect whether a matrix contains raw count data.

    Checks a subsample for integer values and a sufficiently wide dynamic
    range.  Returns ``True`` when the data looks like raw UMI counts and
    ``False`` when it appears to be normalised / log-transformed.
    """
    sample = X[:n_check] if len(X) > n_check else X
    if sparse.issparse(sample):
        sample = sample.toarray()
    sample = np.asarray(sample)
    vals = sample[sample > 0]
    if len(vals) == 0:
        return True
    is_integer = np.allclose(vals, np.round(vals), atol=1e-3)
    has_large_range = float(vals.max()) > 50
    return is_integer and has_large_range

def plot_genes(genes, spa_adata, df_corr=None, is_I=False, n_cols=5, dpi=380, figsize=(20, 20)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams.update({"font.size":30, 'axes.titlesize':30})
    exprs = [spa_adata[:, gene].X.toarray().flatten() for gene in genes]
    if is_I:
        titles = [f'{gene}\n M.I. {spa_adata.uns["moranI"].loc[gene].I:.3f}' for gene in genes]
    else:
        titles = [f'{gene}\n C.S. {df_corr.loc[gene, "cos_sim"]:.3f}' for gene in genes]
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
    import squidpy as sq
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
    sc.tl.leiden(adata_cp)
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
    if top_k is None or top_k <= 0 or top_k >= data_mtx.shape[1]:
        return np.arange(data_mtx.shape[1])
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
              expr_mtx: np.array,
              raw_counts: bool=False):
    """Generate gene signatures by aggregation expression matrix
       based on cell types.

    Args:
        classes (np.array): cell type annotation
        ct_list (np.array): available cell type labels
        expr_mtx (np.array): expression matrix
        raw_counts (bool): If True, normalize each cell by total UMI
            before averaging (rate-based, RCTD-style).

    Returns:
        tensor, tensor: gene signature, class expression abundence
    """
    classes = np.asarray(classes)
    ct_list = np.asarray(ct_list)
    sigs = []
    for cls in ct_list:
        cells = expr_mtx[classes == cls]
        if raw_counts:
            numi = cells.sum(axis=1, keepdims=True) + 1e-10
            sigs.append((cells / numi).mean(axis=0, keepdims=True))
        else:
            sigs.append(np.sum(cells, axis=0, keepdims=True))
    g_cls_sig = np.vstack(sigs)
    cls_abd_sig = np.array([(classes == cls).sum() for cls in ct_list]).reshape(-1, 1)
    return g_cls_sig, cls_abd_sig


def platform_effect_normalize(g_cls_sig, Y, cls_abd_sig):
    """RCTD-style per-gene platform effect correction.

    Computes a pseudo-bulk prediction from the reference signatures weighted by
    cell-type frequencies, then returns the per-gene log-ratio that corrects
    for platform-specific gene detection differences.

    Both reference (predicted) and spatial (observed) sides are compared on the
    *rate* scale (per-UMI probabilities) so that the library-size component is
    factored out and handled separately by ``scaler_s``.

    Returns the *unchanged* signatures and the log-ratio vector.
    """
    freq = cls_abd_sig.flatten().astype(np.float64)
    freq = freq / (freq.sum() + 1e-10)
    predicted_rate = freq @ g_cls_sig

    observed_counts = np.asarray(Y).mean(axis=0)
    mean_lib_size = np.asarray(Y).sum(axis=1).mean()
    observed_rate = observed_counts / (mean_lib_size + 1e-10)

    ratio = observed_rate / (predicted_rate + 1e-10)
    ratio = np.clip(ratio, 0.01, 100.0)
    return g_cls_sig, np.log(ratio + 1e-10)


def _fast_rank_genes(X, labels, group_names, topk=30):
    """Vectorized one-vs-rest t-test for all groups at once.

    Uses sparse indicator matrix multiplication for O(1)-loop group statistics.
    Returns dict mapping group_name -> list of top-k gene column indices.
    """
    from scipy import sparse as _sp
    if _sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_cells, n_genes = X.shape

    unique_labels = np.unique(labels)
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[l] for l in labels], dtype=np.int32)
    n_groups = len(unique_labels)

    indicator = _sp.csc_matrix(
        (np.ones(n_cells, dtype=np.float32), (int_labels, np.arange(n_cells))),
        shape=(n_groups, n_cells))

    group_n = np.asarray(indicator.sum(axis=1)).ravel()
    group_sum = np.asarray(indicator.dot(X))

    X_sq = X ** 2
    group_sq_sum = np.asarray(indicator.dot(X_sq))
    total_sum = group_sum.sum(axis=0)
    total_sq_sum = group_sq_sum.sum(axis=0)
    del X_sq

    group_mean = group_sum / np.maximum(group_n[:, None], 1)

    result = {}
    for gname in group_names:
        if gname not in label_to_int:
            result[gname] = []
            continue
        gi = label_to_int[gname]
        n1 = group_n[gi]
        n2 = n_cells - n1
        if n1 < 2 or n2 < 1:
            result[gname] = []
            continue

        mean1 = group_mean[gi]
        var1 = group_sq_sum[gi] / n1 - mean1 ** 2
        var1 = np.maximum(var1, 0)

        rest_sum = total_sum - group_sum[gi]
        rest_sq = total_sq_sum - group_sq_sum[gi]
        mean2 = rest_sum / n2
        var2 = rest_sq / n2 - mean2 ** 2
        var2 = np.maximum(var2, 0)

        se = np.sqrt(var1 / n1 + var2 / n2 + 1e-30)
        t_stat = (mean1 - mean2) / se

        top_idx = np.argsort(-t_stat)[:topk]
        result[gname] = top_idx.tolist()

    return result


def select_deconv_genes(df_ref: pd.DataFrame, 
                        df_tgt: pd.DataFrame,
                        classes: np.array,
                        ct_list: np.array,
                        topk: int=30):
    """Select cell-type discriminative genes and build a per-type marker mask.

    Runs a Wilcoxon rank-sum test on the reference data to identify the top-k
    marker genes per cell type, intersects with genes present in the spatial
    target data, and returns both the filtered DataFrames and a boolean mask
    indicating which genes are markers for which cell type.

    Args:
        df_ref (pd.DataFrame): Reference expression [cells x genes], raw counts.
        df_tgt (pd.DataFrame): Spatial expression [spots x genes], raw counts.
        classes (np.array): Cell-type label for each reference cell.
        ct_list (np.array): Unique cell-type names.
        topk (int): Number of top marker genes to select per cell type.

    Returns:
        df_ref_sel (pd.DataFrame): Filtered reference expression.
        df_tgt_sel (pd.DataFrame): Filtered spatial expression.
        gene_mask (pd.DataFrame): Boolean mask [n_types x n_selected_genes].
    """
    import anndata

    shared_genes = np.intersect1d(df_ref.columns, df_tgt.columns)
    adata = anndata.AnnData(
        X=df_ref[shared_genes].values.copy(),
        obs=pd.DataFrame({'celltype': classes}),
        var=pd.DataFrame(index=shared_genes),
    )

    adata.layers['raw'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby='celltype', use_raw=False, method='wilcoxon')

    markers_per_ct = {}
    for ct in ct_list:
        markers_per_ct[ct] = list(adata.uns['rank_genes_groups']['names'][ct][:topk])

    all_markers = np.unique(np.concatenate(list(markers_per_ct.values())))
    selected_genes = np.intersect1d(all_markers, shared_genes)

    gene_mask = pd.DataFrame(
        np.zeros((len(ct_list), len(selected_genes)), dtype=bool),
        columns=selected_genes,
        index=ct_list,
    )
    for ct in ct_list:
        gene_mask.loc[ct, gene_mask.columns.isin(markers_per_ct[ct])] = True

    df_ref_sel = df_ref[selected_genes]
    df_tgt_sel = df_tgt[selected_genes]
    return df_ref_sel, df_tgt_sel, gene_mask


def _gini_coefficient(x):
    """Compute the Gini coefficient for a 1-D array (gene expression across cells)."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10)


def _gini_per_gene(expr_mtx):
    """Vectorised Gini coefficient for every column (gene) of a matrix."""
    X = np.asarray(expr_mtx, dtype=np.float64)
    n = X.shape[0]
    X_sorted = np.sort(X, axis=0)
    col_sums = X_sorted.sum(axis=0)
    idx = np.arange(1, n + 1).reshape(-1, 1)
    numerator = 2 * (idx * X_sorted).sum(axis=0) - (n + 1) * col_sums
    gini = numerator / (n * col_sums + 1e-10)
    gini[col_sums == 0] = 0.0
    return gini


def _rare_type_markers_gini(expr_mtx, gene_names, rare_mask,
                            topk=30, gini_quantile=0.9,
                            _precomputed_gini=None):
    """Select markers for rare cell types using a Gini-index heuristic.

    High-Gini genes are those expressed in only a few cells -- exactly the
    pattern that discriminates rare populations.  We intersect the top-Gini
    genes with genes that are upregulated in the rare cells (fold-change
    filter) to produce a focused marker set.

    Args:
        expr_mtx: Normalised expression matrix [cells x genes].
        gene_names: Gene name array aligned with columns of *expr_mtx*.
        rare_mask: Boolean mask (True for cells of the rare type).
        topk: Max markers to return.
        gini_quantile: Quantile threshold on the Gini distribution.
        _precomputed_gini: Pre-computed Gini scores to avoid recomputation.

    Returns:
        list[str]: Selected marker gene names.
    """
    gini_scores = _precomputed_gini if _precomputed_gini is not None else _gini_per_gene(expr_mtx)
    gini_thresh = np.quantile(gini_scores, gini_quantile)
    high_gini = gini_scores >= gini_thresh

    rare_mean = expr_mtx[rare_mask].mean(axis=0) + 1e-10
    other_mean = expr_mtx[~rare_mask].mean(axis=0) + 1e-10
    log2fc = np.log2(rare_mean / other_mean)

    upregulated = log2fc > 0.5
    detected_in_rare = (expr_mtx[rare_mask] > 0).mean(axis=0) > 0.3

    candidate = high_gini & upregulated & detected_in_rare
    if candidate.sum() == 0:
        candidate = upregulated & detected_in_rare

    order = np.argsort(-log2fc)
    selected = [gene_names[i] for i in order if candidate[i]][:topk]
    if len(selected) < 5:
        order_fc = np.argsort(-log2fc)
        for i in order_fc:
            if gene_names[i] not in selected and upregulated[i]:
                selected.append(gene_names[i])
            if len(selected) >= topk:
                break
    return selected


def select_deconv_genes_v2(df_ref, df_tgt, classes, ct_list, topk=30,
                           rare_threshold=20):
    """Abundance-aware marker gene selection (Phase 1).

    For cell types with >= *rare_threshold* cells, uses the standard
    Wilcoxon rank-sum test.  For rare types (fewer cells), uses a
    Gini-index + fold-change heuristic that is more robust at small n.

    Args:
        df_ref, df_tgt, classes, ct_list, topk: Same as select_deconv_genes.
        rare_threshold: Cell count below which a type is considered rare.

    Returns:
        Same as select_deconv_genes: (df_ref_sel, df_tgt_sel, gene_mask).
    """
    import anndata

    classes = np.asarray(classes)
    ct_list = np.asarray(ct_list)
    shared_genes = np.intersect1d(df_ref.columns, df_tgt.columns)

    adata = anndata.AnnData(
        X=df_ref[shared_genes].values.copy(),
        obs=pd.DataFrame({'celltype': classes}),
        var=pd.DataFrame(index=shared_genes),
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    ct_counts = pd.Series(classes).value_counts()
    abundant_types = [ct for ct in ct_list if ct_counts.get(ct, 0) >= rare_threshold]
    rare_types = [ct for ct in ct_list if ct_counts.get(ct, 0) < rare_threshold]

    markers_per_ct = {}

    gene_rank = _fast_rank_genes(adata.X, classes, list(ct_list), topk=topk)
    for ct in ct_list:
        idxs = gene_rank.get(ct, [])
        markers_per_ct[ct] = [shared_genes[i] for i in idxs]

    all_markers = np.unique(np.concatenate(list(markers_per_ct.values())))
    selected_genes = np.intersect1d(all_markers, shared_genes)

    gene_mask = pd.DataFrame(
        np.zeros((len(ct_list), len(selected_genes)), dtype=bool),
        columns=selected_genes, index=ct_list,
    )
    for ct in ct_list:
        gene_mask.loc[ct, gene_mask.columns.isin(markers_per_ct[ct])] = True

    return df_ref[selected_genes], df_tgt[selected_genes], gene_mask


def select_spatial_markers(spa_adata, shared_genes, topk=30,
                           _cached_labels=None):
    """Extract marker genes from spatial data via Leiden clustering (Phase 2).

    Uses a fast t-test instead of Wilcoxon for speed on spatial data
    (many clusters x many genes).

    Args:
        spa_adata: Spatial AnnData.
        shared_genes: Gene names present in both ref and target.
        topk: Number of markers per cluster.
        _cached_labels: Pre-computed Leiden labels to avoid redundant clustering.

    Returns:
        dict: mapping cluster_id -> list of marker gene names.
        pd.Series: cluster labels per spot.
    """
    if _cached_labels is not None:
        cluster_labels = _cached_labels
        cluster_ids = np.unique(cluster_labels.values)
    else:
        cluster_labels, cluster_ids = leiden_cluster(spa_adata, normalize=True)

    genes_to_use = np.intersect1d(spa_adata.var_names, shared_genes)
    spa_cp = spa_adata[:, genes_to_use].copy()
    sc.pp.normalize_total(spa_cp)
    sc.pp.log1p(spa_cp)
    spa_cp.obs['leiden'] = cluster_labels.values
    cids_str = [str(c) for c in cluster_ids]
    gene_rank = _fast_rank_genes(spa_cp.X, spa_cp.obs['leiden'].values, cids_str, topk=topk)

    spa_genes = np.asarray(spa_cp.var_names)
    spa_markers = {}
    for cid in cluster_ids:
        idxs = gene_rank.get(str(cid), [])
        spa_markers[cid] = [spa_genes[i] for i in idxs]
    return spa_markers, cluster_labels


def compute_gene_scores(spa_adata, markers_per_ct, ct_list):
    """Compute per-spot cell-type enrichment scores using sc.tl.score_genes.

    Uses scanpy's score_genes which subtracts background control gene scores,
    providing properly normalized enrichment (as in Sun et al. 2026).

    Returns:
        np.ndarray: Score matrix [n_spots x n_types].
    """
    spa_cp = spa_adata.copy()
    sc.pp.normalize_total(spa_cp)
    sc.pp.log1p(spa_cp)

    var_names = set(spa_cp.var_names)
    scores = np.zeros((spa_cp.n_obs, len(ct_list)))
    for j, ct in enumerate(ct_list):
        ct_genes = [g for g in markers_per_ct.get(ct, []) if g in var_names]
        if len(ct_genes) >= 2:
            score_key = f'_score_{j}'
            sc.tl.score_genes(spa_cp, ct_genes, score_name=score_key)
            scores[:, j] = spa_cp.obs[score_key].values
    return scores


def compute_signature_scores(spa_adata, df_ref, classes, ct_list, topk_sig=50):
    """Compute per-spot cell-type scores from the reference signature profile.

    For each cell type, selects the top *topk_sig* genes with the highest
    fold-change over the mean of other cell types in the reference signature,
    then uses sc.tl.score_genes to score spatial spots.

    Returns:
        np.ndarray: Score matrix [n_spots x n_types].
    """
    classes = np.asarray(classes)
    ct_list = np.asarray(ct_list)
    expr = df_ref.values
    gene_names = list(df_ref.columns)

    sig = np.vstack([expr[classes == ct].mean(axis=0) for ct in ct_list])
    overall_mean = sig.mean(axis=0, keepdims=True) + 1e-10

    spa_cp = spa_adata.copy()
    sc.pp.normalize_total(spa_cp)
    sc.pp.log1p(spa_cp)
    var_set = set(spa_cp.var_names)

    scores = np.zeros((spa_cp.n_obs, len(ct_list)))
    for j, ct in enumerate(ct_list):
        fold_change = sig[j] / overall_mean.ravel()
        ranked_idx = np.argsort(-fold_change)
        ct_genes = []
        for idx in ranked_idx:
            g = gene_names[idx]
            if g in var_set:
                ct_genes.append(g)
            if len(ct_genes) >= topk_sig:
                break
        if len(ct_genes) >= 2:
            score_key = f'_sigscore_{j}'
            sc.tl.score_genes(spa_cp, ct_genes, score_name=score_key)
            scores[:, j] = spa_cp.obs[score_key].values
    return scores


def map_clusters_to_celltypes(spa_markers, ref_markers_per_ct, ct_list):
    """Map spatial Leiden clusters to reference cell types via marker overlap (Phase 4).

    Returns:
        np.ndarray: Association matrix [n_clusters x n_types], Jaccard scores.
    """
    cluster_ids = sorted(spa_markers.keys())
    assoc = np.zeros((len(cluster_ids), len(ct_list)))
    for i, cid in enumerate(cluster_ids):
        spa_set = set(spa_markers[cid])
        for j, ct in enumerate(ct_list):
            ref_set = set(ref_markers_per_ct.get(ct, []))
            inter = len(spa_set & ref_set)
            union = len(spa_set | ref_set)
            assoc[i, j] = inter / max(union, 1)
    row_sums = assoc.sum(axis=1, keepdims=True)
    assoc = assoc / np.maximum(row_sums, 1e-10)
    return assoc, cluster_ids


def train_deconv_step(optimizer, model, X, Y, cls_abd_sig, wt_spa=1.0,
                   truth_autocorr=None, method_autocorr='moranI',
                   epoch_frac=1.0,
                   wt_l1=5.0, wt_abd=2.0, wt_l2_G=2.0, wt_l2_S=2.0,
                   grad_clip=0.0,
                   wt_mse=0.0, huber=True, wt_js=0.0):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(X, Y, cls_abd_sig, 
                      truth_autocorr=truth_autocorr, 
                      wt_spa=wt_spa,
                      wt_l1=wt_l1, wt_abd=wt_abd,
                      wt_l2_G=wt_l2_G, wt_l2_S=wt_l2_S,
                      method_autocorr=method_autocorr,
                      epoch_frac=epoch_frac,
                      wt_mse=wt_mse, huber=huber,
                      wt_js=wt_js,
                      )
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            gene_mask: pd.DataFrame=None,
            normalize_sig: bool=False,
            raw_counts: bool=True,
            init_weights: np.ndarray=None,
            anchor_weight: float=0.0,
            wt_l1: float=5.0,
            wt_abd: float=2.0,
            wt_l2_G: float=2.0,
            wt_l2_S: float=2.0,
            sig_mask_weight: float=1.0,
            rate_sig: bool=False,
            platform_norm: bool=False,
            cosine_lr: bool=False,
            warmup_epochs: int=0,
            grad_clip: float=0.0,
            wt_mse: float=0.0,
            huber: bool=True,
            wt_js: float=0.0,
            device: torch.device=None,
            seed: int=None):
    indices = select_top_variable_genes(df_ref.values, n_top_genes)
    X = df_ref.values[:, indices]
    Y = df_tgt.values[:, indices]

    g_cls_sig, cls_abd_sig = signature(classes, ct_list, X, raw_counts=rate_sig)

    if gene_mask is not None and sig_mask_weight < 1.0:
        gene_cols = df_ref.columns[indices]
        shared = gene_mask.columns.intersection(gene_cols)
        if len(shared) > 0:
            gene_col_list = list(gene_cols)
            col_map = {g: gene_col_list.index(g) for g in shared}
            mask_sub = gene_mask.reindex(index=ct_list, columns=shared).values
            soft_mask = np.ones_like(g_cls_sig)
            for g, j in col_map.items():
                soft_mask[:, j] = np.where(mask_sub[:, list(shared).index(g)],
                                           1.0, sig_mask_weight)
            g_cls_sig = g_cls_sig * soft_mask

    if platform_norm:
        g_cls_sig, log_platform_factor = platform_effect_normalize(
            g_cls_sig, Y, cls_abd_sig)
        scaler_g_init = tensify(log_platform_factor.reshape(1, -1), device)
        lib_size = Y.sum(axis=1, keepdims=True).astype(np.float64)
        scaler_s_init = tensify(np.log(lib_size + 1e-10), device)
    else:
        scaler_g_init = None
        scaler_s_init = None

    X, Y = tensify(g_cls_sig, device), tensify(Y, device)
    cls_abd_sig = tensify(cls_abd_sig, device)
    
    if spa_adj is not None:
        spa_adj = torch.sparse_coo_tensor(indices=np.array([spa_adj.row, spa_adj.col]),
                                                values=spa_adj.data,
                                                size=spa_adj.shape).to(device).float()

    model = TransDeconv(
                 dim_tgt_outputs=Y.shape[0],
                 n_feats=len(indices),
                 dim_ref_inputs=X.shape[0],
                 tau=tau,
                 spa_autocorr=None if spa_adj is None else SpaAutoCorr(spa_adj),
                 anchor_weight=anchor_weight,
                 scaler_g_init=scaler_g_init,
                 scaler_s_init=scaler_s_init,
                 device=device,
                 seed=seed).to(device)

    if init_weights is not None:
        with torch.no_grad():
            W_init = torch.tensor(init_weights.T, dtype=torch.float32, device=device)
            model.trans.trans.copy_(torch.sqrt(W_init + 1e-10))
            model.register_buffer('W_anchor', W_init)

    if spa_adj is not None:
        with torch.no_grad():
            truth_autocorr = model.spa_autocorr.cal_spa_stats(Y, autocorr_method)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if cosine_lr:
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs)
    else:
        scheduler = None
    pbar = tqdm(range(n_epochs), disable=not sys.stderr.isatty())

    for ith_epoch in pbar:
        info  = train_deconv_step(optimizer, model, X, Y, cls_abd_sig, wt_spa,
                                  None if spa_adj is None else truth_autocorr,
                                  autocorr_method,
                                  epoch_frac=(ith_epoch + 1) / n_epochs,
                                  wt_l1=wt_l1, wt_abd=wt_abd,
                                  wt_l2_G=wt_l2_G, wt_l2_S=wt_l2_S,
                                  grad_clip=grad_clip,
                                  wt_mse=wt_mse, huber=huber,
                                  wt_js=wt_js,
                                  )
        if scheduler is not None:
            scheduler.step()
        pbar.set_description(f"[LinTrans] Epoch: {ith_epoch+1}/{n_epochs}, {info}")

    return model, X, Y

def _adata_to_df(adata):
    """Convert AnnData.X to a pd.DataFrame, handling sparse matrices."""
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

def expDeconv(adata_ref: sc.AnnData=None,
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
              autocorr_method: str='moranI',
              spa_adj:sparse.coo_array=None,
              spa_adata: sc.AnnData=None,
              calibrate: float=0.0,
              gene_mask: pd.DataFrame=None,
              normalize_sig: bool=True,
              raw_counts: bool=None,
              smart_markers: bool=False,
              spatial_markers: bool=False,
              score_init: bool=True,
              sig_score_init: bool=False,
              cluster_mapping: bool=True,
              anchor_weight: float=0.0,
              sig_mask_weight: float=1.0,
              rate_sig: bool=False,
              platform_norm: bool=False,
              cosine_lr: bool=True,
              warmup_epochs: int=0,
              grad_clip: float=0.0,
              wt_mse: float=0.0,
              huber: bool=True,
              wt_js: float=2.0,
              sharpen_temp: float=1.0,
              topk_output: int=0,
              device: torch.device=None,
              seed: int=None,
              _cache_path: str=None):
    """Cell type deconvolution.

    Fits a linear translation model from reference cell-type gene signatures to
    spatial gene profiles, and returns the predicted per-spot cell-type weights.

    Data can be provided in two ways (adata-based or DataFrame-based):
      - **AnnData mode**: pass ``adata_ref`` and ``adata_tgt``. The function
        extracts ``df_ref``, ``df_tgt``, ``classes``, and ``ct_list``
        automatically using ``label_key`` to look up cell-type annotations in
        ``adata_ref.obs``. Gene intersection is computed internally.
      - **DataFrame mode** (legacy): pass ``df_ref``, ``df_tgt``, ``classes``,
        and ``ct_list`` explicitly.

    When ``spa_adata`` is not given but ``adata_tgt`` is, the spatial AnnData
    ``adata_tgt`` is used for Leiden-based spatial clustering automatically.

    Args:
        adata_ref (sc.AnnData, optional): Reference scRNA-seq AnnData.
        adata_tgt (sc.AnnData, optional): Spatial transcriptomics AnnData.
        label_key (str): Column in ``adata_ref.obs`` for cell-type labels.
        df_ref, df_tgt, classes, ct_list: Legacy DataFrame-mode inputs.
        lr, weight_decay: Optimiser hyperparameters.
        tau (float, optional): Softmax temperature.
        n_epochs (int): Training epochs.
        n_top_genes (int): Variable genes (ignored when topk is set).
        topk (int): Marker genes per cell type via DE test.
        wt_spa (float): Spatial regularization weight.
        autocorr_method (str): 'moranI' or 'gearyC'.
        spa_adj: Spatial adjacency matrix.
        spa_adata: Spatial AnnData for Leiden clustering.
        calibrate (float): Post-hoc calibration strength.
        gene_mask: Boolean mask [n_types x n_genes].
        normalize_sig (bool): L2-normalize cell-type signatures.
        raw_counts (bool, optional): Auto-detected if None.
        smart_markers (bool): Phase 1 -- abundance-aware marker selection.
        spatial_markers (bool): Phase 2 -- augment with spatial-derived markers.
        score_init (bool): Phase 3 -- sc.tl.score_genes warm-start.
        cluster_mapping (bool): Phase 4 -- cluster-to-celltype mapping reg.
        anchor_weight (float): Phase 5 -- KL anchor penalty weight.
        device: Torch device.
        seed: Random seed.

    Returns:
        np.array, np.ndarray: predicted ST expression, weight matrix
    """
    if adata_ref is not None and adata_tgt is not None:
        shared_genes = np.intersect1d(adata_ref.var_names, adata_tgt.var_names)
        if df_ref is None:
            df_ref = _adata_to_df(adata_ref[:, shared_genes])
        if df_tgt is None:
            df_tgt = _adata_to_df(adata_tgt[:, shared_genes])
        if classes is None:
            classes = np.asarray(adata_ref.obs[label_key].values)
        if ct_list is None:
            ct_list = np.unique(classes)
        if spa_adata is None:
            spa_adata = adata_tgt

    if raw_counts is None:
        raw_counts = is_count_data(df_ref.values if df_ref is not None else df_tgt.values)

    markers_per_ct = {}

    if topk is not None and topk > 0:
        if smart_markers:
            df_ref, df_tgt, _topk_mask = select_deconv_genes_v2(
                df_ref, df_tgt, classes, ct_list, topk=topk)
        else:
            df_ref, df_tgt, _topk_mask = select_deconv_genes(
                df_ref, df_tgt, classes, ct_list, topk=topk)
        if gene_mask is None:
            gene_mask = _topk_mask
        elif gene_mask is False:
            gene_mask = None
        n_top_genes = None

        for ct in ct_list:
            markers_per_ct[ct] = list(
                _topk_mask.columns[_topk_mask.loc[ct].values])

    spa_cluster_labels = None
    spa_markers_dict = None
    if spa_adata is not None:
        cluster_labels, _ = leiden_cluster(spa_adata, normalize=True)
        spa_cluster_labels = cluster_labels

        if spatial_markers:
            spa_markers_dict, spa_cluster_labels = select_spatial_markers(
                spa_adata, df_tgt.columns.tolist(), topk=topk or 30,
                _cached_labels=cluster_labels)
            spa_marker_genes = np.unique(np.concatenate(
                list(spa_markers_dict.values())))
            if gene_mask is not None and len(spa_marker_genes) > 0:
                best_ct_for_cluster = {}
                if markers_per_ct:
                    for cid, cid_genes in spa_markers_dict.items():
                        cid_set = set(cid_genes)
                        best_ct, best_overlap = None, 0
                        for ct in ct_list:
                            overlap = len(cid_set & set(markers_per_ct.get(ct, [])))
                            if overlap > best_overlap:
                                best_ct, best_overlap = ct, overlap
                        if best_ct is not None:
                            best_ct_for_cluster[cid] = best_ct
                for cid, cid_genes in spa_markers_dict.items():
                    ct = best_ct_for_cluster.get(cid)
                    if ct is None:
                        continue
                    for g in cid_genes:
                        if g in gene_mask.columns:
                            gene_mask.loc[ct, g] = True

        spa_clusters = cluster_labels.values
        cluster_ids = np.unique(spa_clusters)
        spot_names = spa_adata.obs_names
        tgt_names = df_tgt.index if hasattr(df_tgt.index, 'tolist') else np.arange(len(df_tgt))
        cluster_map = pd.Series(spa_clusters, index=spot_names).reindex(tgt_names)
        cluster_means = []
        for cid in cluster_ids:
            mask = (cluster_map.values == cid)
            if mask.sum() > 0:
                cluster_means.append(df_tgt.values[mask].mean(axis=0))
        if cluster_means:
            cluster_means = np.vstack(cluster_means)
            df_tgt_aug = pd.DataFrame(
                np.vstack([df_tgt.values, cluster_means]),
                columns=df_tgt.columns,
            )
            df_tgt = df_tgt_aug
            n_original_spots = len(df_tgt) - len(cluster_means)
        else:
            n_original_spots = len(df_tgt)
    else:
        n_original_spots = None

    init_weights = None
    if score_init and markers_per_ct and spa_adata is not None:
        score_matrix = compute_gene_scores(spa_adata, markers_per_ct, ct_list)
        init_weights = np.clip(score_matrix, 0, None)
        row_sums = init_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        init_weights = init_weights / row_sums
        n_total_tgt = len(df_tgt)
        if init_weights.shape[0] < n_total_tgt:
            n_aug = n_total_tgt - init_weights.shape[0]
            aug_w = np.full((n_aug, len(ct_list)), 1.0 / len(ct_list))
            init_weights = np.vstack([init_weights, aug_w])

    if sig_score_init and spa_adata is not None:
        sig_scores = compute_signature_scores(
            spa_adata, df_ref, classes, ct_list, topk_sig=topk or 50)
        sig_weights = np.clip(sig_scores, 0, None)
        row_sums = sig_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sig_weights = sig_weights / row_sums
        if init_weights is not None:
            base = init_weights[:spa_adata.n_obs]
            init_weights[:spa_adata.n_obs] = 0.5 * base + 0.5 * sig_weights
            row_sums = init_weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            init_weights = init_weights / row_sums
        else:
            init_weights = sig_weights
            n_total_tgt = len(df_tgt)
            if init_weights.shape[0] < n_total_tgt:
                n_aug = n_total_tgt - init_weights.shape[0]
                aug_w = np.full((n_aug, len(ct_list)), 1.0 / len(ct_list))
                init_weights = np.vstack([init_weights, aug_w])

    if cluster_mapping and spa_markers_dict is not None and spa_cluster_labels is not None:
        assoc_matrix, cids = map_clusters_to_celltypes(
            spa_markers_dict, markers_per_ct, ct_list)
        cid_to_idx = {c: i for i, c in enumerate(cids)}
        indirect_props = np.zeros((spa_adata.n_obs, len(ct_list)))
        for s, label in enumerate(spa_cluster_labels.values):
            idx = cid_to_idx.get(label, None)
            if idx is not None:
                indirect_props[s] = assoc_matrix[idx]
            else:
                indirect_props[s] = 1.0 / len(ct_list)
        row_sums = indirect_props.sum(axis=1, keepdims=True)
        indirect_props = indirect_props / np.maximum(row_sums, 1e-10)

        if init_weights is not None:
            blend = 0.85
            base = init_weights[:spa_adata.n_obs]
            blended = blend * base + (1 - blend) * indirect_props
            blended = blended / blended.sum(axis=1, keepdims=True)
            init_weights[:spa_adata.n_obs] = blended
        else:
            init_weights = indirect_props
            if n_original_spots is not None:
                n_total_tgt = len(df_tgt)
                if init_weights.shape[0] < n_total_tgt:
                    aug_w = np.full(
                        (n_total_tgt - init_weights.shape[0], len(ct_list)),
                        1.0 / len(ct_list))
                    init_weights = np.vstack([init_weights, aug_w])

    if n_top_genes is not None and n_top_genes > 0:
        n_top_genes = min(n_top_genes, min(df_ref.shape[1], df_tgt.shape[1]))

    # --- Preprocess/train cache split ---
    if _cache_path is not None:
        import pickle as _pkl
        if not os.path.exists(_cache_path):
            _cache_data = dict(
                df_ref=df_ref, df_tgt=df_tgt, classes=classes,
                ct_list=ct_list, n_top_genes=n_top_genes,
                init_weights=init_weights,
                n_original_spots=n_original_spots,
            )
            with open(_cache_path, 'wb') as _cf:
                _pkl.dump(_cache_data, _cf, protocol=4)
            return None, None
        else:
            with open(_cache_path, 'rb') as _cf:
                _cache_data = _pkl.load(_cf)
            df_ref = _cache_data['df_ref']
            df_tgt = _cache_data['df_tgt']
            classes = _cache_data['classes']
            ct_list = _cache_data['ct_list']
            n_top_genes = _cache_data['n_top_genes']
            init_weights = _cache_data['init_weights']
            n_original_spots = _cache_data['n_original_spots']

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
                            gene_mask=gene_mask,
                            init_weights=init_weights,
                            anchor_weight=anchor_weight,
                            wt_l1=wt_l1, wt_abd=wt_abd,
                            wt_l2_G=wt_l2_G, wt_l2_S=wt_l2_S,
                            sig_mask_weight=sig_mask_weight,
                            rate_sig=rate_sig,
                            platform_norm=platform_norm,
                            cosine_lr=cosine_lr,
                            warmup_epochs=warmup_epochs,
                            grad_clip=grad_clip,
                            wt_mse=wt_mse,
                            huber=huber,
                            wt_js=wt_js,
                            device=device,
                            seed=seed) 
    with torch.no_grad():
        model.eval()
        preds, weights = model.predict(X, return_cluster=True)

    if sharpen_temp != 1.0 and sharpen_temp > 0:
        weights = np.power(weights + 1e-30, 1.0 / sharpen_temp)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-30)

    if topk_output > 0 and topk_output < weights.shape[1]:
        topk_idx = np.argsort(weights, axis=1)[:, :-topk_output]
        for i in range(weights.shape[0]):
            weights[i, topk_idx[i]] = 0.0
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-30)

    if calibrate:
        alpha = float(calibrate) if not isinstance(calibrate, bool) else (1.0 if calibrate else 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if alpha > 0:
            ref_freq = np.array([(np.asarray(classes) == ct).sum() for ct in ct_list],
                                dtype=np.float64)
            ref_freq /= ref_freq.sum()
            raw_prop = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
            mean_prop = raw_prop.mean(axis=0)
            scale = ref_freq / (mean_prop + 1e-10)
            dampened_scale = np.power(scale, alpha)
            weights = weights * dampened_scale[np.newaxis, :]

    if n_original_spots is not None:
        preds = preds[:n_original_spots]
        weights = weights[:n_original_spots]

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
    if model.spa_inst is not None:
        info += f', (SPA) {wt_spa} x {spa_reg:.6f}'
    return info


def train_imp_step_batch(optimizer, model, X, Y, spot_indices, 
                         wt_spa=0.1, wt_l1norm=1e-2, wt_l2norm=1e-2):
    """Single training step on a mini-batch of spots."""
    model.train()
    optimizer.zero_grad()
    Y_batch = Y[spot_indices]
    loss, imp_loss, spa_reg = model.loss_batch(
        X, Y_batch, spot_indices,
        wt_l2norm=wt_l2norm, wt_l1norm=wt_l1norm, wt_spa=wt_spa)
    loss.backward()
    optimizer.step()
    return loss.item(), imp_loss, spa_reg


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
    Y = tensify(Y, device)
    if signature_mode == 'cluster':
        g_cls_sig, _ = signature(classes, ct_list, X)
    
        X = tensify(g_cls_sig, device)
    else:
        X = tensify(X, device)
        
    spa_inst = None
    if locations is not None:
        locations = tensify(locations, device)
        spa_inst = SparkX(locations)

    if spa_adj is not None:
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
    pbar = tqdm(range(n_epochs), disable=not sys.stderr.isatty())

    for ith_epoch in pbar:
        info  = train_imp_step(optimizer, model, X, Y, wt_spa, wt_l1norm, wt_l2norm)
        pbar.set_description(f"[TransImp] Epoch: {ith_epoch+1}/{n_epochs}, {info}") 

    if signature_mode == 'cluster':
        test_X, _ = signature(classes, ct_list, df_ref[test_gene].values)
        test_X = tensify(test_X, device)
    else:
        test_X = tensify(df_ref[test_gene].values, device)

    return model, X, Y, test_X

def fit_transImp_batched(
            df_ref: pd.DataFrame, 
            df_tgt: pd.DataFrame, 
            train_gene: list, 
            test_gene: list,
            lr: float, 
            weight_decay: float, 
            n_epochs: int,
            batch_size: int,
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
    """Fit TransImp with mini-batch training over spots.

    The reference X stays fully in memory while the target Y dimension
    is batched via a DataLoader. This enables training on datasets with
    millions of spatial spots.
    """
    from torch.utils.data import DataLoader, TensorDataset

    X = df_ref[train_gene].values
    Y = df_tgt[train_gene].values
    Y = tensify(Y, device)
    if signature_mode == 'cluster':
        g_cls_sig, _ = signature(classes, ct_list, X)
        X = tensify(g_cls_sig, device)
    else:
        X = tensify(X, device)
        
    spa_inst = None
    if locations is not None:
        locations = tensify(locations, device)
        spa_inst = SparkX(locations)

    if spa_adj is not None:
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

    spot_indices = torch.arange(Y.shape[0], device=device)
    loader = DataLoader(TensorDataset(spot_indices),
                        batch_size=batch_size, shuffle=True)
    
    pbar = tqdm(range(n_epochs), disable=not sys.stderr.isatty())
    for ith_epoch in pbar:
        epoch_loss, epoch_imp, epoch_spa = 0.0, 0.0, 0.0
        n_batches = 0
        for (batch_idx,) in loader:
            loss_val, imp_val, spa_val = train_imp_step_batch(
                optimizer, model, X, Y, batch_idx,
                wt_spa, wt_l1norm, wt_l2norm)
            epoch_loss += loss_val
            epoch_imp += imp_val
            epoch_spa += spa_val
            n_batches += 1
        epoch_loss /= n_batches
        epoch_imp /= n_batches
        epoch_spa /= n_batches
        info = f'loss: {epoch_loss:.6f}, (IMP) {epoch_imp:.6f}'
        if model.spa_inst is not None:
            info += f', (SPA) {wt_spa} x {epoch_spa:.6f}'
        pbar.set_description(f"[TransImp-Batch] Epoch: {ith_epoch+1}/{n_epochs}, {info}") 

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
    from torchmetrics.functional.regression import cosine_similarity
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



def _sparsify_joint(imp_S, imp_U, ref_S, ref_U):
    """Independent sparsification of S and U using per-layer detection rates.

    Each layer is sparsified according to its own per-gene detection rate in the
    reference, so the imputed data reproduces the reference's distinct sparsity
    patterns for spliced vs unspliced.  This is critical because
    ``scv.pl.proportions`` (via ``initial_size``) effectively measures the number
    of non-zero genes per layer, not expression magnitude.

    Both arrays are modified **in-place**.
    """
    ref_S_arr = np.asarray(ref_S)
    ref_U_arr = np.asarray(ref_U)
    for g in range(imp_S.shape[1]):
        s_detect = np.mean(ref_S_arr[:, g] != 0)
        if s_detect < 1.0:
            vals = np.abs(imp_S[:, g])
            thresh = np.quantile(vals, 1.0 - s_detect)
            imp_S[:, g][vals <= thresh] = 0.0

        u_detect = np.mean(ref_U_arr[:, g] != 0)
        if u_detect < 1.0:
            vals = np.abs(imp_U[:, g])
            thresh = np.quantile(vals, 1.0 - u_detect)
            imp_U[:, g][vals <= thresh] = 0.0


def _sparsify_rescale_uniform(imputed, reference):
    """Sparsify + rescale when imputed and reference have different gene sets.

    Uses the **mean** detection rate across all reference genes as a uniform
    per-gene threshold for every column of ``imputed``, then rescales the
    surviving non-zero values so their global median matches the reference.

    ``imputed`` is modified **in-place**.

    Args:
        imputed:   ndarray [n_spots, n_imputed_genes]
        reference: ndarray [n_spots, n_ref_genes]  (read-only)
    """
    mean_detect = np.mean(reference != 0)
    for g in range(imputed.shape[1]):
        thresh = np.quantile(np.abs(imputed[:, g]), 1.0 - mean_detect)
        imputed[:, g][np.abs(imputed[:, g]) <= thresh] = 0.0

    nz_ref = reference[reference != 0]
    nz_imp = imputed[imputed != 0]
    if len(nz_imp) > 0 and len(nz_ref) > 0:
        imputed[imputed != 0] *= np.median(np.abs(nz_ref)) / np.median(np.abs(nz_imp))


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
             batch_size: int=None,
             n_simulation: int=None,
             convert_uncertainty_score: bool=True,
             device: torch.device=None,
             seed: int=None,
             sparsify: bool=False):
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
        batch_size (int, optional): Mini-batch size for spot-batched training. 
            When None (default), uses the original full-matrix training.
            Set to e.g. 8192 for large-scale Xenium datasets.
        n_simulation (int, optional): Indicater & the number of local bootstraps for performance uncertainty estimation. Defaults to None.
        convert_uncertainty_score (bool, optional): whether or not to convert uncertainty score to certainty score with $sigmoid(-pred.var.)$, 
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.
        sparsify (bool, optional): When True, post-process predictions with
            per-gene quantile thresholding and magnitude rescaling using
            observed ST training genes as reference. Defaults to False.

    Returns:
        list: results
    """
    fit_fn_kwargs = dict(
        df_ref=df_ref, df_tgt=df_tgt,
        train_gene=train_gene, test_gene=test_gene,
        lr=lr, weight_decay=weight_decay, n_epochs=n_epochs,
        classes=classes, ct_list=ct_list,
        autocorr_method=autocorr_method,
        mapping_mode=mapping_mode,
        mapping_lowdim=mapping_lowdim,
        spa_adj=spa_adj,
        clip_max=clip_max,
        signature_mode=signature_mode,
        wt_spa=wt_spa,
        wt_l1norm=wt_l1norm,
        wt_l2norm=wt_l2norm,
        locations=locations,
        device=device,
        seed=seed,
    )
    if batch_size is not None:
        model, train_X, train_y, test_X = fit_transImp_batched(
            batch_size=batch_size, **fit_fn_kwargs)
    else:
        model, train_X, train_y, test_X = fit_transImp(**fit_fn_kwargs)

    with torch.no_grad():
        model.eval()
        if batch_size is not None:
            preds = model.predict_batched_calibrated(test_X, train_X, train_y,
                                                     batch_size=batch_size)
        else:
            preds = model.predict_calibrated(test_X, train_X, train_y)

        if sparsify:
            train_Y = df_tgt[train_gene].values
            _sparsify_rescale_uniform(preds, train_Y)

        if n_simulation is not None and classes is not None:
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

def _translate_batched(model, input_t, n_spots, device, batch_size=8192):
    """Translate reference matrix through the model in spot-batches.

    Computes ``W_norm.T @ input`` where ``W_norm`` is the column-normalised
    weight matrix, without ever materialising the full ``[n_cells, n_spots]``
    matrix.  Instead it uses ``forward_batch`` to process chunks of spots.

    Args:
        model: Trained ``TransImp`` model.
        input_t: Tensor of shape ``[n_genes, n_cells]`` (reference data, transposed).
        n_spots: Number of spatial spots.
        device: Torch device.
        batch_size: Spots per chunk.

    Returns:
        np.ndarray of shape ``[n_spots, n_genes]``.
    """
    ones_t = torch.ones(1, input_t.shape[1], device=device)
    chunks = []
    for start in range(0, n_spots, batch_size):
        end = min(start + batch_size, n_spots)
        idx = torch.arange(start, end, device=device)
        raw_chunk = model.trans.forward_batch(input_t, idx)
        col_sums = model.trans.forward_batch(ones_t, idx)
        normed = raw_chunk / (col_sums + 1e-10)
        chunks.append(normed.t().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _spliced_frac_per_group(labels, label_list, src_S, src_U, ref_expr=None):
    """Compute per-group mean spliced fraction and (optionally) expression signature.

    Args:
        labels:     (n_cells,) group label per cell.
        label_list: sequence of unique group labels.
        src_S:      (n_cells, n_test_genes) raw spliced counts.
        src_U:      (n_cells, n_test_genes) raw unspliced counts.
        ref_expr:   (n_cells, n_shared_genes) reference expression on shared
                    genes.  When provided the per-group mean expression is
                    returned as *ct_sig*; otherwise *ct_sig* is ``None``.

    Returns:
        ct_frac_gene: (n_groups, n_test_genes) mean per-gene S/(S+U).
        ct_sig:       (n_groups, n_shared_genes) or None.
    """
    labels = np.asarray(labels)
    n_groups = len(label_list)
    n_genes = src_S.shape[1]
    ct_frac_gene = np.zeros((n_groups, n_genes))
    ct_sig = np.zeros((n_groups, ref_expr.shape[1])) if ref_expr is not None else None
    for k, ct in enumerate(label_list):
        mask = labels == ct
        if ct_sig is not None:
            ct_sig[k] = ref_expr[mask].mean(axis=0)
        s_mean = np.mean(src_S[mask], axis=0)
        u_mean = np.mean(src_U[mask], axis=0)
        ct_frac_gene[k] = s_mean / (s_mean + u_mean + 1e-10)
    return ct_frac_gene, ct_sig


def _nnls_decompose(ct_sig, tgt_expr):
    """NNLS decomposition of spatial spots into group proportions.

    Args:
        ct_sig:   (n_groups, n_shared_genes) group expression signatures.
        tgt_expr: (n_spots, n_shared_genes) spatial expression.

    Returns:
        W: (n_groups, n_spots) column-normalised weight matrix.
    """
    from scipy.optimize import nnls
    n_groups, n_spots = ct_sig.shape[0], tgt_expr.shape[0]
    W = np.zeros((n_groups, n_spots))
    for j in range(n_spots):
        w, _ = nnls(ct_sig.T, tgt_expr[j])
        total = w.sum()
        W[:, j] = w / total if total > 0 else 1.0 / n_groups
    return W


def expVeloImp(adata_ref: sc.AnnData=None,
               adata_tgt: sc.AnnData=None,
               label_key: str='Class',
               df_ref: pd.DataFrame=None,
               df_tgt: pd.DataFrame=None,
               train_gene: list=None, 
               test_gene: list=None, 
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
               batch_size: int=None,
               device: torch.device=None,
               seed: int=None,
               adata_raw=None,
               S: np.array=None, 
               U: np.array=None, 
               S_raw: np.array=None,
               U_raw: np.array=None):
    """ST Velocity estimation via TransImp.

    Data can be provided in two ways:
      - **AnnData mode**: pass ``adata_ref`` and ``adata_tgt``. The function
        extracts ``df_ref``, ``df_tgt``, and ``train_gene`` automatically.
        ``classes``/``ct_list`` are extracted via ``label_key`` from
        ``adata_ref.obs`` when ``signature_mode='cluster'`` and they are not
        provided explicitly. ``test_gene`` and ``adata_raw`` must still be
        provided separately (they depend on scvelo filtering).
      - **DataFrame mode** (legacy): pass ``df_ref``, ``df_tgt``,
        ``train_gene``, ``test_gene`` explicitly.

    Args:
        adata_ref (sc.AnnData, optional): Reference scRNA-seq AnnData.
        adata_tgt (sc.AnnData, optional): Spatial transcriptomics AnnData.
        label_key (str, optional): Column in ``adata_ref.obs`` containing
            cell-type labels. Defaults to 'Class'.
        df_ref (pd.DataFrame, optional): Dataframe of reference single cell.
        df_tgt (pd.DataFrame, optional): Dataframe of ST for training.
        train_gene (list, optional): Training genes (shared between ref and ST).
        test_gene (list): Genes for ST prediction; should be in df_ref.
        classes (list, optional): Single-cell type annotations.
        ct_list (list, optional): List of cell type labels.
        autocorr_method (str, optional): Defaults to 'moranI'.
        signature_mode (str, optional): 'cell' or 'cluster'. Defaults to 'cell'.
        mapping_mode (str, optional): 'lowrank' or 'full'. Defaults to 'lowrank'.
        mapping_lowdim (int, optional): Defaults to 256.
        spa_adj (sparse.coo_array, optional): Spatial adjacency matrix.
        lr (float, optional): Defaults to 1e-2.
        weight_decay (float, optional): Defaults to 1e-2.
        n_epochs (int, optional): Defaults to 1000.
        clip_max (int, optional): Defaults to 10.
        wt_spa (float, optional): Defaults to 1.0.
        wt_l1norm (float, optional): Defaults to None.
        wt_l2norm (float, optional): Defaults to None.
        locations (np.array, optional): Spatial coordinates.
        batch_size (int, optional): Mini-batch size for spot-batched training.
            When None (default), uses the original full-matrix training.
            Set to e.g. 8192 for large-scale spatial datasets.
        device (torch.device, optional): Defaults to None.
        seed (int, optional): Defaults to None.
        adata_raw: Pre-normalisation AnnData with 'spliced' and 'unspliced'
            layers.  When provided, S/U/S_raw/U_raw are extracted and
            gene-filtered automatically; the explicit array parameters are
            ignored.
        S (np.array, optional): Spliced expression matrix (legacy).
        U (np.array, optional): Unspliced expression matrix (legacy).
        S_raw (np.array, optional): Cell-level raw spliced matrix.
        U_raw (np.array, optional): Cell-level raw unspliced matrix.

    Returns:
        tuple(np.array): (_S, _U, _V, _X) imputed ST matrices.
    """
    import scipy.sparse as sp

    if adata_ref is not None and adata_tgt is not None:
        if df_ref is None:
            df_ref = _adata_to_df(adata_ref)
        if df_tgt is None:
            df_tgt = _adata_to_df(adata_tgt)
        if train_gene is None:
            train_gene = np.intersect1d(df_ref.columns, df_tgt.columns)
        if signature_mode == 'cluster' and classes is None and label_key in adata_ref.obs.columns:
            classes = np.asarray(adata_ref.obs[label_key].values)
        if classes is not None and ct_list is None:
            ct_list = np.unique(classes)

    if adata_raw is not None:
        S_full = adata_raw.layers['spliced']
        U_full = adata_raw.layers['unspliced']
        if sp.issparse(S_full):
            S_full = S_full.toarray()
        if sp.issparse(U_full):
            U_full = U_full.toarray()
        orig_var = adata_raw.var_names
        gene_idx = np.array([orig_var.get_loc(g) for g in test_gene])
        S = S_full[:, gene_idx]
        U = U_full[:, gene_idx]
        S_raw = S
        U_raw = U

    fit_fn_kwargs = dict(
        df_ref=df_ref, df_tgt=df_tgt,
        train_gene=train_gene, test_gene=test_gene,
        lr=lr, weight_decay=weight_decay, n_epochs=n_epochs,
        classes=classes, ct_list=ct_list,
        autocorr_method=autocorr_method,
        mapping_mode=mapping_mode,
        mapping_lowdim=mapping_lowdim,
        spa_adj=spa_adj,
        clip_max=clip_max,
        signature_mode=signature_mode,
        wt_spa=wt_spa,
        wt_l1norm=wt_l1norm,
        wt_l2norm=wt_l2norm,
        locations=locations,
        device=device,
        seed=seed,
    )
    if batch_size is not None:
        model, train_X, train_y, test_X = fit_transImp_batched(
            batch_size=batch_size, **fit_fn_kwargs)
    else:
        model, train_X, train_y, test_X = fit_transImp(**fit_fn_kwargs)

    with torch.no_grad():
        model.eval()
        if batch_size is not None:
            _X = model.predict_batched_calibrated(test_X, train_X, train_y,
                                                   batch_size=batch_size)
        else:
            _X = model.predict_calibrated(test_X, train_X, train_y)

        np.nan_to_num(_X, copy=False)
        gene_std = _X.std(axis=0)
        zero_var_mask = gene_std < 1e-12
        if zero_var_mask.any():
            _X[:, zero_var_mask] += np.random.default_rng(0).normal(
                scale=1e-6, size=(_X.shape[0], int(zero_var_mask.sum())))

        n_spots = _X.shape[0]

        src_S = S_raw if S_raw is not None else S
        src_U = U_raw if U_raw is not None else U
        src_S_np = src_S if isinstance(src_S, np.ndarray) else np.asarray(src_S)
        src_U_np = src_U if isinstance(src_U, np.ndarray) else np.asarray(src_U)

        if signature_mode == 'cluster':
            ct_frac_gene, _ = _spliced_frac_per_group(
                classes, ct_list, src_S_np, src_U_np)
            ct_frac_gene_t = torch.tensor(ct_frac_gene, dtype=torch.float32,
                                          device=device)
            W = model.trans._get_weight_mtx()
            W_norm = W / (W.sum(dim=0, keepdim=True) + 1e-10)
            frac_S = (W_norm.t() @ ct_frac_gene_t).cpu().numpy()
        else:
            is_lowrank = hasattr(model.trans, 'trans1')
            if is_lowrank:
                ref_expr = train_X.cpu().numpy() if torch.is_tensor(train_X) else train_X
                tgt_expr = train_y.cpu().numpy() if torch.is_tensor(train_y) else train_y

                if classes is not None and ct_list is not None:
                    ct_labels, ct_unique = np.asarray(classes), ct_list
                else:
                    from sklearn.cluster import MiniBatchKMeans
                    n_cells = ref_expr.shape[0]
                    n_pseudo = int(np.clip(np.sqrt(n_cells), 10, 50))
                    km = MiniBatchKMeans(
                        n_clusters=n_pseudo, random_state=0,
                        batch_size=min(n_cells, max(n_pseudo * 100, 1024)))
                    ct_labels = km.fit_predict(ref_expr).astype(str)
                    ct_unique = np.unique(ct_labels).tolist()

                ct_frac_gene, ct_sig = _spliced_frac_per_group(
                    ct_labels, ct_unique, src_S_np, src_U_np,
                    ref_expr=ref_expr)
                W_ct = _nnls_decompose(ct_sig, tgt_expr)
                frac_S = (W_ct.T @ ct_frac_gene)
            else:
                src_S_t = tensify(src_S, device)
                src_U_t = tensify(src_U, device)
                W = model.trans._get_weight_mtx()
                cell_frac = src_S_t / (src_S_t + src_U_t + 1e-10)
                W_norm = W / (W.sum(dim=0, keepdim=True) + 1e-10)
                frac_S = (W_norm.t() @ cell_frac).cpu().numpy()

        frac_S = np.clip(frac_S, 0.0, 1.0)
        _S = _X * frac_S
        _U = _X * (1 - frac_S)

        _V = np.zeros_like(_S)

    if S_raw is not None and U_raw is not None:
        _sparsify_joint(_S, _U, S_raw, U_raw)

    train_Y = df_tgt[train_gene].values
    _sparsify_rescale_uniform(_X, train_Y)

    return _S, _U, _V, _X


def build_adata(S, U, X, spa_adata, ref_adata, celltype_label='Class'):
    """Build an AnnData from ``expVeloImp`` output with proper S/U handling.

    Creates the AnnData, stores spliced/unspliced layers, records
    ``initial_size`` (used by ``scv.pl.proportions``), and normalises
    all layers by a shared per-cell size factor so that the S/U ratio
    is preserved in the normalised space.

    Spots where both spliced and unspliced counts sum to zero are dropped
    to avoid division-by-zero during normalisation.

    Parameters
    ----------
    S : np.ndarray
        Spliced expression matrix (n_spots x n_genes).
    U : np.ndarray
        Unspliced expression matrix (n_spots x n_genes).
    X : np.ndarray
        Total expression matrix (n_spots x n_genes).
    spa_adata : AnnData
        Spatial AnnData (HybISS / Visium etc.) — provides obs, obsm, uns.
    ref_adata : AnnData
        Reference scRNA-seq AnnData — provides var_names and uns colors.
    celltype_label (str, optional): Cell type label column in spa_adata.obs. Defaults to 'Class'.

    Returns
    -------
    AnnData
        Ready for ``run_velocity`` or downstream analysis.
    """
    import scvelo as scv

    adata = sc.AnnData(X)
    adata.obs = spa_adata.obs.copy()
    adata.obs_names = spa_adata.obs_names
    adata.var_names = ref_adata.var_names

    for key in spa_adata.obsm:
        adata.obsm[key] = spa_adata.obsm[key].copy()

    adata.uns = spa_adata.uns.copy()
    if f'{celltype_label}_colors' in ref_adata.uns:
        adata.uns[f'{celltype_label}_colors'] = ref_adata.uns[f'{celltype_label}_colors']

    adata.layers['spliced'] = S
    adata.layers['unspliced'] = U

    scv.core.set_initial_size(adata)
    combined_size = np.array(S.sum(axis=1) + U.sum(axis=1)).flatten()

    valid_mask = combined_size > 0
    if not valid_mask.all():
        adata = adata[valid_mask].copy()
        combined_size = combined_size[valid_mask]

    scv.pp.normalize_per_cell(adata, counts_per_cell=combined_size, enforce=True)
    sc.pp.scale(adata)

    return adata


def run_velocity(adata, vkey='stc_velocity', mode=None, n_pcs=30,
                 n_neighbors=30, n_jobs=10):
    """Run the full RNA-velocity pipeline on an AnnData.

    Performs PCA, neighbor graph, UMAP, Leiden clustering, moment
    estimation, velocity inference, and velocity confidence scoring.

    Parameters
    ----------
    adata : AnnData
        Output of :func:`build_adata` or a preprocessed scRNA-seq AnnData.
    vkey : str
        Key under which velocities are stored.
    mode : str or None
        Velocity mode passed to ``scv.tl.velocity`` (e.g. ``'stochastic'``).
        ``None`` uses scvelo's default.
    n_pcs : int
        Number of principal components.
    n_neighbors : int
        Number of neighbours for the kNN graph.
    n_jobs : int
        Parallelism for ``velocity_graph``.

    Returns
    -------
    AnnData
        The same object, modified in-place and returned for convenience.
    """
    import scvelo as scv

    sc.tl.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    velo_kwargs = dict(vkey=vkey)
    if mode is not None:
        velo_kwargs['mode'] = mode
    scv.tl.velocity(adata, **velo_kwargs)
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
    scv.tl.velocity_confidence(adata, vkey=vkey)

    return adata