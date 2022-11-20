import warnings

import scanpy as sc
import pandas as pd
import numpy as np
import tangram as tg
import stPlus

from SpaGE.main import SpaGE

def SpaGE_impute(RNA_data, Spatial_data, train_gene, predict_gene):
    RNA_data = RNA_data.loc[(RNA_data.sum(axis=1) != 0)]
    RNA_data = RNA_data.loc[(RNA_data.var(axis=1) != 0)]
    train = train_gene
    predict = predict_gene
    pv = len(train)/2 if len(train) > 50 else len(train)
    pv = min(pv, Spatial_data.shape[0])
    Spatial = Spatial_data[train]
    Img_Genes = SpaGE(Spatial, RNA_data, n_pv = int(pv), genes_to_predict = predict)
    result = Img_Genes[predict]
    return result 


def Tangram_impute(RNA_data_adata, 
                   Spatial_data_adata, 
                   train_gene, 
                   predict_gene, 
                   device, 
                   cls_key=None):

    test_list = predict_gene
    train_list = train_gene
    spatial_data_partial = Spatial_data_adata[:, train_list].copy()
    if cls_key is None:
        cls_key = 'leiden'
        RNA_data_adata_label = RNA_data_adata.copy()
        sc.pp.normalize_total(RNA_data_adata_label)
        sc.pp.log1p(RNA_data_adata_label)
        sc.pp.highly_variable_genes(RNA_data_adata_label)
        RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
        sc.pp.scale(RNA_data_adata_label, max_value=10)
        sc.tl.pca(RNA_data_adata_label)
        sc.pp.neighbors(RNA_data_adata_label)
        sc.tl.leiden(RNA_data_adata_label, resolution = 0.5)
        RNA_data_adata.obs[cls_key]  = RNA_data_adata_label.obs.leiden

    tg.pp_adatas(RNA_data_adata, spatial_data_partial, genes=None)
    ad_map = tg.map_cells_to_space(RNA_data_adata, spatial_data_partial, device = device, mode='clusters', cluster_label = cls_key, density_prior='rna_count_based')
    ad_ge = tg.project_genes(ad_map, RNA_data_adata, cluster_label=cls_key)
    test_list = list(map(lambda x:x.lower(), test_list))
    overlapped = np.intersect1d(ad_ge.var_names, test_list)
    no_pred_gene = np.setdiff1d(test_list, overlapped)
    pre_gene = pd.DataFrame(ad_ge[:,overlapped].X, index=ad_ge[:,overlapped].obs_names, columns=overlapped)
    if len(no_pred_gene) > 0:
        print(f'{len(no_pred_gene)} genes missing in tangram output.')
        for g in no_pred_gene:
            pre_gene[g] = 0
    return pre_gene[test_list].values

def stPlus_impute(RNA_data, Spatial_data, train_gene, predict_gene, device):
    stPlus_res = stPlus.stPlus(Spatial_data[train_gene], RNA_data.T, predict_gene, "tmp", verbose=False, device=device)
    return stPlus_res     