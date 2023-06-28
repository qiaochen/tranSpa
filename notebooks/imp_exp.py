import anndata
import torch
import stPlus
import os
import random
import warnings

import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.model_selection import KFold
from transpa.eval_util import calc_corr
from transpa.util import expTransImp, leiden_cluster, compute_autocorr
from benchmark import SpaGE_impute, Tangram_impute


warnings.filterwarnings('ignore')


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def run_stPlus(spatial_df, 
               scrna_df, 
               test_gene,
               save_name='tmp', 
               top_k=2000, 
               seed=None, 
               device=None):
    return stPlus.stPlus(spatial_df, 
                         scrna_df, 
                         test_gene, 
                         save_name, 
                         top_k=top_k, 
                         verbose=False, 
                         random_seed=seed, 
                         device=device)

def run_SpaGE(scrna_df, spatial_df, train_gene, test_gene):
    return SpaGE_impute(scrna_df, spatial_df, train_gene, test_gene)

def run_tangram(rna_adata, spa_adata, train_gene, test_gene, cls_key, device=None):
    return Tangram_impute(rna_adata, spa_adata, train_gene, test_gene, device, cls_key)

def run_transImpClsSpa(raw_scrna_df, raw_spatial_df, train_gene, test_gene, ct_list, classes, spa_adj, n_epochs=2000, seed=None, device=None):
    return expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                ct_list=ct_list,
                classes=classes,
                spa_adj=spa_adj,
                signature_mode='cluster',
                mapping_mode='full',
                n_epochs=n_epochs,
                seed=seed,
                device=device)

def run_transImpCls(raw_scrna_df, raw_spatial_df, train_gene, test_gene, ct_list, classes, spa_adj, n_epochs=2000, seed=None, device=None):
    return expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                ct_list=ct_list,
                classes=classes,
                signature_mode='cluster',
                mapping_mode='full',
                n_epochs=n_epochs,
                seed=seed,
                device=device)

def run_transImp(raw_scrna_df, raw_spatial_df, train_gene, test_gene, ct_list, classes,spa_adj,  n_epochs=2000, seed=None, device=None):
    return expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                signature_mode='cell',
                mapping_mode='lowrank',
                n_epochs=n_epochs,
                seed=seed,
                device=device)

def run_transImp_var(raw_scrna_df, raw_spatial_df, train_gene, test_gene, ct_list, classes,spa_adj,  n_epochs=2000, seed=None, device=None):
    res = expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                signature_mode='cell',
                mapping_mode='lowrank',
                n_epochs=n_epochs,
                seed=seed,
                device=device)
    return

def run_transImpSpa(raw_scrna_df, raw_spatial_df, train_gene, test_gene, ct_list, classes, spa_adj, rna_nb_idx, n_epochs=2000, seed=None, device=None):
    return expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                signature_mode='cell',
                mapping_mode='lowrank',
                n_simulation=100,
                classes=classes,
                clip_max=1,
                n_epochs=n_epochs,
                ref_nb_indices=rna_nb_idx,
                spa_adj=spa_adj,
                seed=seed,
                device=device)                

def cross_validation(
                     method_fn,
                     spa_adata,
                     rna_adata,
                     raw_spatial_df,
                     raw_scrna_df,
                     raw_shared_gene, 
                     K=5, 
                     seed=0, 
                     device=None):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    df_preds = pd.DataFrame(np.zeros((spa_adata.n_obs, len(raw_shared_gene))), columns=raw_shared_gene)

    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        print(f"\n===== Fold {idx+1} =====\nNumber of train genes: {len(train_ind)}, Number of test genes: {len(test_ind)}")
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        
        test_spatial_df = raw_spatial_df[test_gene]
        spatial_df = raw_spatial_df[train_gene]
        scrna_df   = raw_scrna_df
        df_preds[test_gene] = method_fn()
    return df_preds

    


if __name__ == "__main__":
    cross_validation()