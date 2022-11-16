from locale import normalize
import torch

import os
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from transpa.eval_util import calc_corr, CalculateMeteics, mse_moranI, mse_gearyC
from transpa.util import expTransImp, leiden_cluster, compute_autocorr
from benchmark import Tangram_impute, SpaGE_impute, stPlus_impute
from scipy import sparse

seed = 10
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
data_path = "/data/users/cqiao/data/spatialbm/DataUpload"

def load_data(root, normalize=True):
    RNA_file = os.path.join(root, 'scRNA_count.txt')
    Spatial_file = os.path.join(root, 'Insitu_count.txt')
    location_file = os.path.join(root, 'Locations.txt')

    RNA_data = pd.read_table(RNA_file, header=0, index_col = 0)
    Spatial_data = pd.read_table(Spatial_file, sep = '\t',header = 0)
    RNA_data_adata = sc.read(RNA_file, sep = '\t', first_column_names = True).T
    Spatial_data_adata = sc.read(Spatial_file, sep = '\t')
    locations = np.loadtxt(location_file, skiprows=1)
    Spatial_data_adata.obsm['spatial'] = locations
    if normalize:
        sc.pp.normalize_total(Spatial_data_adata)
        sc.pp.log1p(Spatial_data_adata)
        # sc.pp.scale(Spatial_data_adata)
    sc.pp.normalize_total(RNA_data_adata)
    sc.pp.log1p(RNA_data_adata)
    # sc.pp.scale(RNA_data_adata)
    sq.gr.spatial_neighbors(Spatial_data_adata)
    sq.gr.spatial_autocorr(
        Spatial_data_adata,
        n_jobs=10,
        mode='moran',
    )
    sq.gr.spatial_autocorr(
        Spatial_data_adata,
        n_jobs=10,
        mode='geary',
    )
    train_gene = np.load(os.path.join(root, 'train_list.npy'), allow_pickle = True).tolist()
    predict_gene = np.load(os.path.join(root, 'test_list.npy'), allow_pickle = True).tolist()
    print(RNA_data.shape, RNA_data_adata.shape)
    RNA_data.loc[RNA_data_adata.var_names, :] = RNA_data_adata.X.T if not sparse.issparse(RNA_data_adata.X) else RNA_data_adata.X.toarray().T
    Spatial_data.loc[:, Spatial_data_adata.var_names] = Spatial_data_adata.X if not sparse.issparse(Spatial_data_adata.X) else Spatial_data_adata.X.toarray()
    return RNA_data, Spatial_data, RNA_data_adata, Spatial_data_adata, train_gene, predict_gene


def evaluate_data(root):
    dataset = root.split("/")[-1]
    RNA_data, Spatial_data, RNA_data_adata, Spatial_data_adata, \
            train_gene_folds, test_gene_folds = load_data(root, normalize=True if not dataset in ['Dataset6', 'Dataset8'] else False)
    classes, ct_list = leiden_cluster(RNA_data_adata, False)
    cls_key = 'leiden'
    RNA_data_adata.obs[cls_key] = classes

    adata_spatial = Spatial_data_adata
    raw_spatial_df  = Spatial_data
    raw_scrna_df    = RNA_data.T
    

    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)
    
    df_transImp = pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=adata_spatial.var_names)
    df_transImpSpa = pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=adata_spatial.var_names)
    df_transImpCls = pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=raw_shared_gene)
    df_transImpClsSpa = pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=raw_shared_gene)
    df_tangram =  pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=adata_spatial.var_names)
    df_spaGE   =  pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=adata_spatial.var_names)
    df_stplus  = pd.DataFrame(np.zeros((adata_spatial.n_obs, adata_spatial.n_vars)), columns=adata_spatial.var_names)

    for idx, (train_gene, test_gene) in enumerate(zip(train_gene_folds, test_gene_folds)):    
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d"%(idx + 1, len(train_gene), len(test_gene)))
                
        df_transImpSpa[test_gene] = expTransImp(
            df_ref=raw_scrna_df,
            df_tgt=raw_spatial_df,
            train_gene=train_gene,
            test_gene=test_gene,
            signature_mode='cell',
            mapping_mode='lowrank',
            mapping_lowdim=128,
            clip_max=1.0,
            wt_spa=0.01,
            spa_adj=adata_spatial.obsp['spatial_connectivities'].tocoo(),
            seed=seed,
            device=device)

        corr_transImpSpa_res = calc_corr(raw_spatial_df, df_transImpSpa, test_gene)
        print(f'fold {idx}, median correlation: {np.median(corr_transImpSpa_res)} (TransImpSpa)')

        df_transImpCls[test_gene] = expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                ct_list=ct_list,
                classes=classes,
                signature_mode='cluster',
                mapping_mode='full',
                seed=seed,
                device=device)

        corr_transImpSpa_res = calc_corr(raw_spatial_df, df_transImpCls, test_gene)
        print(f'fold {idx}, median correlation: {np.median(corr_transImpSpa_res)} (TransImpCls)')

        df_transImp[test_gene] = expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                signature_mode='cell',
                mapping_mode='lowrank',
                clip_max=1.0,
                mapping_lowdim=128,
                seed=seed,
                device=device)

        corr_transImpSpa_res = calc_corr(raw_spatial_df, df_transImp, test_gene)
        print(f'fold {idx}, median correlation: {np.median(corr_transImpSpa_res)} (TransImp)')

        df_transImpClsSpa[test_gene] = expTransImp(
                df_ref=raw_scrna_df,
                df_tgt=raw_spatial_df,
                train_gene=train_gene,
                test_gene=test_gene,
                ct_list=ct_list,
                classes=classes,
                spa_adj=adata_spatial.obsp['spatial_connectivities'].tocoo(),
                signature_mode='cluster',
                mapping_mode='full',
                wt_spa=0.01,
                seed=seed,
                device=device)

        corr_transImpSpa_res = calc_corr(raw_spatial_df, df_transImpClsSpa, test_gene)
        print(f'fold {idx}, median correlation: {np.median(corr_transImpSpa_res)} (TransImpClsSpa)')

        try: 
            df_stplus[test_gene] = stPlus_impute(RNA_data, Spatial_data, train_gene, test_gene, device)
        except Exception as e:
            print(e)
            print("stplus failed, set predictions to be all -1")
            df_stplus[test_gene] = -1

        corr_res = calc_corr(raw_spatial_df, df_stplus, test_gene)
        print(f'{dataset}\n- Fold {idx}, median correlation: {np.median(corr_res)} (stPlus)')
        
        try:
            df_tangram[test_gene] = Tangram_impute(RNA_data_adata, Spatial_data_adata, train_gene, test_gene, device=device, cls_key=cls_key)
        except Exception as e:
            print(e)
            print("tangram failed, set predictions to be all -1")
            df_tangram[test_gene] = -1
        
        corr_res = calc_corr(raw_spatial_df, df_tangram, test_gene)
        print(f'{dataset}\n- Fold {idx}, median correlation: {np.median(corr_res)} (tangram)')

        try:
            recons_imput = SpaGE_impute(raw_scrna_df, raw_spatial_df.astype('float'), train_gene, test_gene)
            df_spaGE[test_gene] = recons_imput  
        except Exception as e:
            print(e)
            print("SpaGE failed, set predictions to be all -1")
            df_spaGE[test_gene] = -1
              
        corr_res = calc_corr(raw_spatial_df, df_spaGE, test_gene)
        print(f'{dataset}\n- Fold {idx}, median correlation: {np.median(corr_res)} (spaGE)')

    df_spaGE.to_csv(f"./output/imputation/spaGE_{dataset}_NN.csv")
    df_transImp.to_csv(f"./output/imputation/transImp_{dataset}_NN.csv")
    df_transImpSpa.to_csv(f"./output/imputation/transImpSpa_{dataset}_NN.csv")
    df_transImpCls.to_csv(f"./output/imputation/transImpCls_{dataset}_NN.csv")
    df_transImpClsSpa.to_csv(f"./output/imputation/transImpClsSpa_{dataset}_NN.csv")
    df_stplus.to_csv(f"./output/imputation/stPlus_{dataset}_NN.csv")
    df_tangram.to_csv(f"./output/imputation/tangram_{dataset}_NN.csv")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), 
                        f"./output/imputation/transImp_{dataset}_NN.csv", 
                        prefix=f'./output/imputation/metrics/transImp_{dataset}',
                        ).compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImp)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImp)))
    print("====transImp====\n")
    
    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), 
                                f"./output/imputation/transImpSpa_{dataset}_NN.csv", 
                                prefix=f'./output/imputation/metrics/transImpSpa_{dataset}'
                                ).compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpSpa)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpSpa)))
    print("====transImpSpa====\n")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), 
                                f"./output/imputation/transImpCls_{dataset}_NN.csv", 
                                prefix=f'./output/imputation/metrics/transImpCls_{dataset}', 
                                ).compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpCls)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpCls)))
    print("====transImpCls====\n")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), 
                                f"./output/imputation/transImpClsSpa_{dataset}_NN.csv", 
                                prefix=f'./output/imputation/metrics/transImpClsSpa_{dataset}', 
                                ).compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpClsSpa)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_transImpClsSpa)))
    print("====transImpClsSpa====\n")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), 
                                f"./output/imputation/tangram_{dataset}_NN.csv", 
                                prefix=f'./output/imputation/metrics/tangram_{dataset}', 
                                ).compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_tangram)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_tangram)))
    print("====tangram====\n")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), f"./output/imputation/spaGE_{dataset}_NN.csv", prefix=f'./output/imputation/metrics/spaGE_{dataset}').compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_spaGE)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_spaGE)))
    print("====spaGE====\n")

    results = CalculateMeteics(os.path.join(root, 'Insitu_count.txt'), f"./output/imputation/stPlus_{dataset}_NN.csv", prefix=f'./output/imputation/metrics/stPlus_{dataset}').compute_all()
    print(results.median(axis=1))
    print(results.mean(axis=1))
    print(results.std(axis=1))
    # print("MSE moranI", mse_moranI(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_stplus)))
    # print("MSE gearyC", mse_gearyC(Spatial_data_adata, compute_autocorr(Spatial_data_adata, df_stplus)))
    print("====stPlus====\n")

def main():
    sub_dirs = np.sort([os.path.join(data_path, dir) for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))])
    
    for i, root in enumerate(sub_dirs):
        evaluate_data(root)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


    
