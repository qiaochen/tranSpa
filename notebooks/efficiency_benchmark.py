from scipy import sparse
from benchmark import Tangram_impute, SpaGE_impute
from transpa.util import expTransImp, leiden_cluster

import stPlus

from locale import normalize
import torch

import os, argparse, time
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')


seed = 10
device = torch.device(
    "cuda:3") if torch.cuda.is_available() else torch.device("cpu")

def load_AllenVISp():
    VISp_adata = sc.read(
        "../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_exon-matrix.csv").T
    genes = pd.read_csv(
        "../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_genes-rows.csv", header=0, sep=',')
    VISp_meta = pd.read_csv(
        "../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_samples-columns.csv", header=0, sep=',')

    VISp_adata.obs = VISp_meta
    VISp_adata.var_names = genes.gene_symbol

    sc.pp.filter_genes(VISp_adata, min_cells=10)
    VISp_adata = VISp_adata[(VISp_adata.obs['class'] != 'No Class') & (
        VISp_adata.obs['class'] != 'Low Quality')]
    classes, ct_list = leiden_cluster(VISp_adata)
    cls_key = 'leiden'
    VISp_adata.obs[cls_key] = classes
    sc.pp.normalize_total(VISp_adata)
    sc.pp.log1p(VISp_adata)
    raw_scrna_df = pd.DataFrame(VISp_adata.X, columns=VISp_adata.var_names)#.astype(pd.SparseDtype("float32", 0))
    return VISp_adata, raw_scrna_df


def load_osmFISH():
    spatial_df_file = '/data/users/cqiao/data/stPlus/data/osmFISH_df.csv'
    spatial_loom_file = '/data/users/cqiao/data/stPlus/data/osmFISH_SScortex_mouse_all_cells.loom'
    osmFISH = sc.read_loom(spatial_loom_file)
    osmFISH = osmFISH[~np.isin(osmFISH.obs.Region, [
                               'Excluded', 'Hippocampus', 'Internal Capsule Caudoputamen', 'Ventricle', 'White matter'])]
    raw_spatial_df = pd.read_csv(spatial_df_file)
    osmFISH.X = raw_spatial_df.values
    osmFISH.obsm['spatial'] = np.hstack([osmFISH.obs.X.values.reshape(-1,1), osmFISH.obs.Y.values.reshape(-1,1)])
    sq.gr.spatial_neighbors(osmFISH)
    return osmFISH, raw_spatial_df


def load_starmap():
    spa_counts = np.load(
        '/data/users/cqiao/data/stPlus/data/SpaGE Datasets/Spatial/Starmap/visual_1020/20180505_BY3_1kgenes/cell_barcode_count.npy')
    genes = pd.read_csv(
        "/data/users/cqiao/data/stPlus/data/SpaGE Datasets/Spatial/Starmap/visual_1020/20180505_BY3_1kgenes/genes.csv", header=None).iloc[:, 0]
    coordinates = pd.read_csv(
        "/data/users/cqiao/data/stPlus/data/SpaGE Datasets/Spatial/Starmap/visual_1020/20180505_BY3_1kgenes/centroids.tsv", header=None, sep='\t')

    spa_adata = sc.AnnData(spa_counts)
    sc.pp.normalize_total(spa_adata)
    sc.pp.log1p(spa_adata)
    spa_adata.obs['X'] = coordinates.iloc[:, 0].values
    spa_adata.obs['Y'] = coordinates.iloc[:, 1].values
    spa_adata.var_names = genes
    raw_spatial_df = pd.DataFrame(spa_adata.X, columns=spa_adata.var_names)
    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.X.values.reshape(-1,1), spa_adata.obs.Y.values.reshape(-1,1)])
    sq.gr.spatial_neighbors(spa_adata)
    return spa_adata, raw_spatial_df


def load_seqfish():
    spa_adata = sc.read_h5ad(
        "/data/users/cqiao/notebooks/projects/spatial/seqfishdata/seqfish_data.h5ad")
    sc.pp.normalize_total(spa_adata)
    sc.pp.log1p(spa_adata)
    raw_spatial_df = pd.DataFrame(
        spa_adata.X.toarray(), columns=spa_adata.var_names)
    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.x_global_affine.values.reshape(-1,1), spa_adata.obs.y_global_affine.values.reshape(-1,1)])
    sq.gr.spatial_neighbors(spa_adata)

    return spa_adata, raw_spatial_df


def load_seqFISHsinglecell():
    scrna_adata = sc.read_h5ad(
        "/data/users/cqiao/notebooks/projects/spatial/seqfishdata/scRNAseq_seqfish.h5ad")
    classes, ct_list = leiden_cluster(scrna_adata)
    cls_key = 'leiden'
    sc.pp.normalize_total(scrna_adata)
    sc.pp.log1p(scrna_adata)
    var_name = scrna_adata.var_names.values.copy()
    var_name[np.argmax((scrna_adata.var_names == "Prkcdbp"))] = "Cavin3"
    scrna_adata.var_names = var_name
    scrna_adata.obs[cls_key] = classes
    raw_scrna_df = pd.DataFrame(scrna_adata.X.toarray(
    ), columns=scrna_adata.var_names)#.astype(pd.SparseDtype("float32", 0))
    return scrna_adata, raw_scrna_df


def load_merfish():
    merfish_path = '../data/merfish.h5ad'
    if os.path.exists(merfish_path):
        spa_adata = sc.read(merfish_path)
    spa_adata.var_names_make_unique()
    raw_spatial_df = pd.DataFrame(spa_adata.X, columns=spa_adata.var_names)
    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.X.values.reshape(-1,1), spa_adata.obs.Y.values.reshape(-1,1)])
    sq.gr.spatial_neighbors(spa_adata)
    return spa_adata, raw_spatial_df


def load_moffit():
    Moffit_path = '../data/moffit_adata.h5ad'

    if os.path.exists(Moffit_path):
        Moffit_adata = sc.read(Moffit_path)

    Moffit_adata.var_names_make_unique()
    ct_list = np.unique(Moffit_adata.obs.leiden)
    classes = Moffit_adata.obs.leiden.values
    cls_key = 'leiden'
    raw_scrna_df = pd.DataFrame(Moffit_adata.X.toarray(
    ), columns=Moffit_adata.var_names)#.astype(pd.SparseDtype("float32", 0))
    return Moffit_adata, raw_scrna_df


def dataset_osmFISH_AllenVISp():
    rna_adata, rna_df = load_AllenVISp()
    spa_adata, spa_df = load_osmFISH()
    raw_shared_gene = np.intersect1d(spa_df.columns, rna_df.columns)
    return spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene

def dataset_starmap_AllenVISp():
    rna_adata, rna_df = load_AllenVISp()
    spa_adata, spa_df = load_starmap()
    raw_shared_gene = np.intersect1d(spa_df.columns, rna_df.columns)
    return spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene

def dataset_merfish_moffit():
    rna_adata, rna_df = load_moffit()
    spa_adata, spa_df = load_merfish()
    raw_shared_gene = np.intersect1d(spa_df.columns, rna_df.columns)
    return spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene

def dataset_seqfish_singlecell():
    rna_adata, rna_df = load_seqFISHsinglecell()
    spa_adata, spa_df = load_seqfish()
    raw_shared_gene = np.intersect1d(spa_df.columns, rna_df.columns)
    return spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene

def method_transImp(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        expTransImp(
            df_ref=rna_df,
            df_tgt=spa_df,
            train_gene=train_gene,
            test_gene=test_gene,
            mapping_lowdim=128,
            signature_mode='cell',
            mapping_mode='lowrank',
            seed=seed,
            device=device)
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"TransImp Avg Runtime: {avg_time:.4}")
    return avg_time

def method_transImpCls(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    classes = rna_adata.obs.leiden
    ct_list = rna_adata.obs.leiden.unique()
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        expTransImp(
            df_ref=rna_df,
            df_tgt=spa_df,
            train_gene=train_gene,
            test_gene=test_gene,
            ct_list=ct_list,
            classes=classes,
            signature_mode='cluster',
            mapping_mode='full',
            seed=seed,
            device=device)
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"TransImpCls Avg Runtime: {avg_time:.4}")
    return avg_time

def method_transImpSpa(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    spa_adj = spa_adata.obsp['spatial_connectivities'].tocoo()
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        expTransImp(
            df_ref=rna_df,
            df_tgt=spa_df,
            train_gene=train_gene,
            test_gene=test_gene,
            signature_mode='cell',
            mapping_mode='lowrank',
            mapping_lowdim=128,
            wt_spa=1,
            spa_adj=spa_adj,
            seed=seed,
            device=device)
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"TransImpSpa Avg Runtime: {avg_time:.4}")
    return avg_time


def method_transImpClsSpa(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    classes = rna_adata.obs.leiden
    ct_list = rna_adata.obs.leiden.unique()
    spa_adj = spa_adata.obsp['spatial_connectivities'].tocoo()
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        expTransImp(
            df_ref=rna_df,
            df_tgt=spa_df,
            train_gene=train_gene,
            test_gene=test_gene,
            ct_list=ct_list,
            classes=classes,
            spa_adj=spa_adj,
            signature_mode='cluster',
            mapping_mode='full',
            wt_spa=1,
            seed=seed,
            device=device
        )
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"TransImpClsSpa Avg Runtime: {avg_time:.4}")
    return avg_time


def method_tangram(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        Tangram_impute(rna_adata, spa_adata, train_gene,
                       test_gene, device, "leiden")
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"Tangram Avg Runtime: {avg_time:.4}")
    return avg_time


def method_spaGE(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        SpaGE_impute(rna_df, spa_df, train_gene, test_gene)
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"spaGE Avg Runtime: {avg_time:.4}")
    return avg_time

def method_stPlus(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=None, device=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)
    time_costs = []
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]
        start = time.time()
        stPlus.stPlus(
                    spa_df[train_gene],
                    rna_df,
                    test_gene,
                    f"tmp{time.time()}",
                    verbose=False,
                    random_seed=seed,
                    device=device)
        end  = time.time()
        time_costs.append(end - start)
    avg_time = np.mean(time_costs)
    print(f"stPlus Avg Runtime: {avg_time:.4}")
    return avg_time


dict_datasets = dict(osmFISH_AllenVISp=dataset_osmFISH_AllenVISp, 
        starmap_AllenVISp=dataset_starmap_AllenVISp,
        merfish_moffit=dataset_merfish_moffit,
        seqFISH_SingleCell=dataset_seqfish_singlecell,
    )
dict_methods = dict(tangram=method_tangram, stPlus=method_stPlus, spaGE=method_spaGE,
                    transImpClsSpa=method_transImpClsSpa, transImp=method_transImp, transImpCls=method_transImpCls,
                    transImpSpa=method_transImpSpa)
    
def exp(method_name, dataset_name):
    print(method_name, dataset_name)
    method_fn, dataset_fn = dict_methods[method_name], dict_datasets[dataset_name]
    spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene = dataset_fn()
    
    print(f"{dataset_name}, spa shape: {spa_df.shape[0]} X {spa_df.shape[1]}, rna shape: {rna_df.shape[0]} X {rna_df.shape[1]}")
    avg_time = method_fn(spa_adata, rna_adata, spa_df, rna_df, raw_shared_gene, seed=seed, device=device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='benchmark efficiency')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--method_name', type=str, default=None)
    args = parser.parse_args()
    exp(args.method_name, args.dataset_name)

    