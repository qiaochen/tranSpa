import os
import pickle
import warnings
import anndata
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd
import spatialdm as sdm

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "mouseliver.pkl"
spec = "mouse"
ST_name = "mouseliver5fold"

if __name__ == "__main__":
    samples = ['c1']
    rna_adata = anndata.concat(
        [sc.read_csv(f"../../data/scRNAseq/MouseLiver/SCP2045/expression/processed_{sub}.csv.gz").T.copy() for sub in samples], join="inner")
    rna_adata.obs = pd.read_csv("../../data/scRNAseq/MouseLiver/SCP2045/metadata/meta_data_manual.csv", index_col=0).loc[rna_adata.obs_names]
    rna_adata.obsm['X_umap'] = pd.read_csv("../../data/scRNAseq/MouseLiver/SCP2045/cluster/clustering_manual.csv", index_col=0).loc[rna_adata.obs_names][["X", "Y"]].values.astype(np.float32)
    rna_adata.obs['celltype'] = pd.read_csv("../../data/scRNAseq/MouseLiver/SCP2045/cluster/clustering_manual.csv", index_col=0).loc[rna_adata.obs_names]["Category"].values


    spa_adata = anndata.concat(
        [sc.read_csv(f"../../data/ST/MouseLiver/SCP2046/expression/processed_{sub}.csv.gz").T.copy() for sub in samples], join="inner")
    spa_adata.obs = pd.read_csv("../../data/ST/MouseLiver/SCP2046/metadata/meta_data.csv", index_col=0).loc[spa_adata.obs_names]
    spa_adata.obsm['X_umap'] = pd.read_csv("../../data/ST/MouseLiver/SCP2046/cluster/clustering.csv", index_col=0).loc[spa_adata.obs_names][["X", "Y"]].values.astype(np.float32)
    spa_adata.obs['celltype'] = pd.read_csv("../../data/ST/MouseLiver/SCP2046/cluster/clustering.csv", index_col=0).loc[spa_adata.obs_names]["Category"].values
    spa_adata.obsm['spatial'] = pd.concat([pd.read_csv(f"../../data/ST/MouseLiver/SCP2046/cluster/spatial_{sb}.csv", index_col=0).iloc[1:] for sb in samples], axis=0).loc[spa_adata.obs_names].values.astype(np.float32)
    pd.DataFrame(spa_adata.obsm['spatial'], columns=['x','y']).to_csv(f"../../output/locations/{ST_name}.csv", index=False)

    classes, ct_list = leiden_cluster(rna_adata, False)
    cls_key = 'leiden'
    rna_adata.obs[cls_key] = classes
    sq.gr.spatial_neighbors(spa_adata, coord_type="grid", n_neighs=6)
    sq.gr.spatial_autocorr(
        spa_adata,
        n_jobs=10,
    )

    sc.pp.filter_genes(rna_adata, min_cells=10)
    sc.pp.filter_genes(spa_adata, min_cells=3)
    # sc.pp.highly_variable_genes(rna_adata, n_top_genes=10000)
    # rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()
    sdm.extract_lr(spa_adata, spec, min_cell=0)
    spa_genes = set()
    for pair in spa_adata.uns['geneInter'].interaction_name:
        _genes = pair.split('_')
        for g in _genes:
            for _g in g.split(":"):
                spa_genes.add(_g)
    for gene in spa_adata.uns['moranI'].index[spa_adata.uns['moranI'].pval_norm_fdr_bh <= 0.01].values:
        spa_genes.add(gene)
    sc.pp.highly_variable_genes(spa_adata, n_top_genes=5000)
    for gene in spa_adata.var_names[spa_adata.var.highly_variable]:    
        spa_genes.add(gene)    
        
    print(len(np.intersect1d(list(spa_genes), spa_adata.var_names)), len(spa_genes), spa_adata.n_vars)
    spa_adata = spa_adata[:, np.intersect1d(list(spa_genes), spa_adata.var_names)]
    

    raw_spatial_df  = pd.DataFrame(spa_adata.X, columns=spa_adata.var_names)
    raw_scrna_df    = pd.DataFrame(rna_adata.X, columns=rna_adata.var_names)

    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([spa_adata, rna_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


