import os
import pickle
import warnings
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd
import spatialdm as sdm

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "melanoma.pkl"
spec = "human"
ST_name = "melanoma5fold"
# seed = 10

if __name__ == "__main__":
    spa_adata = sc.read_h5ad("../../data/ST/melanoma/spatial.h5ad")
    pd.DataFrame(spa_adata.obsm['spatial'], index=spa_adata.obs_names, columns=['x','y']).to_csv(f"../../output/locations/{ST_name}.csv", index=False)
    spa_adata = spa_adata[:, (spa_adata.var_names.values != 'MARCH1') & (spa_adata.var_names.values != 'MARCH2')].copy()
    rna_adata = sc.read_h5ad("../../data/scRNAseq/melanoma/Tirosh_raw.h5ad")
    rna_adata = rna_adata[:, (rna_adata.var_names.values != 'MARCH1') & (rna_adata.var_names.values != 'MARCH2')].copy()

    classes, ct_list = leiden_cluster(rna_adata, False)
    cls_key = 'leiden'
    rna_adata.obs[cls_key] = classes
    sq.gr.spatial_neighbors(spa_adata, coord_type="grid", n_neighs=4)
    sq.gr.spatial_autocorr(
        spa_adata,
        n_jobs=10,
    )
    sc.pp.filter_genes(rna_adata, min_cells=10)
    sc.pp.highly_variable_genes(spa_adata, n_top_genes=5000)
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


