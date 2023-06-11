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
dataset_path = "breastcancer.pkl"
spec = "human"
ST_name = "breastcancer5fold"

def validate_genes():
    from pyensembl import EnsemblRelease

    # release 77 uses human reference genome GRCh38
    data = EnsemblRelease(108)
    res = []
    valid_ids = []
    counter = 0
    for gid in rna_adata.var_names:
        try:
            name = data.gene_name_of_gene_id(gid)
            if len(name) == 0:
                raise Exception("None name")
            res.append(name)
            valid_ids.append(gid)
        except Exception as e:
            counter += 1
            
    print(f"Missed genes {counter}")    
    return res, valid_ids

if __name__ == "__main__":
    rna_adata = sc.read_mtx("../../data/scRNAseq/BreastCancer/matrix.mtx.gz").T
    rna_adata.var_names = pd.read_csv("../../data/scRNAseq/BreastCancer/features.tsv", header=None).iloc[:, 0].values
    rna_adata.obs_names = pd.read_csv("../../data/scRNAseq/BreastCancer/barcodes.tsv", header=None).iloc[:, 0].values
    
    gene_names, valid_ids = validate_genes()
    rna_adata = rna_adata[:, valid_ids].copy()
    rna_adata.var_names = gene_names
    ids, cts = np.unique(rna_adata.var_names, return_counts=True)
    deduplicated = [id for id,ct in zip(ids,cts) if ct==1]
    rna_adata.var_names_make_unique()
    rna_adata = rna_adata[:, deduplicated]

    spa_adata = sc.read_mtx("../../data/ST/BreastCancer/filtered_count_matrices/1142243F_filtered_count_matrix/matrix.mtx").T
    spa_adata.var_names = pd.read_csv("../../data/ST/BreastCancer/filtered_count_matrices/1142243F_filtered_count_matrix/features.tsv", header=None).iloc[:, 0].values
    spa_adata.obs_names = pd.read_csv("../../data/ST/BreastCancer/filtered_count_matrices/1142243F_filtered_count_matrix/barcodes.tsv", header=None).iloc[:, 0].values
    df_obs = pd.read_csv("../../data/ST/BreastCancer/metadata/1142243F_metadata.csv", header=0, index_col=0)
    spa_adata.obs = df_obs.loc[spa_adata.obs_names]
    df_loc = pd.read_csv("../../data/ST/BreastCancer/spatial/1142243F_spatial/tissue_positions_list.csv", header=None, index_col=0).loc[spa_adata.obs_names]
    spa_adata.obs['In_tissue'] = df_loc.iloc[:, 0]
    spa_adata.obs['array_row'] = df_loc.iloc[:, 1]
    spa_adata.obs['array_col'] = df_loc.iloc[:, 2]
    spa_adata.obs['px_row'] = df_loc.iloc[:, 3]
    spa_adata.obs['px_col'] = df_loc.iloc[:, 4]
    # sc.pp.normalize_total(rna_adata)
    # sc.pp.log1p(rna_adata)
    spa_adata.obsm['spatial'] = spa_adata.obs[['px_row', 'px_col']].values
    pd.DataFrame(spa_adata.obsm['spatial'], columns=['x','y']).to_csv(f"../../output/locations/{ST_name}.csv", index=False)
    classes, ct_list = leiden_cluster(rna_adata, False)

    cls_key = 'leiden'
    rna_adata.obs[cls_key] = classes
    sq.gr.spatial_neighbors(spa_adata, coord_type="grid", n_neighs=6)
    sq.gr.spatial_autocorr(
        spa_adata,
        n_jobs=10,
    )

    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)
    sc.pp.normalize_total(spa_adata)
    sc.pp.log1p(spa_adata)

    sc.pp.filter_genes(rna_adata, min_cells=30)
    sc.pp.filter_genes(spa_adata, min_cells=3)
    # sc.pp.highly_variable_genes(rna_adata, n_top_genes=10000)
    # rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()

    sc.pp.highly_variable_genes(spa_adata, n_top_genes=5000)
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
    
    raw_spatial_df  = pd.DataFrame(spa_adata.X.toarray(), columns=spa_adata.var_names)
    raw_scrna_df    = pd.DataFrame(rna_adata.X.toarray(), columns=rna_adata.var_names)
    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([spa_adata, rna_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


