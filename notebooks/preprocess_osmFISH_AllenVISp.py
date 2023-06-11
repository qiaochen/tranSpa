import os
import pickle
import warnings
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "osmFISH_allenvisp.pkl"
# seed = 10

if __name__ == "__main__":
    spatial_df_file = '../../data/ST/osmFISH/osmFISH_df.csv'
    spatial_loom_file = '../../data/ST/osmFISH/osmFISH_SScortex_mouse_all_cells.loom'
    
    VISp_adata = sc.read("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_exon-matrix.csv").T
    genes = pd.read_csv("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_genes-rows.csv", header=0,sep=',')
    VISp_meta = pd.read_csv("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_samples-columns.csv", header=0,sep=',')

    VISp_adata.obs = VISp_meta
    VISp_adata.var_names = genes.gene_symbol

    sc.pp.filter_genes(VISp_adata, min_cells=10)
    VISp_adata = VISp_adata[(VISp_adata.obs['class'] != 'No Class') & (VISp_adata.obs['class'] != 'Low Quality')]
    # VISp_adata = sc.read_h5ad("./tmp_visp.h5ad")
    classes, ct_list = leiden_cluster(VISp_adata)
    cls_key = 'leiden'
    VISp_adata.obs[cls_key] = classes
    sc.pp.normalize_total(VISp_adata)
    sc.pp.log1p(VISp_adata)
    VISp_adata

    osmFISH = sc.read_loom(spatial_loom_file)
    osmFISH = osmFISH[~np.isin(osmFISH.obs.Region, ['Excluded', 'Hippocampus', 'Internal Capsule Caudoputamen','Ventricle', 'White matter'])].copy()
    raw_spatial_df  = pd.read_csv(spatial_df_file)
    osmFISH.X = raw_spatial_df.values


    raw_scrna_df    = pd.DataFrame(VISp_adata.X, columns=VISp_adata.var_names)
    adata_scrna   = VISp_adata
    raw_spatial_df.to_csv('../../output/osmFISH_raw.csv')

    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    osmFISH.obsm['spatial'] = np.hstack([osmFISH.obs.X.values.reshape(-1,1), osmFISH.obs.Y.values.reshape(-1,1)])
    np.save('../../output/osmFISH_locations.npy', osmFISH.obsm['spatial'])
    sq.gr.spatial_neighbors(osmFISH)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([osmFISH, VISp_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


