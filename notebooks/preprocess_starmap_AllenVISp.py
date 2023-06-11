import os
import pickle
import warnings
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "starmap_allenvisp.pkl"
# seed = 10

if __name__ == "__main__":
    spa_counts = np.load('../../data/ST/Starmap/visual_1020/20180505_BY3_1kgenes/cell_barcode_count.npy')
    genes = pd.read_csv("../../data/ST/Starmap/visual_1020/20180505_BY3_1kgenes/genes.csv", header=None).iloc[:,0]
    coordinates = pd.read_csv("../../data/ST/Starmap/visual_1020/20180505_BY3_1kgenes/centroids.tsv", header=None, sep='\t')

    spa_adata = sc.AnnData(spa_counts)
    sc.pp.normalize_total(spa_adata)
    sc.pp.log1p(spa_adata)
    spa_adata.obs['X'] = coordinates.iloc[:, 0].values
    spa_adata.obs['Y'] = coordinates.iloc[:, 1].values
    spa_adata.var_names = genes

    VISp_adata = sc.read("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_exon-matrix.csv").T
    genes = pd.read_csv("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_genes-rows.csv", header=0,sep=',')
    VISp_meta = pd.read_csv("../../data/scRNAseq/Allen_VISp/mouse_VISp_2018-06-14_samples-columns.csv", header=0,sep=',')
    VISp_adata.obs = VISp_meta
    VISp_adata.var_names = genes.gene_symbol

    sc.pp.filter_genes(VISp_adata, min_cells=10)
    VISp_adata = VISp_adata[(VISp_adata.obs['class'] != 'No Class') & (VISp_adata.obs['class'] != 'Low Quality')]
    classes, ct_list = leiden_cluster(VISp_adata)
    cls_key = 'leiden'
    VISp_adata.obs[cls_key] = classes
    sc.pp.normalize_total(VISp_adata)
    sc.pp.log1p(VISp_adata)
    

    raw_spatial_df  = pd.DataFrame(spa_adata.X, columns=spa_adata.var_names)
    raw_scrna_df    = pd.DataFrame(VISp_adata.X, columns=VISp_adata.var_names)

    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    raw_spatial_df.to_csv('../../output/starmap_raw.csv')
    
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.X.values.reshape(-1,1), spa_adata.obs.Y.values.reshape(-1,1)])
    np.save('../../output/starmap_locations.npy', spa_adata.obsm['spatial'])
    sq.gr.spatial_neighbors(spa_adata)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([spa_adata, VISp_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


