import os
import pickle
import warnings
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "seqFISH_single_cell.pkl"
# seed = 10

if __name__ == "__main__":
    spa_adata = sc.read_h5ad("../../data/ST/seqfish/seqfish_data.h5ad")
    scrna_adata = sc.read_h5ad("../../data/scRNAseq/seqfish/scRNAseq_seqfish.h5ad")
    classes, ct_list = leiden_cluster(scrna_adata)
    cls_key = 'leiden'
    sc.pp.normalize_total(spa_adata)
    sc.pp.normalize_total(scrna_adata)
    sc.pp.log1p(spa_adata)
    sc.pp.log1p(scrna_adata)

    var_name = scrna_adata.var_names.values.copy()
    var_name[np.argmax((scrna_adata.var_names == "Prkcdbp"))] = "Cavin3"
    scrna_adata.var_names = var_name

    scrna_adata.obs[cls_key] = classes

    raw_spatial_df  = pd.DataFrame(spa_adata.X.toarray(), columns=spa_adata.var_names)
    raw_spatial_df.to_csv('../../output/seqfish_raw.csv')
    raw_scrna_df    = pd.DataFrame(scrna_adata.X.toarray(), columns=scrna_adata.var_names).astype(pd.SparseDtype("float32", 0))
    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)

    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.x_global_affine.values.reshape(-1,1), spa_adata.obs.y_global_affine.values.reshape(-1,1)])
    np.save('../../output/seqfish_locations.npy', spa_adata.obsm['spatial'])
    sq.gr.spatial_neighbors(spa_adata)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([spa_adata, scrna_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


