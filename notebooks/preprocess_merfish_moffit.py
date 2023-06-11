import os
import pickle
import warnings
import squidpy as sq
import numpy as np
import scanpy as sc
import pandas as pd

from transpa.util import leiden_cluster


OUTROOT = "../../output/preprocessed_dataset"
dataset_path = "merfish_moffit.pkl"
# seed = 10

if __name__ == "__main__":
    merfish = pd.read_csv('../../data/ST/Merfish/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv')
    merfish_1 = merfish.loc[merfish['Animal_ID'] == 1, :]
    merfish_1 = merfish_1.loc[merfish_1['Cell_class'] != 'Ambiguous',:]
    merfish_meta = merfish_1.iloc[:,0:9]
    merfish_data = merfish_1.iloc[:,9:171]
    merfish_data = merfish_data.drop(columns = ['Blank_1','Blank_2','Blank_3','Blank_4','Blank_5','Fos'])

    spa_adata = sc.AnnData(merfish_data.values)
    spa_adata.obs['X'] = merfish_1.Centroid_X.values
    spa_adata.obs['Y'] = merfish_1.Centroid_Y.values
    spa_adata.var_names = merfish_data.columns.values
    spa_adata.var_names_make_unique()
    sc.pp.normalize_total(spa_adata)
    sc.pp.log1p(spa_adata)

    Moffit_adata = sc.read_mtx("../../data/scRNAseq/Moffit/GSE113576/matrix.mtx").T
    genes = pd.read_csv('../../data/scRNAseq/Moffit/GSE113576/genes.tsv',sep='\t',header=None).loc[:, 1].values
    barcodes = pd.read_csv('../../data/scRNAseq/Moffit/GSE113576/barcodes.tsv',sep='\t',header=None).loc[:, 0].values

    Moffit_adata.var_names = genes
    Moffit_adata.obs_names = barcodes
    Moffit_adata.var_names_make_unique()
    classes, ct_list = leiden_cluster(Moffit_adata)
    cls_key = 'leiden'
    Moffit_adata.obs[cls_key] = classes
    sc.pp.filter_genes(Moffit_adata, min_cells=10)
    sc.pp.normalize_total(Moffit_adata)
    sc.pp.log1p(Moffit_adata)
            
    spa_adata.var_names_make_unique()
    Moffit_adata.var_names_make_unique()
    ct_list = np.unique(Moffit_adata.obs.leiden)
    classes = Moffit_adata.obs.leiden.values
    cls_key = 'leiden'

    raw_spatial_df  = pd.DataFrame(spa_adata.X, columns=spa_adata.var_names)
    # raw_scrna_df    = pd.DataFrame.sparse.from_spmatrix(Moffit_adata.X, columns=Moffit_adata.var_names)
    raw_scrna_df    = pd.DataFrame(Moffit_adata.X.toarray(), columns=Moffit_adata.var_names)
    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)

    raw_spatial_df.to_csv('../../output/merfish_raw.csv')
    print(raw_spatial_df.shape, raw_scrna_df.shape, raw_shared_gene.shape)

    spa_adata.obsm['spatial'] = np.hstack([spa_adata.obs.X.values.reshape(-1,1), spa_adata.obs.Y.values.reshape(-1,1)])
    np.save('../../output/merfish_locations.npy', spa_adata.obsm['spatial'])
    sq.gr.spatial_neighbors(spa_adata)

    with open(os.path.join(OUTROOT, dataset_path), 'wb') as outfile:
        pickle.dump([spa_adata, Moffit_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene], outfile)


