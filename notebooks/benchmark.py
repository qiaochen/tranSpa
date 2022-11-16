import warnings


from SpaGE.main import SpaGE


def SpaGE_impute(RNA_data, Spatial_data, train_gene, predict_gene):
    RNA_data = RNA_data.loc[(RNA_data.sum(axis=1) != 0)]
    RNA_data = RNA_data.loc[(RNA_data.var(axis=1) != 0)]
    train = train_gene
    predict = predict_gene
    pv = len(train)/2 if len(train) > 50 else len(train)
    pv = min(pv, Spatial_data.shape[0])
    Spatial = Spatial_data[train]
    Img_Genes = SpaGE(Spatial, RNA_data, n_pv = int(pv), genes_to_predict = predict)
    result = Img_Genes[predict]
    return result 


######
# Cell2location from 
# https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/Cell2location_pipeline.py
######

import scanpy as sc
import pandas as pd
import numpy as np
import cell2location
import scvi

from matplotlib import rcParams
from scipy.sparse import csr_matrix
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text

def deconv_cell2loc(adata_rna, adata_spa, celltype_key, output_file_path):    
    # adata_rna.X = csr_matrix(adata_rna.X)
    # adata_spa.X = csr_matrix(adata_spa.X)

    # adata_rna = adata_rna[~adata_rna.obs[celltype_key].isin(np.array(adata_rna.obs[celltype_key].value_counts()[adata_rna.obs[celltype_key].value_counts() <=1].index))]

    # remove cells and genes with 0 counts everywhere
    # sc.pp.filter_genes(adata_rna,min_cells=1)
    # sc.pp.filter_cells(adata_rna,min_genes=1)

    # adata_rna.obs[celltype_key] = pd.Categorical(adata_rna.obs[celltype_key])
    # adata_rna = adata_rna[~adata_rna.obs[celltype_key].isna(), :]

    selected = filter_genes(adata_rna, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

    # filter the object
    adata_rna = adata_rna[:, selected].copy()

    # cell2location.models.RegressionModel.setup_anndata(adata=adata_rna, batch_key="Age", labels_key=celltype_key)

    # create and train the regression model
    
    # mod = RegressionModel(adata_rna)
    # mod.view_anndata_setup()
    # Use all data for training (validation not implemented yet, train_size=1)
    # mod.train(max_epochs=250, batch_size=2500, use_gpu=True)

    # plot ELBO loss history during training, removing first 20 epochs from the plot
    #mod.plot_history(20)

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    # adata_rna = mod.export_posterior(
    #     adata_rna, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
    # )

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_rna.varm.keys():
        inf_aver = adata_rna.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_rna.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_rna.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_rna.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_rna.uns['mod']['factor_names']
    inf_aver.iloc[0:5, 0:5]

    intersect = np.intersect1d(adata_spa.var_names, inf_aver.index)
    adata_spa = adata_spa[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_spa)
    #scvi.data.view_anndata_setup(adata_vis)

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_spa, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection (using default here):
        detection_alpha=200
    )

    mod.train(max_epochs=30000,
            # train using full data (batch_size=None)
            batch_size=None,
            # use all data points in training because
            # we need to estimate cell abundance at all locations
            train_size=1,
            use_gpu=True)

    mod.plot_history(1000)
    # plot ELBO loss history during training, removing first 100 epochs from the plot
    # mod.plot_history(1000)
    # plt.legend(labels=['full data training'])
    adata_spa = mod.export_posterior(
        adata_spa, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )
    print(adata_spa)
    # adata_spa.obsm['q05_cell_abundance_w_sf'].to_csv(output_file_path + '/Cell2location_result.txt')    
    return adata_spa.obsm['q05_cell_abundance_w_sf']


def gimVI_impute(RNA_data_adata, Spatial_data_adata, train_gene, predict_gene, device):
    import scvi
    scvi.settings.verbosity = 0
    from scvi.external import GIMVI
           
    # Genes  = list(Spatial_data_adata.var_names)
    test_gene_idx = [Spatial_data_adata.var_names.get_loc(x) for x in predict_gene]
    train_genes = train_gene
    
    spatial_data_partial = Spatial_data_adata[:, train_genes].copy()
        
    seq_data = RNA_data_adata[:, np.intersect1d(Spatial_data_adata.var_names, RNA_data_adata.var_names)].copy()
    sc.pp.filter_cells(spatial_data_partial, min_counts=1)
    sc.pp.filter_cells(seq_data, min_counts=1)

    GIMVI.setup_anndata(spatial_data_partial)
    GIMVI.setup_anndata(seq_data)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GIMVI(seq_data, spatial_data_partial)    
        model.train(200, use_gpu=device.index)
        _, imputation = model.get_imputed_values(normalized = True)
    imputed = imputation[:, test_gene_idx]
    # result = pd.DataFrame(imputed, columns = test_genes)
    return imputed

def Tangram_impute(RNA_data_adata, 
                   Spatial_data_adata, 
                   train_gene, 
                   predict_gene, 
                   device, 
                   cls_key=None):
    import tangram as tg

    test_list = predict_gene
    train_list = train_gene
    spatial_data_partial = Spatial_data_adata[:, train_list].copy()
    if cls_key is None:
        cls_key = 'leiden'
        RNA_data_adata_label = RNA_data_adata.copy()
        sc.pp.normalize_total(RNA_data_adata_label)
        sc.pp.log1p(RNA_data_adata_label)
        sc.pp.highly_variable_genes(RNA_data_adata_label)
        RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
        sc.pp.scale(RNA_data_adata_label, max_value=10)
        sc.tl.pca(RNA_data_adata_label)
        sc.pp.neighbors(RNA_data_adata_label)
        sc.tl.leiden(RNA_data_adata_label, resolution = 0.5)
        RNA_data_adata.obs[cls_key]  = RNA_data_adata_label.obs.leiden

    tg.pp_adatas(RNA_data_adata, spatial_data_partial, genes=None)
    ad_map = tg.map_cells_to_space(RNA_data_adata, spatial_data_partial, device = device, mode='clusters', cluster_label = cls_key, density_prior='rna_count_based')
    ad_ge = tg.project_genes(ad_map, RNA_data_adata, cluster_label=cls_key)
    test_list = list(map(lambda x:x.lower(), test_list))
    overlapped = np.intersect1d(ad_ge.var_names, test_list)
    no_pred_gene = np.setdiff1d(test_list, overlapped)
    pre_gene = pd.DataFrame(ad_ge[:,overlapped].X, index=ad_ge[:,overlapped].obs_names, columns=overlapped)
    if len(no_pred_gene) > 0:
        print(f'{len(no_pred_gene)} genes missing in tangram output.')
        for g in no_pred_gene:
            pre_gene[g] = 0
    return pre_gene[test_list].values

def stPlus_impute(RNA_data, Spatial_data, train_gene, predict_gene, device):
    import stPlus
    stPlus_res = stPlus.stPlus(Spatial_data[train_gene], RNA_data.T, predict_gene, "tmp", verbose=False, device=device)
    return stPlus_res     