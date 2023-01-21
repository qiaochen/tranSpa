import torch
import numpy as np
import pandas as pd

from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error


def mse_moranI(truth_adata, imp_adata):
    valid_sel = ~np.isnan(imp_adata.uns['moranI'].I) & (~np.isnan(truth_adata.uns['moranI'].I))
    n_valid = np.sum(valid_sel)
    if n_valid < truth_adata.uns['moranI'].shape[0]:
        print(f"MoranI nan values: {np.sum(~valid_sel)}/{truth_adata.uns['moranI'].shape[0]}")
    if n_valid > 0:
        return mean_squared_error(truth_adata.uns['moranI'].I[valid_sel], imp_adata.uns['moranI'].I[valid_sel])
    else:
        return np.inf

def mse_gearyC(truth_adata, imp_adata):
    valid_sel = (~np.isnan(imp_adata.uns['gearyC'].C)) # (~np.isnan(truth_adata.uns['gearyC'].C)) & 
    n_valid = np.sum(valid_sel)
    if n_valid < truth_adata.uns['gearyC'].C.shape[0]:
        print(f"MoranI nan values: {np.sum(~valid_sel)}/{truth_adata.uns['gearyC'].C.shape[0]}")
    if n_valid > 0:
        return mean_squared_error(truth_adata.uns['gearyC'].C[valid_sel], imp_adata.uns['gearyC'].C[valid_sel])
    else:
        return np.inf

def standardize(X):
    return (X - torch.mean(X, dim=0, keepdim=True)) / torch.std(X, dim=0, keepdim=True)        

def spark_stat(loc, expr):
    # cell/spot by gene
    loc, expr = torch.FloatTensor(loc), torch.FloatTensor(expr)
    centered_expr = standardize(expr)
    cent_loc = standardize(loc)
    STS_inv = torch.inverse(cent_loc.t() @ cent_loc)
    N = centered_expr.shape[0]
    EHL = centered_expr.t() @ loc
     
    numerator = torch.sum((EHL @ STS_inv) * EHL, dim=1)
    denominator = torch.square(centered_expr).sum(axis=0)
    test_stats = numerator * N / denominator
    return test_stats.cpu().numpy()

def cal_ssim(im1,im2,M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12    
    return ssim

def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result

def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = stats.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result

def logNorm(df):
    df = np.log1p(df)
    df = stats.zscore(df)
    return df
    
class CalculateMeteics:
    def __init__(self, raw_count_file=None, impute_count_file=None, raw_df=None, imp_df=None, prefix=None):
        if raw_df is None:            
            self.raw_count = pd.read_csv(raw_count_file, header = 0, sep="\t")
            self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
            self.raw_count = self.raw_count.T
            self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        else:
            self.raw_count = raw_df
        self.raw_count = self.raw_count.fillna(1e-20)
        
        if imp_df is None:
            self.impute_count = pd.read_csv(impute_count_file, header = 0, index_col = 0)
            self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
            self.impute_count = self.impute_count.T
            self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        else:
            self.impute_count = imp_df
        self.impute_count = self.impute_count.fillna(-1)        
        self.prefix = prefix
        
    def SSIM(self, raw, impute, scale = 'scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print ('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(-1)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                    ssim = cal_ssim(raw_col_2,impute_col_2,M)
                    
                if np.mean(impute_col) == -1:
                    ssim = 0
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
                result = pd.concat([result, ssim_df],axis=1)
        else:
            print("columns error")
        return result
            
    def PCC(self, raw, impute, scale = None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    pearsonr = -1
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(-1)
                    raw_col = raw_col.fillna(1e-20)
                    pearsonr, _ = stats.pearsonr(raw_col,impute_col)

                if np.mean(impute_col) == -1:
                    pearsonr = -1   
                pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[label])
                
                result = pd.concat([result, pearson_df],axis=1)
        else:
            print("columns error")
        return result

    def SCC(self, raw, impute, scale = None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    spearmanr = -1
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(-1)
                    raw_col = raw_col.fillna(1e-20)
                    spearmanr, _ = stats.spearmanr(raw_col,impute_col)

                if np.mean(impute_col) == -1:
                    spearmanr = -1      
                spearmanr_df = pd.DataFrame(spearmanr, index=["SCC"],columns=[label])
                result = pd.concat([result, spearmanr_df],axis=1)
        else:
            print("columns error")
        return result    
    
    def JS(self, raw, impute, scale = 'scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print ('Please note you do not scale data by plus')    
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    print(f'{label} not in column, set JS np.inf')
                    JS = np.inf
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(-1)
                    M = (raw_col + impute_col)/2
                    JS = 0.5*stats.entropy(raw_col,M)+0.5*stats.entropy(impute_col,M)

                if np.mean(impute_col) == -1:
                    print("Imputation values are all nan, set JS inf")
                    JS = np.inf   
                JS_df = pd.DataFrame(JS, index=["JS"],columns=[label])
                result = pd.concat([result, JS_df],axis=1)
        else:
            print("columns error")
        return result
    
    def RMSE(self, raw, impute, scale = 'zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print ('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = np.inf   
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(-1)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                if np.mean(impute_col) == -1:
                    RMSE = np.inf
                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                result = pd.concat([result, RMSE_df],axis=1)
        else:
            print("columns error")
        return result       
        
    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM = self.SSIM(raw,impute)
        Pearson = self.PCC(raw, impute)
        Spearman = self.SCC(raw, impute)
        JS = self.JS(raw, impute)
        RMSE = self.RMSE(raw, impute)
        
        result_all = pd.concat([Pearson, Spearman, SSIM, RMSE, JS],axis=0)
        if not prefix is None:
            result_all.T.to_csv(prefix + "_Metrics.txt", sep='\t', header = 1, index = 1)
        self.accuracy = result_all
        return result_all

def calc_corr(spatial_df, pred_res, test_gene, is_pearsonr=False, use_cosine=True):
    """
    spatial_df: original spatial data (cell by gene dataframe)
    pred_res: predicted results (cell by gene dataframe)
    test_gene: genes to calculate Spearman correlation
    """
    correlation = []
    for gene in test_gene:
        if is_pearsonr:
            correlation.append(stats.pearsonr(spatial_df[gene], pred_res[gene])[0])
        elif not use_cosine:
            correlation.append(stats.spearmanr(spatial_df[gene], pred_res[gene])[0])
        else:
            correlation.append(1-cosine(spatial_df[gene], pred_res[gene]))
        
    return correlation

def calc_mse(spatial_df, pred_res, test_gene):
    """
    spatial_df: original spatial data (cell by gene dataframe)
    pred_res: predicted results (cell by gene dataframe)
    test_gene: genes to calculate Spearman correlation
    """
    mses = []
    for gene in test_gene:
        mses.append(mean_squared_error(spatial_df[gene], pred_res[gene]))
    return mses