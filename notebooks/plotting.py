from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from transpa.util import expTransImp

def get_pred_perf_variability(pre_datapath, seed, device):    
    with open(pre_datapath, 'rb') as infile:
        spa_adata, scrna_adata, raw_spatial_df, raw_scrna_df, raw_shared_gene = pickle.load(infile)
    cls_key = 'leiden'
    classes = scrna_adata.obs[cls_key]
    ct_list = np.unique(classes)  
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf.get_n_splits(raw_shared_gene)

    pred_perf_uncertainty = []
    test_gene_set = []
    
    for idx, (train_ind, test_ind) in enumerate(kf.split(raw_shared_gene)):   
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]

        test_spatial_df = raw_spatial_df[test_gene]
        spatial_df = raw_spatial_df[train_gene]
        scrna_df   = raw_scrna_df

        transImpRes = expTransImp(
                        df_ref=raw_scrna_df,
                        df_tgt=raw_spatial_df,
                        train_gene=train_gene,
                        test_gene=test_gene,
                        n_simulation=200,
                        signature_mode='cell',
                        mapping_mode='lowrank',
                        classes=classes,
                        n_epochs=2000,
                        seed=seed,
                        device=device
        )
        pred_perf_uncertainty.extend(transImpRes[1])
        test_gene_set.extend(test_gene)
    df_gene_var = pd.DataFrame(index=test_gene_set)
    df_gene_var['gene'] = test_gene_set
    df_gene_var['perf_var'] = pred_perf_uncertainty
    
    return df_gene_var

def score_MI(dict_adata, methods, type, dataset, genes=None, thred=0.01):
    if genes is None:
        genes = dict_adata["truth"].var_names
    eval_res = []
    df_moranI_truth = dict_adata["truth"].uns['moranI'].loc[genes].query("I > 0")
    for method in methods:
        df_moranI_m = dict_adata[method].uns['moranI'].loc[df_moranI_truth.index.values]
        sel = (~np.isnan(df_moranI_truth.I.values) ) & (~np.isnan(df_moranI_m.I.values))
        truth_pval = df_moranI_truth.pval_norm_fdr_bh.values[sel]
        score = df_moranI_m.I.values[sel]
        if type == "prec_rec":
            eval_score = metrics.average_precision_score(truth_pval < thred, score)
        elif type == "roc":
            eval_score = metrics.roc_auc_score(truth_pval < thred, score)
        eval_res.append({"method":method, "score":eval_score,"stats":'MoranI', "metric":type, 'dataset':dataset})
    return pd.DataFrame(eval_res)
    
    
def score_sparkX(dict_sparkx_adjpvals, methods, type, dataset, genes=None, thred=0.01):
    df_sparkx_adjpvals = pd.DataFrame(dict_sparkx_adjpvals)
    if not genes is None:
        df_sparkx_adjpvals = df_sparkx_adjpvals.loc[genes].copy()
    eval_res = []
    for method in methods:
        sel = (~np.isnan(df_sparkx_adjpvals['truth'].values) ) & (~np.isnan(df_sparkx_adjpvals[method].values))
        truth_pval = df_sparkx_adjpvals.truth.values[sel]
        score = -np.log(df_sparkx_adjpvals[method][sel].values + min(np.min(df_sparkx_adjpvals[method][sel].values) * 1e-3, 1e-500))
        if type == "prec_rec":
            eval_score = metrics.average_precision_score(truth_pval < thred, score)
        elif type == "roc":
            eval_score = metrics.roc_auc_score(truth_pval < thred, score)
        eval_res.append({"method":method, "score":eval_score, "stats":'SPARKX',"metric":type ,'dataset':dataset})
    return pd.DataFrame(eval_res)
    
def score_SDM(adatas, 
              methods,
              type,
              dataset,
              genes=None,
              thred=0.01):
    if genes is None:
        genes = adatas["truth"].var_names
    
    _genes = set([g.lower() for g in genes])
    sel = []
    for pair in adatas['truth'].uns['global_res'].index.values:
        if np.any([False if g.lower() in _genes else True for g in pair.split('_')]):
            sel.append(False)
        else:
            sel.append(True)
    
    eval_res = []
    for md in methods:
        truth_res = adatas['truth'].uns['global_res'].loc[(adatas['truth'].uns['global_I'] >= 0 ) & np.array(sel)].copy()
        shared_pairs = np.intersect1d(truth_res.index, adatas[md].uns['global_res'].index)
        if len(shared_pairs) < len(truth_res.index):
            print(f"{md} Fewer pairs than truth: {len(shared_pairs)} vs {len(truth_res.index)}")
            
        y = truth_res.loc[shared_pairs].selected.values
        score =  -np.log(adatas[md].uns['global_res'].loc[shared_pairs].fdr.values + min(np.min(adatas[md].uns['global_res'].loc[shared_pairs].fdr.values)/1000, 1e-500))
        score[np.isinf(score)] = score[~np.isinf(score)].max()*10
        
        if type == "prec_rec":
            eval_score = metrics.average_precision_score(y, score)
        elif type == "roc":
            eval_score = metrics.roc_auc_score(y, score)
        eval_res.append({"method":md, "score":eval_score, "stats":'SDM', "metric":type, 'dataset':dataset})
    return pd.DataFrame(eval_res)

def plot_curve_MI(df_corr, 
                 df_I, 
                 df_fdr, 
                 methods, 
                 color, 
                 type,
                 thred=0.01, 
                 excluded_items={'TransImpCls', 'TransImpClsSpa', 'truth'}):
    idx = -1
    plt.rcParams["figure.dpi"] = 380
    _, ax = plt.subplots(figsize=(8, 7))
    for method in methods:
        # predict zero, 1-score
        idx += 1
        if method in excluded_items: continue
        sel = (~np.isnan(df_I['truth'].values) ) & (~np.isnan(df_I[method].values))
        sel = sel & (df_I.truth.values >= 0)
        truth_pval, preds_pval = df_fdr.truth.values[sel], df_fdr[method][sel]
        score = df_I[method][sel]
        print(f'({method}) Valid genes: {sel.sum()}/{sel.shape[0]}, Ground Sig/Total ({(truth_pval < thred).sum()}/{truth_pval.shape[0]}), Pred Sig/total ({(preds_pval < thred).sum()}/{preds_pval.shape[0]})')
        
        if type == "prec_rec":
            prec, rec, _  = metrics.precision_recall_curve(truth_pval < thred, score, pos_label=1)
            disp = metrics.PrecisionRecallDisplay(precision=prec, recall=rec)
            disp.plot(ax=ax, name = f"{method} (AUC=({metrics.average_precision_score(truth_pval < thred, score):.2f})", color=color[idx])
        elif type == "roc":
            fpr, tpr, _  = metrics.roc_curve(truth_pval < thred, score, pos_label=1)
            disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
            disp.plot(ax=ax, name=f"{method} (AUC = {metrics.roc_auc_score(truth_pval < thred, score):.2f})", color=color[idx])
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    if type == "prec_rec":
        ax.set_title(f"Precison-Recall Curve on Moran's I Test")
    elif type == "roc":
        ax.set_title(f"Receiver Operating Characteristic on Moran's I Test")

    ax.legend(loc="best")
    plt.show()
    
    
def plot_curve_SPARKX(
                      df_sparkx_adjpvals,
                      methods, 
                      color, 
                      type,
                      thred=0.01,
                      excluded_items={'TransImpCls', 'TransImpClsSpa', 'truth'}):
    idx = -1
    plt.rcParams["figure.dpi"] = 380
    _, ax = plt.subplots(figsize=(8, 7))
    for method in methods:
        # predict zero, 1-score
        idx += 1
        if method in excluded_items: continue
        
        sel = (~np.isnan(df_sparkx_adjpvals['truth'].values) ) & (~np.isnan(df_sparkx_adjpvals[method].values))
        print(f'({method}) Valid genes: {sel.sum()}/{sel.shape[0]}')
        truth_pval, method_pval = df_sparkx_adjpvals.truth.values[sel], df_sparkx_adjpvals[method][sel]
        score = -np.log(df_sparkx_adjpvals[method][sel].values + min(np.min(df_sparkx_adjpvals[method][sel].values) * 1e-3, 1e-500))
        preds = method_pval < thred
        cm = metrics.confusion_matrix(truth_pval < thred, preds, labels=[0, 1])
        
        if type == "prec_rec":
            prec, rec, _  = metrics.precision_recall_curve(truth_pval < thred, score, pos_label=1)
            disp = metrics.PrecisionRecallDisplay(precision=prec, recall=rec)   
            disp.plot(ax=ax, name = f"{method} (AUC=({metrics.average_precision_score(truth_pval < thred, score):.2f})", color=color[idx])
        elif type == "roc":
            fpr, tpr, _  = metrics.roc_curve(truth_pval < thred, score, pos_label=1)
            disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
            disp.plot(ax=ax, name = f"{method} (AUC=({metrics.roc_auc_score(truth_pval < thred, score):.2f})", color=color[idx])
            
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    if type == "prec_rec":
        ax.set_title(f"Precison-Recall Curve on SparkX Test")
    elif type == "roc":
        ax.set_title(f"Receiver Operating Characteristic on SparkX Test")
    ax.legend(loc="best")
    plt.show()
    
    
def plot_curve_SDM(adatas, 
                   dataset_name, 
                   methods, 
                   color, 
                   type,
                   thred=0.01,
                   excluded_items={'TransImpCls', 'TransImpClsSpa', 'truth'}):
    _, ax = plt.subplots(figsize=(8, 7))
    idx = -1
    for md in methods:
        idx += 1
        if md in excluded_items: continue
        truth_res = adatas['truth'].uns['global_res'].loc[adatas['truth'].uns['global_I'] >= 0].copy()
        genes = np.intersect1d(truth_res.index, adatas[md].uns['global_res'].index)
        print(f"{md} # {len(genes)} pairs, # sig pairs: (Truth): {truth_res.loc[genes].selected.values.sum()}, (Preds) {adatas[md].uns['global_res'].loc[genes].selected.values.sum()}")
        
        if len(genes) < len(truth_res.index):
            print(f"{md} Fewer genes than truth: {len(genes)} vs {len(truth_res.index)}")
            
        y = truth_res.loc[genes].selected.values
        score =  -np.log(adatas[md].uns['global_res'].loc[genes].fdr.values + min(np.min(adatas[md].uns['global_res'].loc[genes].fdr.values)/1000, 1e-500))
        score[np.isinf(score)] = score[~np.isinf(score)].max()*10
        
        if type == "prec_rec":
            prec, rec, _  = metrics.precision_recall_curve(y, score, pos_label=1)
            disp = metrics.PrecisionRecallDisplay(precision=prec, recall=rec)   
            disp.plot(ax=ax, name = f"{md} (AUC=({metrics.average_precision_score(y, score):.2f})", color=color[idx])
        elif type == "roc":
            fpr, tpr, thresholds  = metrics.roc_curve(y, score, pos_label=1)
            disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
            disp.plot(ax=ax, name = f"{md} (AUC=({metrics.roc_auc_score(y, score):.2f})", color=color[idx])
            
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    if type == "prec_rec":
        ax.set_title(f"Precison-Recall Curve on Sig. Ligand-Receptor {dataset_name}")
    elif type == "roc":
        ax.set_title(f"Receiver Operating Characteristic on Sig. Ligand-Receptor {dataset_name}")
    ax.legend(loc="best")
    plt.show()