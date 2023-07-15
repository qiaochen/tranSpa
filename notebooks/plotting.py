from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np


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