import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

def compute_classification_metrics(y_true,y_pred,probs=None,num_classes=None):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    out={'accuracy':accuracy_score(y_true,y_pred),'macro_precision':p,'macro_recall':r,'macro_f1':f,'weighted_f1':wf}
    if probs is not None:
        try:
            labels=np.arange(probs.shape[1]); y_oh=np.eye(probs.shape[1])[np.asarray(y_true,dtype=int)]
            out['macro_roc_auc_ovr']=roc_auc_score(y_oh,probs,average='macro',multi_class='ovr',labels=labels)
        except Exception:
            out['macro_roc_auc_ovr']=float('nan')
    return out

def classwise_report(y_true,y_pred,class_names):
    return pd.DataFrame(classification_report(y_true,y_pred,target_names=class_names,output_dict=True,zero_division=0)).T

def confusion(y_true,y_pred): return confusion_matrix(y_true,y_pred)
