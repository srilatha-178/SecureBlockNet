import itertools
import numpy as np
from sklearn.metrics import f1_score

WEIGHT_CONFIGS = {
    'equal':(.33,.33,.34), 'confidence_dominant':(.60,.25,.15),
    'severity_dominant':(.25,.60,.15), 'proposed':(.45,.35,.20)
}

def grid_weights(step=.05):
    vals=np.arange(0,1+1e-9,step)
    for a in vals:
        for b in vals:
            c=1-a-b
            if c >= -1e-9 and c <= 1+1e-9:
                yield round(float(a),2),round(float(b),2),round(float(max(0,c)),2)

def evaluate_weight_config(mc,severity,history,true_malicious,t1=.35,t2=.70,weights=(.45,.35,.20)):
    w=np.asarray(weights); r=np.clip(w[0]*mc+w[1]*severity+w[2]*history,0,1)
    pred=(r>=t1).astype(int)
    f1=f1_score(true_malicious,pred,zero_division=0)
    benign=np.asarray(true_malicious)==0
    fpr=float(((pred==1)&benign).sum()/max(1,benign.sum()))
    return {'weights':weights,'f1':f1,'false_positive_rate':fpr,'mean_risk':float(r.mean())}
