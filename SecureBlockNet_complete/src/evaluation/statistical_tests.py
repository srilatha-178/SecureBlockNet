import numpy as np
from scipy import stats

def summarize_runs(values):
    x=np.asarray(values,dtype=float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1)) if n>1 else 0.0
    if n>1:
        h=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n))
    else: h=0.0
    return {'mean':mean,'std':sd,'ci95_low':mean-h,'ci95_high':mean+h}

def paired_tests(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    t=stats.ttest_rel(a,b)
    try: w=stats.wilcoxon(a,b)
    except ValueError: w=type('X',(),{'statistic':np.nan,'pvalue':np.nan})()
    return {'paired_t_stat':float(t.statistic),'paired_t_p':float(t.pvalue),'wilcoxon_stat':float(w.statistic),'wilcoxon_p':float(w.pvalue)}
