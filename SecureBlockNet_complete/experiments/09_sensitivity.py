import argparse
from pathlib import Path
import pandas as pd, numpy as np
import _bootstrap

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); a=ap.parse_args(); df=pd.read_csv(a.predictions)
    rows=[]; truth=(df.true_class!='Benign').to_numpy()
    for t1 in [0.25,0.30,0.35,0.40,0.45]:
      for t2 in [0.60,0.65,0.70,0.75,0.80]:
        if t1>=t2: continue
        r=df.risk_score.to_numpy(); pred=r>=t1; high=r>=t2
        fpr=float((pred & ~truth).sum()/max(1,(~truth).sum())); tpr=float((pred & truth).sum()/max(1,truth.sum()))
        rows.append({'low_medium_threshold':t1,'medium_high_threshold':t2,'security_detection_rate':tpr,'false_positive_rate':fpr,'high_risk_fraction':float(high.mean())})
    out=pd.DataFrame(rows); Path('results/tables').mkdir(parents=True,exist_ok=True); out.to_csv('results/tables/threshold_sensitivity.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
