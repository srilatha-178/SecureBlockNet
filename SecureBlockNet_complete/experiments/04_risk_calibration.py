import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import _bootstrap
from src.risk.calibration import WEIGHT_CONFIGS,evaluate_weight_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); a=ap.parse_args(); df=pd.read_csv(a.predictions)
    truth=(df.true_class!='Benign').astype(int).to_numpy(); rows=[]
    for name,w in WEIGHT_CONFIGS.items():
        r=evaluate_weight_config(df.malicious_confidence.to_numpy(),df.severity.to_numpy(),df.historical_score.to_numpy(),truth,weights=w); r['configuration']=name; rows.append(r)
    out=pd.DataFrame(rows); Path('results/tables').mkdir(parents=True,exist_ok=True); out.to_csv('results/tables/risk_weight_sensitivity.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
