import argparse,json
from pathlib import Path
import pandas as pd
import _bootstrap
from src.evaluation.statistical_tests import summarize_runs,paired_tests

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--metric',default='macro_f1'); ap.add_argument('--proposed',default='SecureBlockNet'); ap.add_argument('--baseline',default='CNN-LSTM'); a=ap.parse_args()
    df=pd.read_csv(a.csv); out={}
    for model,g in df.groupby('model'): out[model]=summarize_runs(g[a.metric].to_numpy())
    A=df[df.model==a.proposed].sort_values('seed')[a.metric].to_numpy(); B=df[df.model==a.baseline].sort_values('seed')[a.metric].to_numpy()
    if len(A)==len(B) and len(A)>1: out['paired_comparison']=paired_tests(A,B)
    Path('results/statistics').mkdir(parents=True,exist_ok=True); Path('results/statistics/summary.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
