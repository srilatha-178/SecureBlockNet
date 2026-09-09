import argparse
from pathlib import Path
import pandas as pd
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018
from src.pipeline import prepare_data,train_model,predict_and_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['ciciot2023','cicids2017','cicids2018'],required=True); ap.add_argument('--path',required=True); ap.add_argument('--max-rows',type=int); ap.add_argument('--epochs',type=int); a=ap.parse_args()
    base=load_config(); loader={'ciciot2023':load_ciciot2023,'cicids2017':load_cicids2017,'cicids2018':load_cicids2018}[a.dataset]
    X,y,_,meta=loader(a.path,a.max_rows); rows=[]
    for seed in base['training']['seeds']:
        cfg=load_config(); cfg['seed']=int(seed); set_global_seed(int(seed)); d=prepare_data(X,y,meta,cfg); model,_=train_model(d,cfg,f'checkpoints/{a.dataset}_seed{seed}.keras',a.epochs); *_,metrics=predict_and_score(model,d,cfg); metrics.update({'model':'SecureBlockNet','seed':seed,'dataset':a.dataset}); rows.append(metrics)
    out=pd.DataFrame(rows); Path('results/statistics').mkdir(parents=True,exist_ok=True); out.to_csv('results/statistics/repeated_runs.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
