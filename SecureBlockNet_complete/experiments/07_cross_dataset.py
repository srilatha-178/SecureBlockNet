import argparse
from pathlib import Path
import pandas as pd
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018
from src.data.harmonization import common_numeric_features
from src.pipeline import prepare_data,train_model,predict_and_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-path',required=True)
    ap.add_argument('--ids2018-path',required=True)
    ap.add_argument('--ids2017-path',required=True)
    ap.add_argument('--max-train-rows',type=int)
    ap.add_argument('--max-test-rows',type=int)
    ap.add_argument('--epochs',type=int)
    a=ap.parse_args(); cfg=load_config(); set_global_seed(cfg['seed'])

    X0,y0,_,m0=load_ciciot2023(a.train_path,a.max_train_rows)
    X18,y18,_,m18=load_cicids2018(a.ids2018_path,a.max_test_rows)
    X17,y17,_,m17=load_cicids2017(a.ids2017_path,a.max_test_rows)
    common=common_numeric_features(X0,X18,X17)
    if len(common) < 5:
        raise RuntimeError(f'Only {len(common)} common numeric features found after canonicalization. '
                           'Inspect column aliases in src/data/harmonization.py before cross-dataset testing.')
    X0,X18,X17=X0[common],X18[common],X17[common]

    train=prepare_data(X0,y0,m0,cfg)
    model,_=train_model(train,cfg,'checkpoints/crossdataset_primary.keras',a.epochs)
    train.preprocessor.save('checkpoints/crossdataset_preprocessor.joblib')
    Path('results/tables').mkdir(parents=True,exist_ok=True)
    Path('results/tables/cross_dataset_common_features.txt').write_text('\n'.join(common),encoding='utf-8')

    rows=[]
    for name,Xe,ye,me in [('CSE-CIC-IDS2018',X18,y18,m18),('CIC-IDS2017',X17,y17,m17)]:
        ext=prepare_data(Xe,ye,me,cfg,preprocessor=train.preprocessor,fit=False)
        ext.label_encoder=train.label_encoder
        _,_,_,_,metrics=predict_and_score(model,ext,cfg)
        metrics.update({'train_dataset':'CICIoT2023','test_dataset':name,'common_features':len(common)})
        rows.append(metrics)
    out=pd.DataFrame(rows); out.to_csv('results/tables/cross_dataset_results.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
