import argparse,json
from pathlib import Path
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018
from src.pipeline import prepare_data,train_model,predict_and_score,risk_dataframe
from src.evaluation.classification_metrics import classwise_report
from src.evaluation.plots import save_confusion_matrix

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['ciciot2023','cicids2017','cicids2018'],required=True); ap.add_argument('--path',required=True); ap.add_argument('--max-rows',type=int); ap.add_argument('--epochs',type=int)
    a=ap.parse_args(); cfg=load_config(); set_global_seed(cfg['seed'])
    loader={'ciciot2023':load_ciciot2023,'cicids2017':load_cicids2017,'cicids2018':load_cicids2018}[a.dataset]
    X,y,_,meta=loader(a.path,a.max_rows); d=prepare_data(X,y,meta,cfg)
    model,_=train_model(d,cfg,f'checkpoints/{a.dataset}_secureblocknet.keras',a.epochs)
    d.preprocessor.save(f'checkpoints/{a.dataset}_preprocessor.joblib')
    probs,pred,pl,tl,metrics=predict_and_score(model,d,cfg)
    rdf,_,_=risk_dataframe(model,d,cfg); Path('results/predictions').mkdir(parents=True,exist_ok=True); rdf.to_csv(f'results/predictions/{a.dataset}_predictions.csv',index=False)
    Path('results/tables').mkdir(parents=True,exist_ok=True); Path(f'results/tables/{a.dataset}_metrics.json').write_text(json.dumps(metrics,indent=2))
    classwise_report(d.y_test,pred,d.label_encoder.classes_).to_csv(f'results/tables/{a.dataset}_classwise.csv')
    Path(f'results/tables/{a.dataset}_selected_features.txt').write_text('\n'.join(d.preprocessor.selected_features_),encoding='utf-8')
    save_confusion_matrix(d.y_test,pred,d.label_encoder.classes_,f'results/figures/{a.dataset}_confusion_matrix.png')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
