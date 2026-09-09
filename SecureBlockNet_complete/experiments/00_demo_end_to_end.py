import argparse, json
from pathlib import Path
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import make_synthetic_flow_dataset
from src.pipeline import prepare_data,train_model,predict_and_score,risk_dataframe
from src.enforcement.traffic_controller import TrafficController
from src.evaluation.blocking_metrics import compute_blocking_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples',type=int,default=3000); ap.add_argument('--epochs',type=int,default=2); args=ap.parse_args()
    cfg=load_config(); set_global_seed(cfg['seed'])
    X,y,_,meta=make_synthetic_flow_dataset(args.samples,seed=cfg['seed'])
    d=prepare_data(X,y,meta,cfg); model,_=train_model(d,cfg,'checkpoints/demo_secureblocknet.keras',args.epochs)
    _,_,_,_,metrics=predict_and_score(model,d,cfg)
    rdf,re,_=risk_dataframe(model,d,cfg)
    controller=TrafficController(re,cfg['crypto']['quarantine_dir'])
    outcomes=[]
    for i,row in rdf.head(min(100,len(rdf))).iterrows():
        outcomes.append(controller.enforce({'row':int(i),'predicted_class':row.predicted_class},row.risk_score,row.confidence))
    b=compute_blocking_metrics(rdf.true_class,rdf.action)
    Path('results/predictions').mkdir(parents=True,exist_ok=True); rdf.to_csv('results/predictions/demo_predictions.csv',index=False)
    print(json.dumps({'classification':metrics,'blocking':b,'crypto_smoke_samples':len(outcomes)},indent=2))
if __name__=='__main__': main()
