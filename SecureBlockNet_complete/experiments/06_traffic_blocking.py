import argparse,json
from pathlib import Path
import pandas as pd
import _bootstrap
from src.utils.config import load_config
from src.risk.risk_engine import RiskEngine
from src.enforcement.traffic_controller import TrafficController
from src.evaluation.blocking_metrics import compute_blocking_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--enforce-limit',type=int,default=1000); a=ap.parse_args()
    cfg=load_config(); df=pd.read_csv(a.predictions)
    re=RiskEngine(cfg['risk']['confidence_weight'],cfg['risk']['severity_weight'],cfg['risk']['history_weight'],cfg['risk']['low_medium_threshold'],cfg['risk']['medium_high_threshold'],cfg['risk']['block_confidence_threshold'])
    controller=TrafficController(re,cfg['crypto']['quarantine_dir'])
    details=[]
    for i,row in df.head(a.enforce_limit).iterrows():
        o=controller.enforce(row.to_dict(),float(row.risk_score),float(row.confidence)); details.append(o.__dict__)
    metrics=compute_blocking_metrics(df.true_class,df.action)
    Path('results/tables').mkdir(parents=True,exist_ok=True); pd.DataFrame(details).to_csv('results/tables/enforcement_timing.csv',index=False); Path('results/tables/blocking_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
