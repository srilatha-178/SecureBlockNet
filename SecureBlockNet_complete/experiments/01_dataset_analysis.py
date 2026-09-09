import argparse,json
from pathlib import Path
import pandas as pd
import _bootstrap
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['ciciot2023','cicids2017','cicids2018'],required=True); ap.add_argument('--path',required=True); ap.add_argument('--max-rows',type=int); a=ap.parse_args()
    loader={'ciciot2023':load_ciciot2023,'cicids2017':load_cicids2017,'cicids2018':load_cicids2018}[a.dataset]
    X,y,yo,meta=loader(a.path,a.max_rows); out={'rows':len(X),'numeric_features':X.shape[1],'mapped_class_counts':y.value_counts().to_dict(),'original_unique_labels':int(yo.nunique()),'missing_fraction_after_loader':float(X.isna().mean().mean())}
    Path('results/tables').mkdir(parents=True,exist_ok=True); Path(f'results/tables/{a.dataset}_dataset_analysis.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
