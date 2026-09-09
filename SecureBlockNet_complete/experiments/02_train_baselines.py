import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018
from src.pipeline import prepare_data
from src.models.baselines import svm_baseline,rf_baseline,cnn_baseline,lstm_baseline,cnn_lstm_baseline,bilstm_baseline
from src.evaluation.classification_metrics import compute_classification_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['ciciot2023','cicids2017','cicids2018'],required=True); ap.add_argument('--path',required=True); ap.add_argument('--max-rows',type=int); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--classical-max-train',type=int,default=50000); a=ap.parse_args()
    cfg=load_config(); set_global_seed(cfg['seed']); loader={'ciciot2023':load_ciciot2023,'cicids2017':load_cicids2017,'cicids2018':load_cicids2018}[a.dataset]
    X,y,_,meta=loader(a.path,a.max_rows); d=prepare_data(X,y,meta,cfg); rows=[]
    flat_tr=d.X_train.reshape(len(d.X_train),-1); flat_te=d.X_test.reshape(len(d.X_test),-1)
    if len(flat_tr)>a.classical_max_train:
        rng=np.random.default_rng(cfg['seed']); idx=rng.choice(len(flat_tr),a.classical_max_train,replace=False); ctr,cy=flat_tr[idx],d.y_train[idx]
    else: ctr,cy=flat_tr,d.y_train
    for name,model in [('SVM',svm_baseline()),('RandomForest',rf_baseline())]:
        model.fit(ctr,cy); pred=model.predict(flat_te); probs=model.predict_proba(flat_te); r=compute_classification_metrics(d.y_test,pred,probs); r['model']=name; rows.append(r)
    for name,builder in [('CNN',cnn_baseline),('LSTM',lstm_baseline),('CNN-LSTM',cnn_lstm_baseline),('BiLSTM',bilstm_baseline)]:
        m=builder(d.X_train.shape[1:],len(d.label_encoder.classes_)); m.fit(d.X_train,d.y_train,validation_data=(d.X_val,d.y_val),epochs=a.epochs,batch_size=cfg['model']['batch_size'],verbose=0,callbacks=[]); probs=m.predict(d.X_test,verbose=0); pred=probs.argmax(1); r=compute_classification_metrics(d.y_test,pred,probs); r['model']=name; rows.append(r)
    out=pd.DataFrame(rows); Path('results/tables').mkdir(parents=True,exist_ok=True); out.to_csv(f'results/tables/{a.dataset}_baselines.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
