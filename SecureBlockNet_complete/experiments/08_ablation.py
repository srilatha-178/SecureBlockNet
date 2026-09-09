import argparse
from pathlib import Path
import pandas as pd
from tensorflow import keras
import _bootstrap
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.loaders import load_ciciot2023,load_cicids2017,load_cicids2018
from src.pipeline import prepare_data
from src.models.attention import TemporalAttention
from src.evaluation.classification_metrics import compute_classification_metrics

def build_variant(shape,ncls,variant):
    inp=keras.Input(shape=shape); x=inp
    x=keras.layers.Conv1D(64,3,padding='same',activation='relu')(x); x=keras.layers.MaxPooling1D(2,padding='same')(x)
    if variant in ['CNN_BiLSTM','CNN_BiLSTM_Attention','FULL']:
        x=keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=variant in ['CNN_BiLSTM_Attention','FULL']))(x)
    if variant in ['CNN_BiLSTM_Attention','FULL']:
        x=TemporalAttention(64)(x)
    elif variant=='CNN_ONLY':
        x=keras.layers.GlobalAveragePooling1D()(x)
    x=keras.layers.Dense(64,activation='relu')(x); x=keras.layers.Dropout(.5)(x); out=keras.layers.Dense(ncls,activation='softmax')(x)
    m=keras.Model(inp,out); m.compile(keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy',['accuracy']); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['ciciot2023','cicids2017','cicids2018'],required=True); ap.add_argument('--path',required=True); ap.add_argument('--max-rows',type=int); ap.add_argument('--epochs',type=int,default=30); a=ap.parse_args()
    cfg=load_config(); set_global_seed(cfg['seed']); loader={'ciciot2023':load_ciciot2023,'cicids2017':load_cicids2017,'cicids2018':load_cicids2018}[a.dataset]
    X,y,_,meta=loader(a.path,a.max_rows); d=prepare_data(X,y,meta,cfg); rows=[]
    for v in ['CNN_ONLY','CNN_BiLSTM','CNN_BiLSTM_Attention','FULL']:
        m=build_variant(d.X_train.shape[1:],len(d.label_encoder.classes_),v); m.fit(d.X_train,d.y_train,validation_data=(d.X_val,d.y_val),epochs=a.epochs,batch_size=cfg['model']['batch_size'],verbose=0); probs=m.predict(d.X_test,verbose=0); pred=probs.argmax(1); r=compute_classification_metrics(d.y_test,pred,probs); r['variant']=v; rows.append(r)
    out=pd.DataFrame(rows); Path('results/tables').mkdir(parents=True,exist_ok=True); out.to_csv(f'results/tables/{a.dataset}_ablation.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
