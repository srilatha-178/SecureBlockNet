from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .data.splitting import stratified_three_way_split
from .data.preprocessing import LeakageFreePreprocessor
from .data.sequence_builder import build_sequences
from .models.secureblocknet import build_secureblocknet, training_callbacks
from .risk.abnormality import BenignCentroidAbnormality
from .risk.severity import protocol_impact, severity_score
from .risk.history import HistoricalBehaviorTracker
from .risk.risk_engine import RiskEngine
from .evaluation.classification_metrics import compute_classification_metrics, classwise_report

class PreparedData:
    pass

def prepare_data(X,y,meta,config,preprocessor=None,fit=True):
    p=config['preprocessing']; seed=config['seed']
    if fit:
        (Xtr,ytr,mtr),(Xv,yv,mv),(Xte,yte,mte)=stratified_three_way_split(
            X,y,meta,p['test_size'],p['validation_size'],seed)
        pre=LeakageFreePreprocessor(p['correlation_threshold'],p['top_k_features'],seed)
        Xtr_s=pre.fit_transform(Xtr,ytr); Xv_s=pre.transform(Xv); Xte_s=pre.transform(Xte)
    else:
        if preprocessor is None: raise ValueError('preprocessor required for frozen transform')
        pre=preprocessor; Xtr_s=None; Xv_s=None; ytr=yv=mtr=mv=None
        Xte_s=pre.transform(X); yte=y.reset_index(drop=True); mte=meta.reset_index(drop=True)
    le=LabelEncoder(); le.fit(config['classes'])
    out=PreparedData(); out.preprocessor=pre; out.label_encoder=le
    L=p['sequence_length']; S=p['sequence_stride']
    if fit:
        out.X_train,out.y_train,out.meta_train=build_sequences(Xtr_s,le.transform(ytr),mtr,L,S)
        out.X_val,out.y_val,out.meta_val=build_sequences(Xv_s,le.transform(yv),mv,L,S)
    out.X_test,out.y_test,out.meta_test=build_sequences(Xte_s,le.transform(yte),mte,L,S)
    return out

def train_model(prepared,config,checkpoint_path='checkpoints/secureblocknet.keras',epochs_override=None):
    mcfg=config['model']
    model=build_secureblocknet(
        prepared.X_train.shape[1:], len(prepared.label_encoder.classes_),
        tuple(mcfg['cnn_filters']), mcfg['kernel_size'], mcfg['bilstm_units'],
        tuple(mcfg['dense_units']), mcfg['dropout'], mcfg['learning_rate'])
    classes=np.unique(prepared.y_train)
    weights=compute_class_weight('balanced',classes=classes,y=prepared.y_train)
    cw={int(c):float(w) for c,w in zip(classes,weights)}
    Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
    hist=model.fit(prepared.X_train,prepared.y_train,validation_data=(prepared.X_val,prepared.y_val),
                   epochs=int(epochs_override or mcfg['epochs']),batch_size=mcfg['batch_size'],class_weight=cw,
                   callbacks=training_callbacks(checkpoint_path,mcfg['early_stopping_patience']),verbose=config.get('training',{}).get('verbose',1))
    return model,hist

def predict_and_score(model,prepared,config):
    probs=model.predict(prepared.X_test,batch_size=config['model']['batch_size'],verbose=0)
    pred=probs.argmax(axis=1)
    metrics=compute_classification_metrics(prepared.y_test,pred,probs)
    labels=prepared.label_encoder.inverse_transform(pred)
    true_labels=prepared.label_encoder.inverse_transform(prepared.y_test)
    return probs,pred,labels,true_labels,metrics

def risk_dataframe(model,prepared,config,abnormality_model=None):
    probs=model.predict(prepared.X_test,batch_size=config['model']['batch_size'],verbose=0)
    pred=probs.argmax(axis=1)
    labels=prepared.label_encoder.inverse_transform(pred)
    true_labels=prepared.label_encoder.inverse_transform(prepared.y_test)
    benign_id=int(prepared.label_encoder.transform(['Benign'])[0])
    if abnormality_model is None:
        abnormality_model=BenignCentroidAbnormality().fit(prepared.X_train,prepared.y_train,benign_id)
    abnormality=abnormality_model.score(prepared.X_test)
    impact=protocol_impact(labels)
    tracker=HistoricalBehaviorTracker(config['risk']['history_decay'],config['risk']['history_window'])
    persistence=[]; history=[]
    entities = prepared.meta_test['src_ip'].astype(str).tolist() if 'src_ip' in prepared.meta_test else [f'global_{i%20}' for i in range(len(pred))]
    # causal history: read previous state, compute persistence, then update with current maliciousness proxy
    maxp=probs.max(axis=1)
    mc=RiskEngine.malicious_confidence(probs,pred,benign_id)
    for ent,susp in zip(entities,mc):
        persistence.append(tracker.persistence(ent)); history.append(tracker.current(ent)); tracker.update(ent,susp)
    sev=severity_score(abnormality,impact,persistence,(config['risk']['severity_abnormality_weight'],config['risk']['severity_impact_weight'],config['risk']['severity_persistence_weight']))
    re=RiskEngine(config['risk']['confidence_weight'],config['risk']['severity_weight'],config['risk']['history_weight'],config['risk']['low_medium_threshold'],config['risk']['medium_high_threshold'],config['risk']['block_confidence_threshold'])
    risk=re.score(mc,sev,history)
    actions=[re.action(r,c) for r,c in zip(risk,maxp)]
    levels=[re.risk_level(r) for r in risk]
    df=prepared.meta_test.copy()
    df['true_class']=true_labels; df['predicted_class']=labels; df['confidence']=maxp; df['malicious_confidence']=mc
    df['abnormality']=abnormality; df['protocol_impact']=impact; df['persistence']=persistence; df['historical_score']=history; df['severity']=sev; df['risk_score']=risk; df['risk_level']=levels; df['action']=actions
    for j,c in enumerate(prepared.label_encoder.classes_): df[f'prob_{c}']=probs[:,j]
    return df,re,abnormality_model
