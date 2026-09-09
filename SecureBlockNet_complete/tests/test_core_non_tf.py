from src.data.loaders import make_synthetic_flow_dataset
from src.data.preprocessing import LeakageFreePreprocessor
from src.data.sequence_builder import build_sequences
from src.risk.risk_engine import RiskEngine
from src.crypto.crypto_engine import CryptoEngine

def test_preprocess_and_sequence():
    X,y,_,m=make_synthetic_flow_dataset(300,24,42)
    p=LeakageFreePreprocessor(top_k_features=12).fit(X,y)
    z=p.transform(X)
    s,lab,meta=build_sequences(z,y,m,10,1)
    assert s.shape[1:]==(10,12)

def test_risk_actions():
    r=RiskEngine()
    assert r.action(.1,.9)=='ALLOW'
    assert r.action(.5,.9)=='MONITOR'
    assert r.action(.8,.9)=='BLOCK'
    assert r.action(.8,.7)=='ENCRYPTED_QUARANTINE'

def test_crypto_profiles():
    ce=CryptoEngine()
    for p in ['LOW_AES128_GCM','MEDIUM_AES256_GCM','HIGH_AES256_GCM_ECDHE_ECDSA']:
        result,_=ce.protect(b'abc',p,'sid','ts',.8)
        assert result.ciphertext
