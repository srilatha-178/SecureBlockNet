import argparse,json,time,os
from pathlib import Path
import _bootstrap
from src.crypto.crypto_engine import CryptoEngine
from src.evaluation.crypto_metrics import summarize_crypto

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--iterations',type=int,default=200); ap.add_argument('--payload-bytes',type=int,default=1500); a=ap.parse_args()
    ce=CryptoEngine(); payload=os.urandom(a.payload_bytes); out={}
    for policy in ['LOW_AES128_GCM','MEDIUM_AES256_GCM','HIGH_AES256_GCM_ECDHE_ECDSA']:
        rs=[ce.protect(payload,policy,f's{i}',str(time.time_ns()),.8)[0] for i in range(a.iterations)]
        out[policy]=summarize_crypto(rs)
    Path('results/tables').mkdir(parents=True,exist_ok=True); Path('results/tables/crypto_benchmark.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
