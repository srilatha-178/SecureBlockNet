import numpy as np

def summarize_crypto(records):
    keys=['encryption_seconds','decryption_seconds','key_derivation_seconds','handshake_seconds','signature_seconds','verify_seconds']
    return {k:{'mean':float(np.mean([getattr(r,k) for r in records])),'std':float(np.std([getattr(r,k) for r in records]))} for k in keys}
