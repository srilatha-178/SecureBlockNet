import json, os, time
from dataclasses import dataclass
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

@dataclass
class CryptoResult:
    policy: str
    nonce: bytes
    ciphertext: bytes
    encryption_seconds: float
    decryption_seconds: float
    key_derivation_seconds: float
    handshake_seconds: float = 0.0
    signature_seconds: float = 0.0
    verify_seconds: float = 0.0

class CryptoEngine:
    def __init__(self):
        self.enforcement_signing_key = ec.generate_private_key(ec.SECP256R1())
        self.quarantine_signing_key = ec.generate_private_key(ec.SECP256R1())

    @staticmethod
    def _derive_key(secret: bytes, length: int, session_id: str, timestamp: str, risk: float, policy: str):
        info=f'{session_id}|{timestamp}|{risk:.6f}|{policy}'.encode()
        t0=time.perf_counter()
        key=HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(secret)
        return key,time.perf_counter()-t0

    @staticmethod
    def _aad(session_id, policy, routing='SecureBlockNet'):
        return f'{session_id}|{policy}|{routing}'.encode()

    def low_or_medium_secret(self):
        return os.urandom(32)

    def authenticated_ecdh_secret(self, session_id: str):
        t0=time.perf_counter()
        a=ec.generate_private_key(ec.SECP256R1())
        b=ec.generate_private_key(ec.SECP256R1())
        apub=a.public_key(); bpub=b.public_key()
        message=(session_id.encode()+
                 apub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)+
                 bpub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))
        ts=time.perf_counter(); sig=self.enforcement_signing_key.sign(message,ec.ECDSA(hashes.SHA256())); sig_t=time.perf_counter()-ts
        tv=time.perf_counter();
        try:
            self.enforcement_signing_key.public_key().verify(sig,message,ec.ECDSA(hashes.SHA256()))
        except InvalidSignature as e:
            raise RuntimeError('Endpoint signature verification failed') from e
        ver_t=time.perf_counter()-tv
        s1=a.exchange(ec.ECDH(),bpub); s2=b.exchange(ec.ECDH(),apub)
        if s1 != s2: raise RuntimeError('ECDH shared secret mismatch')
        return s1,time.perf_counter()-t0,sig_t,ver_t

    def protect(self, payload: bytes, policy: str, session_id: str, timestamp: str, risk: float):
        if policy == 'LOW_AES128_GCM':
            secret=self.low_or_medium_secret(); hs=sig=ver=0.0; length=16
        elif policy == 'MEDIUM_AES256_GCM':
            secret=self.low_or_medium_secret(); hs=sig=ver=0.0; length=32
        elif policy == 'HIGH_AES256_GCM_ECDHE_ECDSA':
            secret,hs,sig,ver=self.authenticated_ecdh_secret(session_id); length=32
        else:
            raise ValueError(f'Unknown crypto policy: {policy}')
        key,kdf_t=self._derive_key(secret,length,session_id,timestamp,risk,policy)
        nonce=os.urandom(12); aad=self._aad(session_id,policy)
        aes=AESGCM(key)
        t0=time.perf_counter(); ct=aes.encrypt(nonce,payload,aad); enc=time.perf_counter()-t0
        t1=time.perf_counter(); recovered=aes.decrypt(nonce,ct,aad); dec=time.perf_counter()-t1
        if recovered != payload: raise RuntimeError('Authenticated decryption mismatch')
        return CryptoResult(policy,nonce,ct,enc,dec,kdf_t,hs,sig,ver), key

    def quarantine(self, record: dict, path, session_id: str, timestamp: str, risk: float):
        payload=json.dumps(record,default=str,sort_keys=True).encode()
        result,_=self.protect(payload,'HIGH_AES256_GCM_ECDHE_ECDSA',session_id,timestamp,risk)
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        package={'session_id':session_id,'timestamp':timestamp,'risk':risk,'policy':result.policy,
                 'nonce_hex':result.nonce.hex(),'ciphertext_hex':result.ciphertext.hex()}
        p.write_text(json.dumps(package,indent=2),encoding='utf-8')
        return result
