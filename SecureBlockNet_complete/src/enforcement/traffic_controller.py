from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json, time, uuid
from ..crypto.crypto_engine import CryptoEngine
from ..crypto.policy_manager import policy_for_level

@dataclass
class EnforcementOutcome:
    action: str
    risk_level: str
    crypto_policy: str
    decision_seconds: float
    crypto_seconds: float
    session_id: str
    quarantine_path: str = ''

class TrafficController:
    def __init__(self, risk_engine, quarantine_dir='quarantine'):
        self.risk_engine=risk_engine
        self.crypto=CryptoEngine()
        self.quarantine_dir=Path(quarantine_dir)

    def enforce(self, record: dict, risk: float, confidence: float):
        t0=time.perf_counter()
        level=self.risk_engine.risk_level(risk)
        action=self.risk_engine.action(risk,confidence)
        policy=policy_for_level(level)
        session_id=str(uuid.uuid4())
        timestamp=datetime.now(timezone.utc).isoformat()
        payload=json.dumps(record,default=str,sort_keys=True).encode()
        quarantine_path=''
        if action=='BLOCK':
            crypto_s=0.0
        elif action=='ENCRYPTED_QUARANTINE':
            quarantine_path=str(self.quarantine_dir/f'{session_id}.json')
            cr=self.crypto.quarantine(record,quarantine_path,session_id,timestamp,risk)
            crypto_s=cr.encryption_seconds+cr.key_derivation_seconds+cr.handshake_seconds+cr.signature_seconds+cr.verify_seconds
        else:
            cr,_=self.crypto.protect(payload,policy,session_id,timestamp,risk)
            crypto_s=cr.encryption_seconds+cr.key_derivation_seconds+cr.handshake_seconds+cr.signature_seconds+cr.verify_seconds
        return EnforcementOutcome(action,level,policy,time.perf_counter()-t0,crypto_s,session_id,quarantine_path)
