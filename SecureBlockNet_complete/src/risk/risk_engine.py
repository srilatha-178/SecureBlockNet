import numpy as np

class RiskEngine:
    def __init__(self, confidence_weight=.45, severity_weight=.35, history_weight=.20,
                 low_medium_threshold=.35, medium_high_threshold=.70, block_confidence_threshold=.85):
        self.cw=confidence_weight; self.sw=severity_weight; self.hw=history_weight
        self.t1=low_medium_threshold; self.t2=medium_high_threshold; self.block_threshold=block_confidence_threshold
    @staticmethod
    def malicious_confidence(probabilities, predicted_ids, benign_id=0):
        p=np.asarray(probabilities); pred=np.asarray(predicted_ids)
        maxp=p.max(axis=1); benignp=p[:,benign_id]
        return np.where(pred==benign_id,1.0-benignp,maxp)
    def score(self, malicious_confidence, severity, history):
        return np.clip(self.cw*np.asarray(malicious_confidence)+self.sw*np.asarray(severity)+self.hw*np.asarray(history),0,1)
    def risk_level(self,r):
        if r < self.t1: return 'LOW'
        if r < self.t2: return 'MEDIUM'
        return 'HIGH'
    def action(self,r,confidence):
        level=self.risk_level(float(r))
        if level=='LOW': return 'ALLOW'
        if level=='MEDIUM': return 'MONITOR'
        return 'BLOCK' if float(confidence)>=self.block_threshold else 'ENCRYPTED_QUARANTINE'
