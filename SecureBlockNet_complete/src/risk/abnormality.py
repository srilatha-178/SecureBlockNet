import numpy as np

class BenignCentroidAbnormality:
    def __init__(self): self.centroid_=None; self.scale_=None
    def fit(self, X, y, benign_id=0):
        flat=np.asarray(X).reshape(len(X),-1)
        benign=flat[np.asarray(y)==benign_id]
        if len(benign)==0: raise ValueError('No benign samples in training data')
        self.centroid_=benign.mean(axis=0)
        d=np.linalg.norm(benign-self.centroid_,axis=1)
        self.scale_=max(float(np.quantile(d,.99)),1e-9)
        return self
    def score(self,X):
        flat=np.asarray(X).reshape(len(X),-1)
        d=np.linalg.norm(flat-self.centroid_,axis=1)
        return np.clip(d/self.scale_,0,1)
