from collections import defaultdict, deque
import numpy as np

class HistoricalBehaviorTracker:
    def __init__(self, decay=0.8, window=20):
        self.decay=decay; self.window=window
        self.scores=defaultdict(float); self.recent=defaultdict(lambda: deque(maxlen=window))
    def update(self, entity, suspiciousness):
        x=float(np.clip(suspiciousness,0,1)); prev=self.scores[entity]
        new=self.decay*prev+(1-self.decay)*x
        self.scores[entity]=new; self.recent[entity].append(x)
        return new
    def persistence(self, entity):
        q=self.recent[entity]
        return float(np.mean(q)) if q else 0.0
    def current(self, entity): return self.scores[entity]
