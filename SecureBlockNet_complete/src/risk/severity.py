import numpy as np
DEFAULT_IMPACT = {
    'Benign':0.00,'Reconnaissance':0.35,'Web_BruteForce':0.55,'Spoofing':0.65,
    'DoS':0.80,'DDoS':0.90,'Other_Malicious':0.60
}

def protocol_impact(predicted_labels, impact_map=None):
    m=impact_map or DEFAULT_IMPACT
    return np.array([m.get(str(x),0.60) for x in predicted_labels],dtype=float)

def severity_score(abnormality, impact, persistence, weights=(0.40,0.35,0.25)):
    a,b,c=weights
    return np.clip(a*np.asarray(abnormality)+b*np.asarray(impact)+c*np.asarray(persistence),0,1)
