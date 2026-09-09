import numpy as np

def compute_blocking_metrics(true_labels, actions, benign_label='Benign'):
    y=np.asarray(true_labels); a=np.asarray(actions)
    mal=y!=benign_label; benign=~mal
    contained=np.isin(a,['BLOCK','ENCRYPTED_QUARANTINE'])
    monitored=a=='MONITOR'
    correct=((benign & np.isin(a,['ALLOW','MONITOR'])) | (mal & np.isin(a,['MONITOR','BLOCK','ENCRYPTED_QUARANTINE'])))
    return {
        'attack_block_or_quarantine_rate': float((contained & mal).sum()/max(1,mal.sum())),
        'false_block_or_quarantine_rate': float((contained & benign).sum()/max(1,benign.sum())),
        'monitoring_rate': float(monitored.mean()),
        'quarantine_rate': float((a=='ENCRYPTED_QUARANTINE').mean()),
        'security_reliability': float(correct.mean())
    }
