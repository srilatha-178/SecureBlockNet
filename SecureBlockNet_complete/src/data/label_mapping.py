import re
import pandas as pd

CLASSES = ['Benign','DoS','DDoS','Reconnaissance','Spoofing','Web_BruteForce','Other_Malicious']

_PATTERNS = [
    ('Benign', [r'^benign$', r'^normal$', r'benigntraffic']),
    ('DDoS', [r'ddos', r'distributed.*denial']),
    ('DoS', [r'(^|[^d])dos', r'denial.?of.?service', r'slowloris', r'slowhttptest', r'hulk', r'goldeneye']),
    ('Reconnaissance', [r'recon', r'port.?scan', r'portscan', r'network.?scan', r'vulnerability.?scan', r'ping.?sweep']),
    ('Spoofing', [r'spoof', r'arp', r'dns.?spoof', r'mitm']),
    ('Web_BruteForce', [r'brute', r'web', r'xss', r'sql', r'injection', r'ftp.?patator', r'ssh.?patator']),
]

def normalize_label(value):
    s = str(value).strip().lower().replace('_', ' ').replace('-', ' ')
    for cls, patterns in _PATTERNS:
        if any(re.search(p, s) for p in patterns):
            return cls
    return 'Other_Malicious'

def map_labels(series: pd.Series):
    return series.map(normalize_label)
