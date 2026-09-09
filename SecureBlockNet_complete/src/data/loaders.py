from pathlib import Path
import numpy as np
import pandas as pd
from .harmonization import canonicalize_columns
from .label_mapping import map_labels

LABEL_CANDIDATES = ['label','class','attack','attack_type','category','type']
TIME_CANDIDATES = ['timestamp','flow_start_time','time','ts']
GROUP_CANDIDATES = ['src_ip','source_ip','srcip','flow_id','session_id']

def _discover_csvs(path):
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(list(p.rglob('*.csv')) + list(p.rglob('*.csv.gz')))
    if not files:
        raise FileNotFoundError(f'No CSV/CSV.GZ files found under {p}')
    return files

def _read_many(files, max_rows=None):
    frames, remain = [], max_rows
    for fp in files:
        nrows = None if remain is None else max(0, remain)
        if nrows == 0: break
        df = pd.read_csv(fp, low_memory=False, nrows=nrows)
        frames.append(df)
        if remain is not None:
            remain -= len(df)
    return pd.concat(frames, ignore_index=True)

def load_flow_dataset(path, dataset_name, max_rows=None):
    files = _discover_csvs(path)
    df = canonicalize_columns(_read_many(files, max_rows=max_rows))
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        # permissive suffix search but never guess values
        label_col = next((c for c in df.columns if 'label' in c or c.endswith('class')), None)
    if label_col is None:
        raise ValueError(f'No label column found. Available columns: {list(df.columns)[:80]}')
    y_original = df[label_col].astype(str).copy()
    y = map_labels(y_original)
    df = df.drop(columns=[label_col])
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    # align labels after duplicate removal by retained index
    y_original = y_original.loc[df.index]
    y = y.loc[df.index]
    # Keep time/group metadata where available, but not as model features.
    meta = pd.DataFrame(index=df.index)
    for c in TIME_CANDIDATES + GROUP_CANDIDATES:
        if c in df.columns:
            meta[c] = df[c]
    # Convert numeric-looking columns, then retain numeric model inputs.
    for c in df.columns:
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors='coerce')
            if converted.notna().mean() > 0.90:
                df[c] = converted
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.dropna(axis=1, how='all')
    valid = numeric.notna().any(axis=1)
    numeric = numeric.loc[valid]
    y = y.loc[numeric.index]
    y_original = y_original.loc[numeric.index]
    meta = meta.loc[numeric.index]
    numeric = numeric.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_original = y_original.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    meta['dataset'] = dataset_name
    return numeric, y, y_original, meta

def load_ciciot2023(path, max_rows=None): return load_flow_dataset(path, 'CICIoT2023', max_rows)
def load_cicids2017(path, max_rows=None): return load_flow_dataset(path, 'CIC-IDS2017', max_rows)
def load_cicids2018(path, max_rows=None): return load_flow_dataset(path, 'CSE-CIC-IDS2018', max_rows)

def make_synthetic_flow_dataset(n=3000, features=24, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.array(['Benign','DoS','DDoS','Reconnaissance','Spoofing','Web_BruteForce','Other_Malicious'])
    probs = np.array([.45,.10,.15,.08,.07,.08,.07])
    yi = rng.choice(len(classes), size=n, p=probs)
    X = rng.normal(0, 1, size=(n, features))
    # class-dependent structured shifts to make the smoke test learnable
    for k in range(len(classes)):
        idx = yi == k
        X[idx, k:min(k+5, features)] += k * 0.55
        X[idx, 10:14] += (k % 3) * 0.35
    cols = [f'feature_{i:02d}' for i in range(features)]
    meta = pd.DataFrame({'timestamp': np.arange(n), 'src_ip': [f'host_{i%50}' for i in range(n)], 'dataset':'Synthetic'})
    return pd.DataFrame(X, columns=cols), pd.Series(classes[yi]), pd.Series(classes[yi]), meta
