import numpy as np

def build_sequences(X, y=None, meta=None, sequence_length=10, stride=1):
    X = np.asarray(X, dtype=np.float32)
    y_arr = None if y is None else np.asarray(y)
    seqs, labels, end_indices = [], [], []
    for end in range(sequence_length - 1, len(X), stride):
        start = end - sequence_length + 1
        seqs.append(X[start:end+1])
        if y_arr is not None:
            labels.append(y_arr[end])
        end_indices.append(end)
    if not seqs:
        raise ValueError(f'Not enough rows ({len(X)}) for sequence_length={sequence_length}')
    out_meta = None
    if meta is not None:
        out_meta = meta.iloc[end_indices].reset_index(drop=True)
    return np.asarray(seqs, dtype=np.float32), (None if y is None else np.asarray(labels)), out_meta
