from sklearn.model_selection import train_test_split

def stratified_three_way_split(X, y, meta, test_size=0.15, validation_size=0.15, random_state=42):
    idx = list(range(len(X)))
    trainval, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
    val_frac = validation_size / (1.0 - test_size)
    train, val = train_test_split(trainval, test_size=val_frac, stratify=y.iloc[trainval], random_state=random_state)
    def take(indices):
        xx=X.iloc[indices].copy(); yy=y.iloc[indices].copy(); mm=meta.iloc[indices].copy()
        # Preserve temporal semantics inside each non-overlapping partition when a timestamp is available.
        if 'timestamp' in mm.columns:
            order=mm['timestamp'].sort_values(kind='stable').index
            xx=xx.loc[order]; yy=yy.loc[order]; mm=mm.loc[order]
        return xx.reset_index(drop=True), yy.reset_index(drop=True), mm.reset_index(drop=True)
    return take(train), take(val), take(test)
