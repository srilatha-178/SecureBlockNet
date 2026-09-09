from pathlib import Path
import csv

class CSVEventLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None

    def log(self, record: dict):
        record = {k: self._serialize(v) for k, v in record.items()}
        if self._fieldnames is None:
            self._fieldnames = list(record.keys())
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction='ignore')
            if not exists:
                w.writeheader()
            w.writerow(record)

    @staticmethod
    def _serialize(v):
        if isinstance(v, (list, tuple, dict)):
            return repr(v)
        return v
