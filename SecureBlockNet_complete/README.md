# SecureBlockNet — Complete Research Implementation

This repository implements the manuscript architecture:

**flow data → leakage-free preprocessing → 1D-CNN → BiLSTM → temporal attention → class/confidence → severity/history-aware risk → AES-GCM/ECDHE/ECDSA policy → allow/monitor/block/encrypted quarantine → logging/feedback.**

## 1. Environment

Recommended: Python 3.11, TensorFlow 2.16.1, Keras 3.3.3.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Quick functional smoke run

A synthetic flow dataset is included through an on-demand generator, so the complete pipeline can be tested without downloading benchmark datasets:

```bash
python experiments/00_demo_end_to_end.py --epochs 2 --samples 3000
```

This is only a software smoke test, not a manuscript result.

## 3. Real datasets

Extract CSV/CSV.GZ files into any paths, e.g.:

```text
data/raw/ciciot2023/
data/raw/cicids2018/
data/raw/cicids2017/
```

Then train a dataset:

```bash
python experiments/03_train_secureblocknet.py \
  --dataset ciciot2023 --path data/raw/ciciot2023
```

The loader recursively discovers CSV/CSV.GZ files. It normalizes column names, removes invalid/duplicate rows, maps labels to the seven-class taxonomy, creates leakage-free 70/15/15 partitions, fits correlation filtering + mutual information + MinMax scaling on training data only, then forms temporal flow windows.

## 4. Cross-dataset evaluation

The strict manuscript protocol is implemented: train on CICIoT2023 and apply the frozen preprocessing/model to harmonized CSE-CIC-IDS2018 and CIC-IDS2017 representations without refitting.

```bash
python experiments/07_cross_dataset.py \
  --train-path data/raw/ciciot2023 \
  --ids2018-path data/raw/cicids2018 \
  --ids2017-path data/raw/cicids2017
```

Cross-dataset evaluation requires semantically common columns. Column aliases are normalized in `src/data/harmonization.py`. Unavailable selected features are rejected rather than silently imputed.

## 5. Main experiments

```bash
python experiments/02_train_baselines.py --dataset ciciot2023 --path data/raw/ciciot2023
python experiments/04_risk_calibration.py --predictions results/predictions/ciciot2023_predictions.csv
python experiments/05_crypto_benchmark.py
python experiments/06_traffic_blocking.py --predictions results/predictions/ciciot2023_predictions.csv
python experiments/08_ablation.py --dataset ciciot2023 --path data/raw/ciciot2023
python experiments/09_sensitivity.py --predictions results/predictions/ciciot2023_predictions.csv
python experiments/10_statistical_analysis.py --csv results/statistics/repeated_runs.csv
```

## 6. Important scientific implementation choices

- Temporal windows are genuine ordered flow windows; a tabular row is not merely reshaped and mislabeled as temporal data.
- The operational risk engine uses **malicious confidence**: for a benign prediction it uses `1 - P(benign)`, preventing high-confidence benign predictions from artificially increasing risk.
- Final-test feedback updates history/policy state only. Neural weights remain frozen to prevent test leakage.
- AES-GCM uses a fresh 96-bit nonce. High-risk quarantine uses ECDHE P-256, ECDSA P-256 endpoint authentication, HKDF-SHA-256, then AES-256-GCM.
- Encrypted quarantine serializes and encrypts the flow object; it is not only a Boolean label.

## 7. Outputs

Outputs are written under `results/`, `logs/`, `checkpoints/`, and `quarantine/`.

## 8. Reproducibility

Default seed: 42. Manuscript settings are stored in `config.yaml`. Five-run seeds are 42, 52, 62, 72, 82.
