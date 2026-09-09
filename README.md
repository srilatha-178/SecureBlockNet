[![DOI](https://zenodo.org/badge/1362446300.svg)](https://doi.org/10.5281/zenodo.22671811)
# SecureBlockNet

## A Deep Learning-Enhanced Cryptographic Framework for Intelligent Traffic Blocking in Secure Networks

SecureBlockNet is a research-oriented cybersecurity framework that integrates deep learning-based encrypted traffic classification with risk-aware cryptographic policy selection and intelligent traffic enforcement.

The implemented pipeline is:

```text
Network Traffic
        |
        v
Leakage-Free Preprocessing
        |
        v
1D-CNN Feature Extraction
        |
        v
BiLSTM Temporal Modelling
        |
        v
Attention-Based Feature Prioritization
        |
        v
Traffic Class + Prediction Confidence
        |
        v
Risk Scoring
(Confidence + Attack Severity + Historical Behaviour)
        |
        v
Risk-Adaptive Cryptographic Policy
        |
        v
Allow / Monitor / Block / Encrypted Quarantine
        |
        v
Security Logging and Feedback Refinement
```

---

## 1. Research Motivation

Modern encrypted network traffic reduces the effectiveness of payload-oriented intrusion detection and traditional deep packet inspection.

SecureBlockNet addresses this problem using flow-level traffic features and a hybrid **1D-CNN-BiLSTM-Attention architecture**.

The resulting class prediction and confidence are not treated as the final security decision. Instead, they are combined with attack severity and historical traffic behaviour to estimate a calibrated risk score, which controls cryptographic policy selection and traffic enforcement.

The framework supports four traffic-control actions:

* **Allow** — securely forward low-risk traffic.
* **Monitor** — retain suspicious traffic under stronger cryptographic protection and additional inspection.
* **Block** — deny highly confident high-risk malicious traffic access to protected resources.
* **Encrypted Quarantine** — cryptographically protect and isolate high-risk but comparatively uncertain traffic for controlled analysis.

---

## 2. Main Contributions Implemented

This repository implements the following research components:

* Hybrid 1D-CNN + BiLSTM + Attention traffic classifier.
* Seven-class harmonized attack taxonomy for heterogeneous intrusion datasets.
* Leakage-free train/validation/test preprocessing.
* Correlation-based feature filtering and mutual-information ranking.
* Temporal flow-window construction for meaningful recurrent modelling.
* Prediction-confidence extraction from Softmax output.
* Attack severity estimation using abnormality, protocol impact, and persistence.
* Historical traffic behaviour modelling.
* Risk scoring using calibrated confidence, severity, and history weights.
* Adaptive AES-128-GCM and AES-256-GCM security profiles.
* HKDF-SHA-256 session-key derivation.
* ECDHE-P256 ephemeral secret establishment for high-risk containment.
* ECDSA-P256 endpoint authentication.
* Intelligent allow/monitor/block/encrypted-quarantine decisions.
* Security logging and feedback-state refinement.
* Intra-dataset and strict cross-dataset evaluation.
* Baseline comparison, ablation, sensitivity analysis, cryptographic benchmarking, and statistical evaluation.

---

## 3. Datasets

The implementation is designed for the following public benchmark datasets:

| Dataset         | Primary Use                                               |
| --------------- | --------------------------------------------------------- |
| CICIoT2023      | Main SecureBlockNet training and intra-dataset evaluation |
| CSE-CIC-IDS2018 | Cross-environment enterprise-network validation           |
| CIC-IDS2017     | Additional robustness and cross-dataset evaluation        |

Dataset files are not included in this repository because of their size and distribution terms.

Download the datasets from their official or approved public sources and place the extracted CSV/CSV.GZ files under `data/raw/`.

Recommended layout:

```text
data/
└── raw/
    ├── ciciot2023/
    ├── cicids2018/
    └── cicids2017/
```

The dataset loader searches recursively for `.csv` and `.csv.gz` files.

---

## 4. Unified Attack Taxonomy

For cross-dataset analysis, dataset-specific labels are mapped into the following seven classes:

| ID | Unified Class     |
| -: | ----------------- |
|  0 | Benign            |
|  1 | DoS               |
|  2 | DDoS              |
|  3 | Reconnaissance    |
|  4 | Spoofing          |
|  5 | Web / Brute-Force |
|  6 | Other Malicious   |

Dataset-specific mappings are handled in the data-processing modules.

---

## 5. Repository Structure

```text
SecureBlockNet/
│
├── README.md
├── requirements.txt
├── config.yaml
├── verify_install.py
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── mappings/
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── label_mapping.py
│   │   ├── harmonization.py
│   │   ├── preprocessing.py
│   │   ├── sequence_builder.py
│   │   └── splitting.py
│   │
│   ├── models/
│   │   ├── attention.py
│   │   ├── secureblocknet.py
│   │   └── baselines.py
│   │
│   ├── risk/
│   │   ├── abnormality.py
│   │   ├── severity.py
│   │   ├── history.py
│   │   ├── risk_engine.py
│   │   └── calibration.py
│   │
│   ├── crypto/
│   │   ├── crypto_engine.py
│   │   └── policy_manager.py
│   │
│   ├── enforcement/
│   │   ├── traffic_controller.py
│   │   └── feedback.py
│   │
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── blocking_metrics.py
│   │   ├── crypto_metrics.py
│   │   ├── statistical_tests.py
│   │   └── plots.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   └── timing.py
│   │
│   └── pipeline.py
│
├── experiments/
│   ├── 00_demo_end_to_end.py
│   ├── 01_dataset_analysis.py
│   ├── 02_train_baselines.py
│   ├── 03_train_secureblocknet.py
│   ├── 04_risk_calibration.py
│   ├── 05_crypto_benchmark.py
│   ├── 06_traffic_blocking.py
│   ├── 07_cross_dataset.py
│   ├── 08_ablation.py
│   ├── 09_sensitivity.py
│   ├── 10_statistical_analysis.py
│   └── 11_repeated_runs.py
│
├── tests/
│   └── test_core_non_tf.py
│
├── checkpoints/
├── logs/
├── quarantine/
│
└── results/
    ├── figures/
    ├── tables/
    ├── predictions/
    └── statistics/
```

---

## 6. Software Requirements

Recommended environment:

* Python 3.11
* TensorFlow 2.16.1
* Keras 3.3.3
* Scikit-learn 1.5.1
* NumPy 1.26.4
* Pandas 2.2.2
* SciPy
* Matplotlib
* cryptography

A CUDA-compatible NVIDIA GPU is strongly recommended for large-scale training, although the core code can run on CPU for testing and smaller experiments.

---

## 7. Installation

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SecureBlockNet
```

### Create a virtual environment

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify the installation

```bash
python verify_install.py
```

---

## 8. Quick Functional Test

The repository includes a synthetic end-to-end demonstration so that the software pipeline can be checked before downloading the benchmark datasets.

```bash
python experiments/00_demo_end_to_end.py --epochs 2 --samples 3000
```

This test validates the implementation workflow only. Synthetic-data results must not be reported as experimental results of the research paper.

---

## 9. Dataset Preprocessing

The preprocessing pipeline performs:

1. Recursive dataset loading.
2. Column-name normalization.
3. Removal of duplicate records.
4. Replacement/removal of invalid and infinite values.
5. Label harmonization.
6. Stratified train/validation/test splitting.
7. Correlation-based feature removal.
8. Mutual-information feature ranking.
9. Min-Max scaling.
10. Temporal sequence construction.

To prevent information leakage, feature selection and scaling are fitted on the training partition only and subsequently applied unchanged to the validation and test partitions.

Default split:

| Partition  | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

---

## 10. Temporal Sequence Construction

SecureBlockNet uses genuine ordered traffic-flow windows rather than simply reshaping one independent tabular row into a pseudo-sequence.

Default initial sequence length:

```text
10 consecutive flows
```

If appropriate chronological or grouping fields are present, samples should be sorted/grouped before sequence construction.

Sequence windows must never cross train/validation/test boundaries.

---

## 11. SecureBlockNet Architecture

The implemented deep-learning architecture follows:

```text
Input Sequence
      |
      v
Conv1D(64, kernel=3, ReLU)
      |
      v
Batch Normalization
      |
      v
Max Pooling
      |
      v
Conv1D(128, kernel=3, ReLU)
      |
      v
Batch Normalization
      |
      v
Max Pooling
      |
      v
Bidirectional LSTM (128 units)
      |
      v
Temporal Attention
      |
      v
Dense Representation
      |
      v
Dropout (0.5)
      |
      v
Softmax Classification
```

Default training settings:

| Parameter               | Value   |
| ----------------------- | ------- |
| Optimizer               | Adam    |
| Learning rate           | 0.001   |
| Batch size              | 64      |
| Maximum epochs          | 100     |
| Early-stopping patience | 10      |
| CNN filters             | 64, 128 |
| CNN kernel size         | 3       |
| BiLSTM units            | 128     |
| Dropout                 | 0.5     |
| Hidden activation       | ReLU    |
| Output activation       | Softmax |
| Default seed            | 42      |

---

## 12. Train SecureBlockNet

Example using CICIoT2023:

```bash
python experiments/03_train_secureblocknet.py \
    --dataset ciciot2023 \
    --path data/raw/ciciot2023
```

Expected outputs include trained checkpoints, prediction files, evaluation metrics, and figures under the configured output directories.

---

## 13. Train Baseline Models

The repository includes commonly used baselines for comparison:

* Support Vector Machine (SVM)
* Random Forest
* 1D-CNN
* LSTM
* CNN-LSTM

Run:

```bash
python experiments/02_train_baselines.py \
    --dataset ciciot2023 \
    --path data/raw/ciciot2023
```

---

## 14. Risk-Aware Decision Engine

The overall traffic-risk score is implemented as:

```text
Risk = 0.45 × PredictionConfidence
     + 0.35 × AttackSeverity
     + 0.20 × HistoricalBehaviour
```

The attack-severity score combines:

```text
Severity = 0.40 × TrafficAbnormality
         + 0.35 × ProtocolThreatImpact
         + 0.25 × AttackPersistence
```

The default calibrated thresholds are:

| Condition                     | Decision             |
| ----------------------------- | -------------------- |
| Risk < 0.35                   | Low Risk             |
| 0.35 ≤ Risk < 0.70            | Medium Risk          |
| Risk ≥ 0.70                   | High Risk            |
| High Risk + Confidence ≥ 0.85 | Direct Block         |
| High Risk + Confidence < 0.85 | Encrypted Quarantine |

### Important Operational Detail

For a benign prediction, the implementation uses malicious confidence rather than raw maximum Softmax confidence.

This prevents a highly confident benign prediction from incorrectly increasing the malicious risk score.

---

## 15. Cryptographic Profiles

| Risk   | Security Profile                      | Traffic Handling                |
| ------ | ------------------------------------- | ------------------------------- |
| Low    | AES-128-GCM                           | Allow / secure forwarding       |
| Medium | AES-256-GCM                           | Monitor / controlled inspection |
| High   | AES-256-GCM + ECDHE-P256 + ECDSA-P256 | Block / encrypted quarantine    |

The implementation also uses:

* HKDF-SHA-256 for session-specific symmetric-key derivation.
* Unique 96-bit nonces for AES-GCM operations.
* 128-bit GCM authentication tags.
* ECDHE over NIST P-256 / SECP256R1 for high-risk ephemeral secret establishment.
* ECDSA-P256 for trusted endpoint authentication.

The risk score, session identifier, timestamp, and selected policy may be used as contextual information during key derivation, but they are not treated as cryptographic entropy.

Secret entropy is derived from securely established keying material.

---

## 16. Intelligent Traffic Enforcement

The decision logic is:

```text
LOW RISK
    |
    v
AES-128-GCM
    |
    v
ALLOW / SECURE FORWARDING


MEDIUM RISK
    |
    v
AES-256-GCM
    |
    v
MONITOR / CONTROLLED INSPECTION


HIGH RISK + HIGH CONFIDENCE
    |
    v
STRONG SECURITY PROFILE
    |
    v
BLOCK


HIGH RISK + LOWER CONFIDENCE
    |
    v
AES-256-GCM + ECDHE-P256 + ECDSA-P256
    |
    v
ENCRYPTED QUARANTINE
```

Encrypted quarantine serializes and cryptographically protects the suspicious flow object before controlled isolation.

It is therefore implemented as more than a Boolean classification label.

---

## 17. Risk Calibration

Run risk-weight calibration using saved prediction outputs:

```bash
python experiments/04_risk_calibration.py \
    --predictions results/predictions/ciciot2023_predictions.csv
```

The implemented candidate configurations include:

| Configuration       | Confidence | Severity | History |
| ------------------- | ---------: | -------: | ------: |
| Equal weighting     |       0.33 |     0.33 |    0.34 |
| Confidence dominant |       0.60 |     0.25 |    0.15 |
| Severity dominant   |       0.25 |     0.60 |    0.15 |
| Proposed calibrated |       0.45 |     0.35 |    0.20 |

---

## 18. Cryptographic Benchmark

Run:

```bash
python experiments/05_crypto_benchmark.py
```

The benchmark can measure operations such as:

* AES-128-GCM encryption/decryption time.
* AES-256-GCM encryption/decryption time.
* HKDF-SHA-256 derivation time.
* ECDHE-P256 key-establishment time.
* ECDSA-P256 signing and verification time.
* High-risk containment processing latency.
* Encryption overhead and throughput.

---

## 19. Traffic-Blocking Evaluation

Run:

```bash
python experiments/06_traffic_blocking.py \
    --predictions results/predictions/ciciot2023_predictions.csv
```

Traffic-level evaluation can include:

* Attack blocking rate.
* False blocking rate.
* Monitoring rate.
* Quarantine rate.
* Security reliability.
* Blocking latency.
* Decision latency.
* Cryptographic overhead.
* Total processing latency.

---

## 20. Strict Cross-Dataset Evaluation

The repository implements a strict generalization protocol:

```text
Train:
    CICIoT2023

Freeze:
    - selected feature set
    - preprocessing transformations
    - Min-Max scaler
    - trained neural weights
    - risk coefficients
    - decision thresholds

Test without retraining:
    - CSE-CIC-IDS2018
    - CIC-IDS2017
```

Run:

```bash
python experiments/07_cross_dataset.py \
    --train-path data/raw/ciciot2023 \
    --ids2018-path data/raw/cicids2018 \
    --ids2017-path data/raw/cicids2017
```

Only semantically harmonized flow-level features are used.

The external datasets do not refit the training-derived scaler or feature-selection pipeline.

---

## 21. Ablation Study

Run:

```bash
python experiments/08_ablation.py \
    --dataset ciciot2023 \
    --path data/raw/ciciot2023
```

Recommended variants include:

| Variant             | CNN | BiLSTM | Attention | Risk Engine | Crypto Enforcement |
| ------------------- | :-: | :----: | :-------: | :---------: | :----------------: |
| A1                  | Yes |        |           |             |                    |
| A2                  | Yes |   Yes  |           |             |                    |
| A3                  | Yes |   Yes  |    Yes    |             |                    |
| A4                  | Yes |   Yes  |    Yes    |     Yes     |                    |
| Full SecureBlockNet | Yes |   Yes  |    Yes    |     Yes     |         Yes        |

This experiment helps distinguish classifier improvements from the contribution of risk-aware security enforcement.

---

## 22. Sensitivity Analysis

Run:

```bash
python experiments/09_sensitivity.py \
    --predictions results/predictions/ciciot2023_predictions.csv
```

Sensitivity analysis evaluates the effects of changing:

* Risk-component weights.
* Risk thresholds.
* Confidence threshold.
* Severity-component weights.
* Monitoring/history parameters.

---

## 23. Repeated Runs and Statistical Analysis

Recommended seeds:

```text
42, 52, 62, 72, 82
```

Run repeated experiments:

```bash
python experiments/11_repeated_runs.py \
    --dataset ciciot2023 \
    --path data/raw/ciciot2023
```

Then perform statistical analysis:

```bash
python experiments/10_statistical_analysis.py \
    --csv results/statistics/repeated_runs.csv
```

Recommended reporting:

* Mean.
* Standard deviation.
* 95% confidence interval.
* Paired t-test where assumptions are appropriate.
* Wilcoxon signed-rank test as a non-parametric alternative.

---

## 24. Evaluation Metrics

### Classification

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* ROC-AUC
* Class-wise Precision/Recall/F1
* Detection Rate
* False Positive Rate
* Confusion Matrix

### Security Enforcement

* Attack Blocking Rate
* False Blocking Rate
* Security Reliability
* Monitoring Rate
* Quarantine Rate

### Computational / Cryptographic

* Classification latency
* Risk-computation latency
* Cryptographic processing latency
* Blocking latency
* Total processing latency
* Encryption overhead
* Throughput
* Ciphertext expansion

---

## 25. Output Directories

Generated artifacts are stored under:

```text
results/
├── figures/
├── tables/
├── predictions/
└── statistics/

checkpoints/
logs/
quarantine/
```

Do not commit large trained models, raw datasets, or generated result files unless needed for a tagged release.

---

## 26. Reproducibility Notes

The main reproducibility controls are defined in `config.yaml`.

Key defaults:

| Setting                 | Value                   |
| ----------------------- | ----------------------- |
| Random seed             | 42                      |
| Split                   | 70 / 15 / 15            |
| Optimizer               | Adam                    |
| Learning rate           | 0.001                   |
| Batch size              | 64                      |
| Maximum epochs          | 100                     |
| Early stopping patience | 10                      |
| Checkpoint criterion    | Minimum validation loss |

Important safeguards:

* Preprocessing is trained only on the training partition.
* Validation data is used for parameter/model selection.
* Test data is isolated from tuning.
* External datasets do not refit the CICIoT2023 preprocessing pipeline.
* Model weights remain frozen during final test-time feedback evaluation.
* Security-state feedback does not retrain the neural classifier on test samples.

---

## 27. Testing

Run the included tests with:

```bash
pytest -q
```

The non-TensorFlow core tests validate major preprocessing/risk/cryptographic components.

For full deep-learning verification, TensorFlow and the required dependencies must be installed.

---

## 28. Recommended GitHub .gitignore

A GitHub repository should normally ignore raw data, checkpoints, temporary logs, generated results, quarantine contents, caches, and virtual environments.

Suggested entries:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyd
*.so
.pytest_cache/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Raw and processed datasets
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Model checkpoints
checkpoints/*
!checkpoints/.gitkeep
*.keras
*.h5

# Generated outputs
results/figures/*
results/tables/*
results/predictions/*
results/statistics/*
logs/*
quarantine/*

# Keep folders
!results/figures/.gitkeep
!results/tables/.gitkeep
!results/predictions/.gitkeep
!results/statistics/.gitkeep
!logs/.gitkeep
!quarantine/.gitkeep
```

---

## 29. Security and Scope

This repository is intended for defensive cybersecurity research and reproducible academic experimentation.

The current implementation assumes that:

* The SecureBlockNet enforcement environment is trusted.
* Long-term authentication credentials are securely provisioned.
* Cryptographic key material is securely stored.
* Cryptographic libraries are correctly implemented.
* A secure random-number generator is available.
* AES-GCM nonces are not reused under the same key.

The current experimental scope does not claim protection against:

* Physical compromise of trusted endpoints.
* Hardware side-channel attacks.
* Operating-system compromise.
* Cryptographic-library compromise.
* Secret-key extraction.
* Model poisoning.
* Carefully optimized adversarial evasion.
* Post-quantum attacks against P-256 elliptic-curve mechanisms.

---

## 30. Citation

If you use this repository in academic work, cite the associated manuscript after its bibliographic details are finalized.

Suggested temporary citation format:

```bibtex
@article{secureblocknet,
    title   = {A Deep Learning-Enhanced Cryptographic Framework for Intelligent Traffic Blocking in Secure Networks},
    author  = {Srilatha},
    journal = {To be updated},
    year    = {2026}
}
```

Replace the placeholder metadata with the final published citation, DOI, and journal information after publication.

---

## 31. License

Add the license selected for your repository, for example:

```text
MIT License
Apache License 2.0
BSD 3-Clause License
```

For academic code releases, ensure that the selected license is consistent with institutional, sponsor, and dataset-license requirements.

---

## 32. Contact

For research questions, reproducibility issues, or collaboration requests, add the corresponding author/research-group contact information here.

```text
Research Group / Laboratory: <TO BE ADDED>
Corresponding Author: <TO BE ADDED>
Email: <TO BE ADDED>
Institution: <TO BE ADDED>
```

---

## 33. Disclaimer

SecureBlockNet is a research prototype.

Results depend on dataset versions, preprocessing choices, available flow fields, hardware, library versions, and experimental settings.

The repository should not be treated as a production firewall, intrusion-prevention appliance, or certified cryptographic product without additional security engineering, deployment validation, and independent review.
