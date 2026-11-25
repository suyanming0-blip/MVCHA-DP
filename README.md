# MVCHA-DP

Code for **Adaptive Multi-View Hypergraph Representation Learning for Multi-Scenario Software Defect Prediction**.

MVCHA-DP combines multi-view code features with adaptive hypergraph learning to model relationships among software modules. It supports within-project (WPDP), cross-version (CVDP), and cross-project (CPDP) defect prediction, with domain adaptation for the cross-domain scenarios.

## Installation

Use Python 3.10 and an NVIDIA GPU with CUDA support. The implementation uses PyTorch 1.12.1 with CUDA 11.6. Install the dependencies from the repository directory:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

## Running experiments

From the repository directory, start the full experiment:

```bash
python main.py
```

The program runs all three scenarios sequentially with 20 random seeds (321–340).

| Scenario | Tasks | Trainings per task | Total trainings |
|---|---:|---:|---:|
| WPDP | 70 | 100 | 7,000 |
| CVDP | 62 | 20 | 1,240 |
| CPDP | 192 | 20 | 3,840 |
| Total | 324 | | 12,080 |

WPDP uses stratified five-fold evaluation for each seed: three folds for training, one for validation, and one for testing. For test fold `i`, validation uses fold `(i+1)%5`. Each fold trains independently.

CVDP and CPDP retain the source training/validation split across seeds and evaluate on the complete target project. Neural training and the final readout use the same seed. Validation selects the checkpoint before test evaluation.

## Results

Each invocation creates a new timestamped directory:

```text
results/full_<timestamp>/
├── plan/
└── results/
    ├── summary.json
    ├── progress.jsonl
    ├── logs/
    └── runs/
```

`summary.json` reports aggregate AUC, F1, and MCC. WPDP metrics are averaged over five folds within each task and seed. Tasks receive equal weight within each scenario, and the final summary reports the mean and standard deviation over the 20 seed means.

Individual metrics and selected model snapshots are saved under `runs/`. `progress.jsonl` records completed runs, and `logs/` contains their output.
