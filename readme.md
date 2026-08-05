# Med-Diff Evaluation Code

This repository contains the runnable training and evaluation code for Med-Diff
traffic anomaly detection experiments. The repository is intentionally kept
lean: source code, model definitions, documentation, the architecture figure,
and the checkpoint files that are currently available locally.

Large graph datasets, generated visualizations, Python bytecode caches, and
unused helper modules are not kept in the working tree.

## Repository Layout

```text
.
|-- 2024-iomt-traffic-data/
|   |-- train.py
|   |-- test.py
|   `-- readme.md
|-- CIC_IOMT_2024/
|   |-- train.py
|   |-- test.py
|   |-- readme.md
|   `-- checkpoints/CIC-IOMT-2024.pth
|-- CIC_TON_IOT/
|   |-- train.py
|   |-- test.py
|   |-- readme.md
|   `-- checkpoints/best_CIC-ToN-IoT.pth
|-- NF-UNSW-NB15/
|   |-- train.py
|   |-- test.py
|   `-- readme.md
|-- Figure/model_architecture.png
|-- model/Med_Diff.py
|-- utils/
|-- link.txt
`-- requirements.txt
```

Empty checkpoint folders are not stored. If you download or train additional
weights, create the corresponding `checkpoints/` directory under the dataset
folder.

## Model Overview

`model/Med_Diff.py` defines the Med-Diff encoder. The dataset scripts combine
PyTorch Geometric temporal graph loading, `TGNMemory`, recent-neighbor rollout,
and the Med-Diff implicit diffusion module to produce edge-pair embeddings.

Training uses self-supervised representation learning losses inside Med-Diff.
Evaluation fits a lightweight `LogisticRegression` probe on a configurable
fraction of training embeddings and applies a binary anomaly threshold.

![Med-Diff framework](Figure/model_architecture.png)

## Environment

The original experiments target Python 3.10 with a CUDA-compatible PyTorch and
PyTorch Geometric stack. The project requirements are listed in
`requirements.txt`; install `torch`, `torch-geometric`, and `torch-scatter`
using wheels that match your CUDA and PyTorch versions.

## Training

Training hyperparameters are defined in each dataset script's `Config` class.
Run from the repository root:

```bash
python 2024-iomt-traffic-data/train.py
python CIC_IOMT_2024/train.py
python CIC_TON_IOT/train.py
python NF-UNSW-NB15/train.py
```

## Testing

Use the dataset-specific README files for the recommended checkpoint path,
`--train_frac`, and `--thr` values. Example:

```bash
python CIC_IOMT_2024/test.py \
  --ckpt CIC_IOMT_2024/checkpoints/CIC-IOMT-2024.pth \
  --data ./data/CIC-IOMT-2024.pt \
  --use_scaler \
  --train_frac 0.99 \
  --thr 0.23
```

Test scripts may generate local t-SNE visualization images. These are runtime
outputs and are not committed.
