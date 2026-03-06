Standard Notes

Folder Contents
- `train.py` Training script (Med_Diff ).
- `test.py` Evaluation script.
- `test-tnse.py` Evaluation script.
- `checkpoints/` Trained model weights and best threshold.
- `tsne_2d_*.png` t-SNE 2D visualization images.
- `__init__.py` Package initializer.

Training Command
```bash
CUDA_VISIBLE_DEVICES=2  /opt/anaconda3/envs/xxxxx/bin/python NF-UNSW-NB15/train.py
```

Test Command
```bash
CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/xxxxx/bin/python NF-UNSW-NB15/test.py \
  --ckpt NF-UNSW-NB15/checkpoints/NF-UNSW-NB15.pth \
  --data ./data/NF-UNSW-NB15.pt\
  --use_scaler \
  --train_frac 0.00001 \
  --thr 0.88
```

Arguments

- `--ckpt` Path to the pretrained model checkpoint.
- `--data` Path to the graph dataset file (`.pt`).
- `--use_scaler` Enable feature normalization.
- `--train_frac` Fraction of training data.
- `--thr` Decision threshold for binary classification.

Model Weights (Google Drive)

- https://drive.google.com/file/d/1TTWoiRlvHNCs6N4mNDVDBAzqdnA9JYK-/view?usp=drive_link

