Standard Notes

Folder Contents
- `train.py` Training script .
- `test.py` Evaluation script.
- `test-tnse.py` Evaluation script.
- `checkpoints/` Trained model weights and best threshold.
- `tsne_2d_*.png` t-SNE 2D visualization images.
- `__init__.py` Package initializer.

Training Command
```bash
CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/xxxx/bin/python CIC_IOMT_2024/train.py
```

Test Command
```bash
CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/xxxx/bin/python CIC_IOMT_2024/test.py \
  --ckpt CIC_IOMT_2024/checkpoints/CIC-IOMT-2024.pth \
  --data ./data/CIC-IOMT-2024.pt \
  --use_scaler \
  --train_frac 0.99 \
  --thr 0.23
```

Arguments

- `--ckpt` Path to the pretrained model checkpoint.
- `--data` Path to the graph dataset file (`.pt`).
- `--train_frac` Tag data ratio.
- `--thr` Decision threshold for binary classification.

Model Weights (Google Drive)

- https://drive.google.com/file/d/1TTWoiRlvHNCs6N4mNDVDBAzqdnA9JYK-/view?usp=drive_link
