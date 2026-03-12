Standard Notes

Folder Contents
- `train.py` Training script.
- `test.py` Evaluation script.
- `test-tnse.py` Evaluation script.
- `checkpoints/` Trained model weights and best threshold.
- `tsne_2d_*.png` t-SNE 2D visualization images.
- `__init__.py` Package initializer.

Training Command
```bash
CUDA_VISIBLE_DEVICES=1  /opt/anaconda3/envs/xxxx/bin/python 2024-iomt-traffic-data/train.py
```

Test Command
```bash
 CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/envs/xxxx/bin/python 2024-iomt-traffic-data/test.py \
--ckpt 2024-iomt-traffic-data/checkpoints/2024-iomt-traffic-data.pth \
--data ./data/2024-iomt-traffic-data.pt \
--use_scaler \
--train_frac 0.01 \
--thr 0.56
```

Arguments
- `--ckpt` Path to the pretrained model checkpoint.
- `--data` Path to the graph dataset file (`.pt`).
- `--train_frac` Tag data ratio.
- `--thr` Decision threshold for binary classification.


Model Weights (Google Drive)
- https://drive.google.com/file/d/1TTWoiRlvHNCs6N4mNDVDBAzqdnA9JYK-/view?usp=drive_link
