# Med-Diff

This repository contains the Med-Diff code for traffic anomaly detection
experiments.

## Repository Layout

```text
.
|-- 2024-iomt-traffic-data/
|   |-- train.py
|   |-- test.py
|   `-- __init__.py
|-- CIC_IOMT_2024/
|   |-- train.py
|   |-- test.py
|   `-- __init__.py
|-- CIC_TON_IOT/
|   |-- train.py
|   |-- test.py
|   `-- __init__.py
|-- NF-UNSW-NB15/
|   |-- train.py
|   |-- test.py
|   `-- __init__.py
|-- Figure/
|   `-- model_architecture.png
|-- model/
|   |-- Med_Diff.py
|   `-- __init__.py
|-- requirements.txt
`-- readme.md
```

## Code Overview

`model/Med_Diff.py` contains the Med-Diff model implementation. The model
architecture figure is stored in `Figure/model_architecture.png`.

![Med-Diff framework](Figure/model_architecture.png)

## Environment

Install the Python dependencies listed in `requirements.txt`.

## Training

```bash
python 2024-iomt-traffic-data/train.py
python CIC_IOMT_2024/train.py
python CIC_TON_IOT/train.py
python NF-UNSW-NB15/train.py
```

## Testing

```bash
python 2024-iomt-traffic-data/test.py
python CIC_IOMT_2024/test.py
python CIC_TON_IOT/test.py
python NF-UNSW-NB15/test.py
```

## Contact

If you have any questions, please contact wxybboy8340@163.com.
