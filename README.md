# Face Recognition Framework

PyTorch face recognition training/evaluation framework with configurable backbones and margin-based heads.

- 中文版文档: [README_zh.md](README_zh.md)

## Features

- Training, validation (pairs), and checkpointing
- Evaluation on `.bin` pair datasets (e.g., LFW / CFP-FP / AgeDB-30)
- YAML-driven configuration
- TensorBoard logging (optional)
- Resume training from latest run directory

## Project Layout

- `train.py`: training entrypoint
- `test.py`: evaluation entrypoint (pairs verification)
- `engine/`
  - `trainer.py`: training loop + validation + checkpointing
  - `evaluator.py`: pair verification metrics (10-fold accuracy, TAR@FAR, AUC)
  - `predictor.py`: two-image similarity inference helper
- `models/`
  - `backbones/`: feature extractors
  - `heads/`: margin-based classification heads
  - `model_factory.py`: `build_model(config)` factory
- `data/`
  - `dataset.py`: `FaceEmoreDataset` (RecordIO) + `BinPairDataset` (`.bin`)
  - `transforms.py`: train/val transforms
- `utils/`
  - `checkpoint.py`: save/load checkpoints
  - `logger.py`: file logger + TensorBoard
  - `metrics.py`: verification metrics
  - `common.py`: reproducibility seed helper
- `config/`: example configs

## Installation

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

Notes:

- `mxnet` is required to read InsightFace RecordIO training data (`train.rec` / `train.idx`).
- `scikit-learn` is required for ROC/AUC/TAR@FAR metrics.

## Data Preparation

### Training set (InsightFace RecordIO)

Your dataset root (set by `data.root` in YAML) should contain:

- `train.rec`
- `train.idx`
- `property` (comma-separated, at least: `num_classes,height,width`)

Example:

```yaml
data:
  root: /data/faces_emore
```

### Evaluation set (`.bin` verification pairs)

Put verification bins under the same dataset root:

- `lfw.bin`
- `cfp_fp.bin`
- `agedb_30.bin`

Select which one to evaluate via:

```yaml
eval:
  bin_file: "lfw.bin"
```

## Training

### Quick Start

```bash
python train.py --config config/train_resnet50.yaml --device cuda
```

Resume from the latest run directory under `checkpoint.save_dir`:

```bash
python train.py --config config/train_resnet50.yaml --resume
```

### Training Configuration (YAML)

Key fields used by the trainer:

- `seed`: random seed (default: 42)
- `device`: runtime override supported via `--device` (`cuda` or `cpu`)
- `model.backbone`: backbone name (`resnet50` or `fastcontextface`)
- `model.embedding_size`: embedding dimension (e.g., 512)
- `head.type`: head type (`arcface` / `cosface` / `adaface`)
- `head.num_classes`: class count (must match training dataset classes)
- `data.root`: dataset root directory
- `data.batch_size`, `data.num_workers`, `data.img_size`
- `training.epochs`, `training.optimizer`, `training.lr`, `training.weight_decay`, `training.momentum`
- `training.scheduler`: `multistep` or `cosine`
- `training.warmup_epochs`
- `training.amp`: enable automatic mixed precision on CUDA (default: true)
- `training.grad_clip_norm`: gradient clipping norm (default: 5.0)
- `training.resume`: enable resume logic (also can be set by `--resume`)
- `checkpoint.save_dir`: checkpoint root directory
- `logging.log_dir`: log directory (files + tensorboard)
- `eval.bin_file`, `eval.test_batch_size`, `eval.eval_freq`

## Evaluation (Verification)

Evaluate a trained checkpoint against the `.bin` pair dataset:

```bash
python test.py \
  --config config/train_resnet50.yaml \
  --checkpoint checkpoints/resnet50/<RUN_TIMESTAMP>/best.pth \
  --name resnet50_adaface
```

Artifacts:

- Log file in `logs/...`
- A summary text file `test_result_<name>_<timestamp>.txt`

## Inference (Two-Image Similarity)

`engine/predictor.py` provides a simple API to compute cosine similarity between two images:

```python
from engine.predictor import Predictor

pred = Predictor(
    config_path="config/train_resnet50.yaml",
    checkpoint_path="checkpoints/resnet50/<RUN_TIMESTAMP>/best.pth",
    use_cpu=False,
)
score = pred.predict("a.jpg", "b.jpg")
print(score)
```

## Reproducibility

Training seeds are set by `seed` in YAML. Deterministic mode can be controlled by:

```yaml
deterministic: true
```

When `deterministic=true`, cuDNN benchmarking is disabled to improve reproducibility.
