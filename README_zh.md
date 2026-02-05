# 人脸识别训练框架

基于 PyTorch 的人脸识别训练/验证/测试框架，骨干网络（backbone）与分类头（margin head）均可通过 YAML 配置切换。

## 功能特点

- 训练、验证（成对验证集）、断点恢复与模型保存
- 支持 `.bin` 成对验证集（例如 LFW / CFP-FP / AgeDB-30）
- YAML 驱动的实验配置
- 可选 TensorBoard 记录
- 自动从 `checkpoint.save_dir` 下最近一次运行目录恢复训练

## 目录结构

- `train.py`：训练入口
- `test.py`：验证/测试入口（成对验证）
- `engine/`
  - `trainer.py`：训练循环 + 验证 + checkpoint
  - `evaluator.py`：成对验证指标（10-fold accuracy、TAR@FAR、AUC）
  - `predictor.py`：两张图片相似度推理
- `models/`
  - `backbones/`：特征提取网络（不建议在整理过程中修改）
  - `heads/`：ArcFace/CosFace/AdaFace 等头部（不建议在整理过程中修改）
  - `model_factory.py`：`build_model(config)` 工厂
- `data/`
  - `dataset.py`：`FaceEmoreDataset`（RecordIO）与 `BinPairDataset`（.bin）
  - `transforms.py`：训练/验证变换
- `utils/`
  - `checkpoint.py`：保存/加载权重
  - `logger.py`：日志与 TensorBoard
  - `metrics.py`：验证指标
  - `common.py`：随机种子与可复现设置
- `config/`：示例训练配置

## 安装

建议 Python 3.9+。

```bash
pip install -r requirements.txt
```

说明：

- `mxnet`：读取 InsightFace RecordIO 训练集（`train.rec`/`train.idx`）必需。
- `scikit-learn`：用于 ROC/AUC/TAR@FAR 的计算。

## 数据准备

### 训练集（InsightFace RecordIO）

YAML 中的 `data.root` 指向数据根目录，该目录需要包含：

- `train.rec`
- `train.idx`
- `property`（逗号分隔，至少包括：`num_classes,height,width`）

示例：

```yaml
data:
  root: /data/faces_emore
```

### 验证集（`.bin` 成对数据）

将验证 `.bin` 放在同一个 `data.root` 下，例如：

- `lfw.bin`
- `cfp_fp.bin`
- `agedb_30.bin`

通过配置选择要评估的文件：

```yaml
eval:
  bin_file: "lfw.bin"
```

## 训练

### 快速开始

```bash
python train.py --config config/train_resnet50.yaml --device cuda
```

从 `checkpoint.save_dir` 下最近一次运行目录恢复训练：

```bash
python train.py --config config/train_resnet50.yaml --resume
```

### 训练配置（YAML）关键项

Trainer 会读取并使用以下字段：

- `seed`：随机种子（默认 42）
- `deterministic`：是否启用更强的可复现模式（默认 true）
- `device`：也支持通过命令行 `--device` 覆盖（`cuda` 或 `cpu`）
- `model.backbone`：骨干网络（`resnet50` 或 `fastcontextface`）
- `model.embedding_size`：embedding 维度（如 512）
- `head.type`：`arcface` / `cosface` / `adaface`
- `head.num_classes`：类别数（应与训练集 `property` 的 num_classes 一致）
- `data.root`：数据根目录
- `data.batch_size` / `data.num_workers` / `data.img_size`
- `training.epochs` / `training.optimizer` / `training.lr` / `training.weight_decay` / `training.momentum`
- `training.scheduler`：`multistep` 或 `cosine`
- `training.warmup_epochs`：warmup epoch 数
- `training.amp`：CUDA 上是否启用自动混合精度（默认 true）
- `training.grad_clip_norm`：梯度裁剪阈值（默认 5.0）
- `training.resume`：是否启用恢复训练逻辑（也可由 `--resume` 打开）
- `checkpoint.save_dir`：checkpoint 根目录
- `logging.log_dir`：日志目录（log + tensorboard）
- `eval.bin_file` / `eval.test_batch_size` / `eval.eval_freq`

### 训练设置建议（最佳实践）

- **batch size**：显存不足时优先降低 `data.batch_size`，再降低 `eval.test_batch_size`。
- **num_workers**：Linux 服务器通常 4~16 合理；Windows 建议从 0/2 开始排查问题。
- **AMP**：在 CUDA 上默认开启，能显著加速并降低显存占用；如遇数值不稳定可临时关闭 `training.amp=false`。
- **类别数检查**：框架会在启动时提示 `head.num_classes` 与数据集 `num_classes` 是否一致，避免“标签越界/权重维度不匹配”。

## 验证/测试（成对验证）

对 `.bin` 成对数据进行评估：

```bash
python test.py \
  --config config/train_resnet50.yaml \
  --checkpoint checkpoints/resnet50/<RUN_TIMESTAMP>/best.pth \
  --name resnet50_adaface
```

输出：

- `logs/...` 下生成日志文件
- 同目录写入 `test_result_<name>_<timestamp>.txt` 汇总结果

## 推理（两图相似度）

使用 `engine/predictor.py` 计算两张图片 embedding 的余弦相似度：

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

## 上传 GitHub 注意事项

本项目已添加 `.gitignore` 来忽略：

- 训练日志、TensorBoard 事件文件
- checkpoint/权重文件（如 `.pth`）
- 缓存文件（`__pycache__` 等）

建议在 push 前执行一次 `git status`，确认不会误提交权重与日志。
