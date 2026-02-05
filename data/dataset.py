# data/dataset.py
# 使用 MXNet 标准方法读取 InsightFace RecordIO 格式

import os
import io
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import warnings

# 尝试导入 MXNet
try:
    import mxnet as mx
    from mxnet import recordio
    HAS_MXNET = True
except ImportError:
    HAS_MXNET = False
    print("警告: 未安装 MXNet，将使用备用解析方法")
    print("推荐安装: pip install mxnet")


class FaceEmoreDataset(Dataset):
    """
    InsightFace MS1MV2 (faces_emore) 训练集
    使用 MXNet RecordIO 格式
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.max_retry = 10
        
        idx_path = os.path.join(root_dir, "train.idx")
        rec_path = os.path.join(root_dir, "train.rec")
        prop_path = os.path.join(root_dir, "property")
        
        for p in (idx_path, rec_path, prop_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少文件: {p}")

        # 读取 property
        with open(prop_path) as f:
            parts = f.read().strip().split(",")
            self.num_classes = int(parts[0])
            self.height = int(parts[1])
            self.width = int(parts[2])
        self.channels = 3

        # ✅ 使用 MXNet 标准方法读取
        if HAS_MXNET:
            self.imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
            # 读取第一个 record 获取总数
            s = self.imgrec.read_idx(0)
            header, _ = mx.recordio.unpack(s)
            self.num_samples = int(header.label[0])
        else:
            raise RuntimeError("必须安装 MXNet 才能读取 InsightFace 数据集: pip install mxnet")
        
        print(f"加载 faces_emore 数据集:")
        print(f"  {self.num_samples:,} 张图像")
        print(f"  {self.num_classes:,} 个类别")
        print(f"  {self.width}x{self.height}x{self.channels}")
        

    def __len__(self):
        return self.num_samples

    def _load_sample(self, idx):
        """
        使用 MXNet 标准方法加载样本
        Args:
            idx: 样本索引 (从 1 开始，0 是 header)
        """
        # 读取 record
        s = self.imgrec.read_idx(idx)
        header, img_bytes = mx.recordio.unpack(s)
        
        # 提取标签（可能是多个值，取第一个或求和）
        if isinstance(header.label, (list, tuple, np.ndarray)):
            if len(header.label) > 1:
                # 多标签情况（如 InsightFace 的格式）
                label = int(header.label[0])
            else:
                label = int(header.label[0])
        else:
            label = int(header.label)
        
        # 验证标签范围
        if label < 0 or label >= self.num_classes:
            raise ValueError(f"标签超出范围: {label} (有效范围: 0-{self.num_classes-1})")
        
        # 解码图像
        img = mx.image.imdecode(img_bytes).asnumpy()
        
        # 转换为 PIL Image (RGB格式)
        if img.ndim == 3:
            # MXNet 返回的是 RGB 格式
            img = Image.fromarray(img.astype(np.uint8))
        else:
            raise ValueError(f"图像维度错误: {img.shape}")
        
        return img, label

    def __getitem__(self, idx):
        """
        返回: (image_tensor, label)
        """
        # MXNet 索引从 1 开始（0 是 header record）
        mx_idx = idx + 1
        
        try:
            img, label = self._load_sample(mx_idx)
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        
        except Exception as e:
            warnings.warn(f"样本 {mx_idx} 加载失败: {e}", RuntimeWarning)
            for step in range(1, self.max_retry + 1):
                next_idx = (idx + step) % len(self)
                try:
                    img, label = self._load_sample(next_idx + 1)
                    if self.transform:
                        img = self.transform(img)
                    return img, label
                except Exception as inner_e:
                    warnings.warn(f"回退样本加载失败: {inner_e}", RuntimeWarning)
            raise


class BinPairDataset(Dataset):
    """.bin 格式验证集（LFW、CFP-FP、AgeDB-30）"""

    def __init__(self, root_dir, bin_name="lfw.bin", transform=None):
        self.transform = transform
        self.bin_name = bin_name
        bin_path = os.path.join(root_dir, bin_name)
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"未找到文件: {bin_path}")

        print(f"加载 {bin_name}...", end=" ")
        
        with open(bin_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        
        if not (isinstance(data, tuple) and len(data) == 2):
            raise ValueError(f"{bin_name} 格式错误")

        images_raw, labels_raw = data
        self.issame = np.array([bool(x) for x in labels_raw], dtype=bool)
        self.images_raw = list(images_raw)
        
        if len(self.images_raw) != len(self.issame) * 2:
            raise ValueError(f"数据不一致: {len(self.images_raw)} 图像, {len(self.issame)} 标签")
        
        pos_ratio = self.issame.sum() / len(self.issame) * 100
        print(f"{len(self.issame):,} 对 (正样本 {pos_ratio:.1f}%)")

    def __len__(self):
        return len(self.issame)

    def _load_image(self, img_data):
        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, np.ndarray):
            if img_data.ndim == 2 and img_data.shape[1] == 1:
                jpeg_bytes = img_data.flatten().tobytes()
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            elif img_data.ndim == 3:
                if img_data.dtype != np.uint8:
                    img_data = img_data.astype(np.uint8)
                img = Image.fromarray(img_data)
            else:
                raise ValueError(f"不支持的 ndarray shape: {img_data.shape}")
        else:
            raise ValueError(f"不支持的图像格式: {type(img_data)}")
        return img

    def __getitem__(self, idx):
        img1_data = self.images_raw[idx * 2]
        img2_data = self.images_raw[idx * 2 + 1]
        issame = bool(self.issame[idx])
        
        img1 = self._load_image(img1_data)
        img2 = self._load_image(img2_data)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, issame
