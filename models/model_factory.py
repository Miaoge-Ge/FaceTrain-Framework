import torch.nn as nn
from models.backbones.resnet import ResNet50
from models.backbones.fastcontextface import fastcontextface
from models.heads.arcface import ArcFace
from models.heads.cosface import CosFace
from models.heads.adaface import AdaFace

class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is None:
            return embeddings
        logits = self.head(embeddings, labels)
        return logits, embeddings

def build_model(config):
    model_cfg = config['model']
    head_cfg = config['head']
    
    backbone_type = model_cfg['backbone']
    embedding_size = model_cfg['embedding_size']
    
    backbone_map = {
        'resnet50': ResNet50,
        'fastcontextface': fastcontextface,
    }
    if backbone_type not in backbone_map:
        raise ValueError(f"Unsupported backbone: {backbone_type}. Supported: {sorted(backbone_map.keys())}")
    backbone = backbone_map[backbone_type](embedding_size)
    
    head_type = head_cfg['type']
    num_classes = head_cfg['num_classes']
    s = head_cfg.get('scale', 64.0)
    m = head_cfg.get('margin', 0.5 if head_type == 'arcface' else 0.35)
    
    head_map = {
        'arcface': ArcFace,
        'cosface': CosFace,
        'adaface': AdaFace,
    }
    if head_type not in head_map:
        raise ValueError(f"Unsupported head: {head_type}. Supported: {sorted(head_map.keys())}")
    
    head = head_map[head_type](
        embedding_size=embedding_size,
        num_classes=num_classes,
        s=s,
        m=m
    )
    
    return FaceRecognitionModel(backbone, head)
