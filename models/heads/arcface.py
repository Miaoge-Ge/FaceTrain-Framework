import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight, gain=0.1)

        # 预计算常量，用于稳定计算 cos(theta + m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # 阈值，用于处理 theta + m > pi 的情况
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)
        cosine = F.linear(embeddings, weight)
        
        if labels is None:
            return cosine * self.s
            
        # 1. 计算 cos(theta + m)
        # 使用公式: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 2. 稳定性处理 (Critical Fix)
        # 原版简单的 cos(acos(x) + m) 在 theta + m > pi 时会不再单调，导致 Loss 无法下降。
        # 这里使用泰勒展开近似来处理大角度情况，保证梯度方向正确。
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 3. 生成 Target Logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # 4. 数值稳定性检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=self.s, neginf=-self.s)
        
        return output
