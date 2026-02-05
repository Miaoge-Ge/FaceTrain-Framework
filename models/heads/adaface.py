import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaFace(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.4, h=0.333):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.h = h
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        embeddings_norm = embeddings.norm(p=2, dim=1, keepdim=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Ensure fp32 for matrix multiplication to avoid overflow
        embeddings = embeddings.float()
        weight = weight.float()

        cosine = F.linear(embeddings, weight)
        
        if labels is None:
            return cosine * self.s
        
        safe_norms = torch.clip(embeddings_norm, min=1e-6)
        ada_margin = self.m * torch.tanh(safe_norms - self.h) + self.m
        one_hot = F.one_hot(labels, self.num_classes).float()
        logits = (cosine - one_hot * ada_margin) * self.s
        return logits
