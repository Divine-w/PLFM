import torch
import torch.nn.functional as F
import numpy as np


markedfeats = torch.load('marked_feats.pt')
splits = ["train", "val", "test"]

for split in splits:
    feats = torch.load(f'{split}_feats.pt')
    # 计算余弦相似度
    similarities = F.cosine_similarity(feats.unsqueeze(1), markedfeats.unsqueeze(0), dim=2)

    # 获取每个样本的前三个最相似的人工标记样本
    top_k_indices = torch.topk(similarities, 5, dim=1).indices
    print(top_k_indices.size())
    print(top_k_indices)
    np.save(f'{split}_match_indices', top_k_indices)