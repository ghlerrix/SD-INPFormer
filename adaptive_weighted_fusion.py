import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def fuse_feature(feat_list, weights=None):

    feat_tensor = torch.stack(feat_list, dim=1)
    
    if weights is None:
        return torch.mean(feat_tensor, dim=1)
    else:
        norm_weights = F.softmax(weights, dim=0).view(1, -1, 1, 1)
        weighted_feat = torch.sum(feat_tensor * norm_weights, dim=1)
        return weighted_feat