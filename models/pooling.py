import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer to aggregate features from multiple images.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        
    def forward(self, x, mask=None):
        # x: [B, N, D], mask: [B, N]
        scores = self.attn(x).squeeze(-1)  # [B, N]
        
        if mask is not None:
            # Create a boolean mask (True for valid images, False for padding)
            mask_bool = mask.bool()
            # Set scores for padding to -infinity
            scores = scores.masked_fill(~mask_bool, -1e9)
            
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, N, 1]
        weighted = (weights * x).sum(dim=1)  # [B, D]
        return weighted

class MeanPooling(nn.Module):
    """
    Mean pooling layer with mask support to aggregate features from multiple images.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask=None):
        # x: [B, N, D], mask: [B, N]
        if mask is not None:
            # Expand mask to feature dimension
            mask = mask.float().unsqueeze(-1)  # [B, N, 1]
            # Compute masked mean (sum / count)
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, D]
        else:
            return x.mean(dim=1)  # [B, D]