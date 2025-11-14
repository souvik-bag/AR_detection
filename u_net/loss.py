import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F

# ─── add this near your imports ───────────────────────────────────────────
def jaccard_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """Soft Jaccard / IoU loss for logits ∈ ℝ and targets ∈ [0,1]."""
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))          # [B,1]
    union = (probs + targets - probs * targets).sum(dim=(2, 3))
    iou   = (inter + eps) / (union + eps)              # [B,1]
    return 1.0 - iou.mean()                            # scalar