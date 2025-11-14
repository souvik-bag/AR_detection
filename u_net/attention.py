import torch
import torch.nn as nn
import torch.nn.functional as F

class AnnotatorDualAttention(nn.Module):
    """
    1) SE over annotator dimension → per-expert weights
    2) fuse weighted masks → single-channel map
    3) CBAM‐style spatial attention on the fused map
    """
    def __init__(self, n_annotators: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        # — channel/expert SE —
        self.fc1 = nn.Linear(n_annotators, n_annotators // reduction, bias=False)
        self.fc2 = nn.Linear(n_annotators // reduction, n_annotators, bias=False)
        # — spatial attention (CBAM) —
        assert kernel_size in (3, 7), "Use 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, y_stack: torch.Tensor):
        """
        Args:
          y_stack: [B, N, H, W]  (N annotator binary/soft masks)
        Returns:
          attn:    [B, 1, H, W]  final attention map
          weights: [B, N]        per-expert reliability
        """
        B, N, H, W = y_stack.shape

        # — 1) Squeeze & excite to get per-expert weights —
        summary = y_stack.view(B, N, -1).mean(-1)         # [B,N]
        w = F.relu(self.fc1(summary))                    # [B,N//r]
        w = torch.sigmoid(self.fc2(w))                    # [B,N]

        # — 2) fuse weighted annotations —
        w_map = w.view(B, N, 1, 1)                        # [B,N,1,1]
        fused = (y_stack * w_map).sum(1, keepdim=True)    # [B,1,H,W]

        # — 3) spatial attention (CBAM) —
        #    use average+max pooling across channel=1 dimension
        avg_pool = fused.mean(1, keepdim=True)            # [B,1,H,W]
        max_pool, _ = fused.max(1, keepdim=True)          # [B,1,H,W]
        cat = torch.cat([avg_pool, max_pool], dim=1)      # [B,2,H,W]
        spat_attn = torch.sigmoid(self.conv_spatial(cat)) # [B,1,H,W]

        # — final map: modulate fused by spatial mask —
        attn_map = fused * spat_attn                     # [B,1,H,W]

        return attn_map, w
