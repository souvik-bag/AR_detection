# train_ar_unet.py
# ============================================================
"""
End-to-end training of a U-Net–based AR segmenter that fuses
  • K gridded variables (IVT, IWV, wind …)
  • N_i annotator masks (N_i varies per date)
and predicts the *soft consensus* mask.

Requires:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install xarray netcdf4 tqdm
"""

import argparse, math, re, glob, os
from pathlib import Path
from typing   import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr
from tqdm import tqdm

# -------------------------------------------------------------------------
# 0. utilities
# -------------------------------------------------------------------------

def dice_coeff(logits: torch.Tensor, targets: torch.Tensor, eps=1e-6) -> float:
    """Soft-Dice over the batch."""
    probs = torch.sigmoid(logits)
    num   = (probs * targets).sum((2,3))
    den   = probs.sum((2,3)) + targets.sum((2,3))
    dice  = (2*num + eps) / (den + eps)
    return dice.mean().item()

# -------------------------------------------------------------------------
# 1. dataset  (exactly the class you already have, shortened here)
# -------------------------------------------------------------------------

FNAME_RE = re.compile(r"data-(\d{4}-\d{2}-\d{2})-.*?_(\d+)\.nc$")

class ARMultiAnnDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, var_names: List[str]):
        super().__init__()
        self.root, self.var_names = Path(root), var_names

        by_date = {}
        for f in glob.glob(str(self.root / "*.nc")):
            m = FNAME_RE.search(Path(f).name)
            if m:
                by_date.setdefault(m.group(1), {})[int(m.group(2))] = Path(f)

        self.samples = sorted(by_date.items())
        self.real_max_ann = max(len(d) for _,d in self.samples)

    def _load_nc(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = xr.open_dataset(path, engine="netcdf4")

        # squeeze() drops ANY singleton dimension, e.g. the leading "time=1"
        vars_arr = [
            torch.as_tensor(ds[v].values).squeeze()     # (H, W) after squeeze
            for v in self.var_names
        ]
        x = torch.stack(vars_arr).float()               # [C, H, W]
        
        raw = torch.as_tensor(ds["LABELS"].values).squeeze()
        y   = (raw == 2).float().unsqueeze(0) 

        # y = torch.as_tensor(ds["LABELS"].values).squeeze().unsqueeze(0).float()
        ds.close()
        return x, y

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        _, ann_map = self.samples[idx]
        x_vars, y_list = None, []
        for p in ann_map.values():
            x, y = self._load_nc(p)
            if x_vars is None: x_vars = x
            y_list.append(y)
        y_cons = torch.stack(y_list).mean(0)
        return x_vars, y_list, y_cons      # variable-length list

def collate_var(batch):
    xs, y_lists, y_cons = zip(*batch)
    return torch.stack(xs), list(y_lists), torch.stack(y_cons)

# -------------------------------------------------------------------------
# 2. model
# -------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

def down(c): return nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2))

class VarUNet(nn.Module):                         # ← standard UNet encoder-decoder
    def __init__(self, c_in, base=48):
        super().__init__()
        self.inc   = DoubleConv(c_in, base)
        self.down1 = down(base)
        self.down2 = down(base*2)
        self.down3 = down(base*4)
        self.up3   = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1   = nn.ConvTranspose2d(base*2, base,   2, 2)
        self.conv1 = DoubleConv(base*2, base)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.conv3(torch.cat([self.up3(x4), x3], 1))
        x  = self.conv2(torch.cat([self.up2(x),  x2], 1))
        x  = self.conv1(torch.cat([self.up1(x),  x1], 1))
        return x                              # (B, base, H, W)
# Original
# class AnnEncoder(nn.Module):
#     """Shared CNN → mean-pool over variable annotators."""
#     def __init__(self, c_out=24):
#         super().__init__()
#         self.enc = nn.Sequential(
#             nn.Conv2d(1, 12, 3, 1, 1), nn.ReLU(True),
#             nn.Conv2d(12, c_out, 3, 1, 1), nn.ReLU(True)
#         )
#     def forward(self, y_list: List[torch.Tensor]):   # len = N_i
#         feats = [self.enc(y.unsqueeze(0)) for y in y_list]   # N, c_out, H, W
#         return torch.stack(feats).mean(0)                   # 1, c_out, H, W

class AnnEncoder(nn.Module):
    def __init__(self, c_out=24):
        super().__init__()
        self.c_out = c_out
        self.enc = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(12, c_out, 3, padding=1), nn.ReLU(True)
        )

    def forward(self, y_list):
        if len(y_list) == 0:                             # ← ADD THIS BLOCK
            # return a tensor of zeros shaped like a single feature map
            # (1, c_out, H, W) – but H,W unknown here, so create a dummy
            return None                                  # signal “empty”
        feats = [self.enc(y.unsqueeze(0)) for y in y_list]
        return torch.stack(feats).mean(0)                # (1, c_out, H, W)

# class ARSegNet(nn.Module):
#     def __init__(self, c_vars: int, base=48, c_ann=24):
#         super().__init__()
#         self.var_unet = VarUNet(c_vars, base)
#         self.ann_enc  = AnnEncoder(c_ann)
#         self.head     = nn.Sequential(
#             nn.Conv2d(base + c_ann, base, 3, 1, 1), nn.ReLU(True),
#             nn.Conv2d(base, 1, 1)                        # logits
#         )
#     def forward(self, x_vars, y_lists):
#         B = x_vars.size(0)
#         f_var = self.var_unet(x_vars)                    # [B,base,H,W]
#         f_ann = torch.cat([self.ann_enc(lst) for lst in y_lists], 0)  # [B,c_ann,H,W]
#         return self.head(torch.cat([f_var, f_ann], 1))   # [B,1,H,W]

class ARSegNet(nn.Module):
    def __init__(self, c_vars: int, base=48, c_ann=24):
        super().__init__()
        self.var_unet = VarUNet(c_vars, base)
        self.ann_enc  = AnnEncoder(c_ann)
        self.head     = nn.Sequential(
            nn.Conv2d(base + c_ann, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, 1, 1)                      # logits
        )

    def forward(self, x_vars, y_lists):
        """
        x_vars  : Tensor  [B, c_vars, H, W]
        y_lists : list length B, each element is a list of Nᵢ masks (may be 0)
        """
        B, _, H, W = x_vars.shape
        f_var = self.var_unet(x_vars)                  # [B, base, H, W]

        f_ann = []                                     # will collect B tensors
        for lst in y_lists:
            feat = self.ann_enc(lst)                   # None or [1, c_ann, H, W]
            if feat is None:                           # ← empty annotator list
                feat = x_vars.new_zeros(1,             # create zeros on-device,
                                        self.ann_enc.c_out,
                                        H, W)          # same dtype, same H,W
            f_ann.append(feat)

        f_ann = torch.cat(f_ann, dim=0)                # [B, c_ann, H, W]
        fused = torch.cat([f_var, f_ann], dim=1)       # [B, base+c_ann, H, W]
        return self.head(fused)                        # [B, 1, H, W]  logits

# -------------------------------------------------------------------------
# 3. training
# -------------------------------------------------------------------------

# def main(cfg):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     ds   = ARMultiAnnDataset(cfg.data_dir, cfg.vars)
#     loader = DataLoader(ds, batch_size=cfg.bs, shuffle=True,
#                         num_workers=cfg.workers, collate_fn=collate_var,
#                         pin_memory=True)
#     net  = ARSegNet(len(cfg.vars)).to(device)
#     opt  = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
#     sched= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

#     bce  = nn.BCEWithLogitsLoss()
#     scaler = torch.cuda.amp.GradScaler()

#     logdir = Path(cfg.out)
#     logdir.mkdir(parents=True, exist_ok=True)

#     for epoch in range(cfg.epochs):
#         net.train()
#         epoch_loss, epoch_dice = 0.0, 0.0
#         pbar = tqdm(loader, desc=f"Epoch {epoch:02d}", ncols=100)
#         for x_vars, y_lists, y_cons in pbar:
#             x_vars, y_cons = x_vars.to(device), y_cons.to(device)

#             with torch.cuda.amp.autocast():
#                 logits = net(x_vars, y_lists)
#                 loss   = bce(logits, y_cons) + 0.5 * (1 - dice_coeff(logits, y_cons))
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#             scaler.step(opt); scaler.update(); opt.zero_grad()

#             with torch.no_grad():
#                 d = dice_coeff(logits, y_cons)
#             epoch_loss += loss.item()*x_vars.size(0)
#             epoch_dice += d*x_vars.size(0)
#             pbar.set_postfix(loss=f"{loss.item():.3f}", dice=f"{d:.3f}")

#         sched.step()
#         N = len(ds)
#         print(f"Epoch {epoch:02d}  mean loss {epoch_loss/N:.4f} | mean Dice {epoch_dice/N:.4f}")

#         # checkpoint
#         torch.save({"model": net.state_dict(),
#                     "opt": opt.state_dict(),
#                     "epoch": epoch},
#                    logdir / f"ckpt_{epoch:02d}.pt")

# -------------------------------------------------------------------------
# 3. training  – with dtype-safe AMP
# -------------------------------------------------------------------------
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds      = ARMultiAnnDataset(cfg.data_dir, cfg.vars)
    loader  = DataLoader(ds, batch_size=cfg.bs, shuffle=True,
                         num_workers=cfg.workers, collate_fn=collate_var,
                         pin_memory=True)

    net     = ARSegNet(len(cfg.vars)).to(device)
    opt     = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    bce     = nn.BCEWithLogitsLoss()

    scaler  = torch.amp.GradScaler(device="cuda")        # new API
    logdir = Path(cfg.out)
    logdir.mkdir(parents=True, exist_ok=True)
    MASK_DROP_P = 0.5 


    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = epoch_dice = 0.0
        for x_vars, y_lists, y_cons in tqdm(loader, desc=f"Epoch {epoch:02d}", ncols=100):
            x_vars, y_cons = x_vars.to(device), y_cons.to(device)
            if torch.rand(1).item() < MASK_DROP_P:
                y_lists = [[] for _ in y_lists]     # <- makes annotator branch get zeros

            with torch.amp.autocast(device_type="cuda"):
                # ---- annotate branch (cast masks) ----
                ann_feats = []
                for y_list in y_lists:
                    y_gpu = [m.to(device, dtype=x_vars.dtype) for m in y_list]
                    ann_feats.append(net.ann_enc(y_gpu))      # 1,c_ann,H,W
                ann_feats = torch.cat(ann_feats, 0)           # B,c_ann,H,W

                var_feats = net.var_unet(x_vars)
                logits    = net.head(torch.cat([var_feats, ann_feats], 1))
                loss      = bce(logits, y_cons) + 0.5 * (1 - dice_coeff(logits, y_cons))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()

            with torch.no_grad():
                d = dice_coeff(logits, y_cons)

            epoch_loss += loss.item() * x_vars.size(0)
            epoch_dice += d * x_vars.size(0)

        sched.step()
        N = len(ds)
        print(f"Epoch {epoch:02d}  mean loss {epoch_loss/N:.4f} | mean Dice {epoch_dice/N:.4f}")
        
        # checkpoint
        torch.save({"model": net.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch},
                   logdir / f"ckpt_{epoch:02d}.pt")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vars", nargs="+", required=True,
                        help="16 variable names present in every .nc file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs",     type=int, default=2)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--workers",type=int, default=4)
    parser.add_argument("--out",    default="runs/arseg")
    cfg = parser.parse_args()
    main(cfg)
