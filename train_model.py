import argparse, math, re, glob, os
from pathlib import Path
from typing   import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import xarray as xr
from tqdm import tqdm
from unet_block import UNet   
from loss import jaccard_loss
from procrustes_loss_new import ProcrustesLossBag

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


# ───────────────────────────────── MODEL DEFINITIONS ─────────────────────────────────
# class DoubleConv(nn.Module):
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(c_in, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.ReLU(True),
#             nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.ReLU(True),
#         )
#     def forward(self, x): return self.net(x)

# def down(c): return nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2))

# class VarUNet(nn.Module):                         # ← standard UNet encoder-decoder
#     def __init__(self, c_in, base=48):
#         super().__init__()
#         self.inc   = DoubleConv(c_in, base)
#         self.down1 = down(base)
#         self.down2 = down(base*2)
#         self.down3 = down(base*4)
#         self.up3   = nn.ConvTranspose2d(base*8, base*4, 2, 2)
#         self.conv3 = DoubleConv(base*8, base*4)
#         self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, 2)
#         self.conv2 = DoubleConv(base*4, base*2)
#         self.up1   = nn.ConvTranspose2d(base*2, base,   2, 2)
#         self.conv1 = DoubleConv(base*2, base)
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x  = self.conv3(torch.cat([self.up3(x4), x3], 1))
#         x  = self.conv2(torch.cat([self.up2(x),  x2], 1))
#         x  = self.conv1(torch.cat([self.up1(x),  x1], 1))
#         return x                              # (B, base, H, W)

class AnnEncoder(nn.Module):
    """shared 2-layer CNN → mean-pooled feature map, returns None for empty list."""
    def __init__(self, c_out=24):
        super().__init__()
        self.c_out = c_out
        self.enc = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(12, c_out, 3, padding=1), nn.ReLU(True)
        )

    def forward(self, y_list):
        if len(y_list) == 0:
            return None                          # caller handles None
        feats = [self.enc(y.unsqueeze(0)) for y in y_list]  # [N,c_out,H,W]
        return torch.stack(feats).mean(0)                   # [1,c_out,H,W]

class FuseNet(nn.Module):
    """Learns soft composite mask from annotator list only (no TMQ)."""
    def __init__(self, c_ann=24):
        super().__init__()
        self.ann_enc = AnnEncoder(c_ann)
        self.head    = nn.Conv2d(c_ann, 1, 1)              # logits

    def forward(self, y_list):
        feat = self.ann_enc(y_list)                        # None or (1,c_ann,H,W)
        if feat is None:
            raise ValueError("FuseNet received empty annotator list.")
        return self.head(feat)                             # (1,1,H,W) logits

# class SegNet(nn.Module):
#     """Plain U-Net that sees TMQ only and predicts AR mask."""
#     def __init__(self):
#         super().__init__()
#         self.unet = VarUNet(c_in=1, base=48)
#         self.head = nn.Conv2d(48, 1, 1)

#     def forward(self, x):                                  # (B,1,H,W)
#         return self.head(self.unet(x))                     # logits
    
class SegNet(nn.Module):
    """
    TMQ-only U-Net that predicts a single-channel AR probability map.
    """
    def __init__(self):
        super().__init__()
        self.net = UNet(n_channels=1, n_classes=1, bilinear=False)

    def forward(self, x):                         # x: [B,1,H,W]
        return self.net(x)                        # logits [B,1,H,W]

# # ─────────────────────────────────── MAIN TRAINING ───────────────────────────────────

# def main(cfg):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     ds      = ARMultiAnnDataset(cfg.data_dir, ["TMQ"])
#     loader  = DataLoader(ds, batch_size=cfg.bs, shuffle=True,
#                          num_workers=cfg.workers, collate_fn=collate_var,
#                          pin_memory=True)

#     # ---------------- Phase-1 : train FuseNet on annotator masks ------------------
#     fuse   = FuseNet().to(device)
#     opt_f  = torch.optim.AdamW(fuse.parameters(), lr=1e-3)
#     bce    = nn.BCEWithLogitsLoss()

#     for epoch in range(10):                                 # short pre-train
#         fuse.train()
#         for _, y_lists, _ in loader:
#             f_loss = 0.0
#             for lst in y_lists:                            # loop within batch
#                 logit = fuse([m.to(device) for m in lst])  # (1,1,H,W)
#                 target= torch.stack(lst).mean(0, keepdim=True).to(device)
#                 f_loss += bce(logit, target)
#             opt_f.zero_grad(); f_loss.backward(); opt_f.step()
#         print(f"FuseNet epoch {epoch}: loss {f_loss.item():.4f}")

#     # freeze FuseNet
#     fuse.eval()
#     for p in fuse.parameters():
#         p.requires_grad_(False)

#     # ---------------- Phase-2 : train SegNet to mimic FuseNet map ------------------
#     seg   = SegNet().to(device)
#     opt_s = torch.optim.AdamW(seg.parameters(), lr=1e-4)
#     scaler= torch.amp.GradScaler(device="cuda")

#     for epoch in range(cfg.epochs):
#         seg.train()
#         epoch_loss = 0.0
#         for x_vars, y_lists, _ in loader:
#             x_vars = x_vars.to(device)

#             # composite map (no grad)
#             with torch.no_grad():
#                 comp = torch.cat([ torch.sigmoid(fuse([m.to(device) for m in lst]))
#                                    for lst in y_lists ], dim=0)   # (B,1,H,W)

#             with torch.amp.autocast(device_type="cuda"):
#                 logits = seg(x_vars)                     # TMQ-only
#                 loss   = bce(logits, comp) + 0.5*(1 - dice_coeff(logits, comp))

#             scaler.scale(loss).backward()
#             scaler.step(opt_s); scaler.update(); opt_s.zero_grad()
#             epoch_loss += loss.item() * x_vars.size(0)

#         print(f"SegNet epoch {epoch:02d}  mean loss {epoch_loss/len(ds):.4f}")

#     # ------------- save final SegNet checkpoint -----------------------------------
#     torch.save(seg.state_dict(), Path(cfg.out) / "segnet_final.pt")

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── dataset + 80/20 split (keeps y_cons) ───────────────────────────────────
    full_ds  = ARMultiAnnDataset(cfg.data_dir, ["TMQ"])
    val_frac = 0.20
    n_val    = int(len(full_ds) * val_frac)
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True,
                          num_workers=cfg.workers, collate_fn=collate_var,
                          pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.bs, shuffle=False,
                          num_workers=cfg.workers, collate_fn=collate_var,
                          pin_memory=True)

    # ── model, loss, optimiser ────────────────────────────────────────────────
    seg    = SegNet().to(device)                       # your UNet wrapper
    opt    = torch.optim.AdamW(seg.parameters(), lr=1e-4)
    bce    = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device ="cuda")
    criterion = ProcrustesLossBag(alpha=2.0)

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        # ---- training ----
        seg.train(); tr_loss = 0.0
        for x_vars, _, y_cons in tqdm(train_dl, desc=f"Train {epoch:02d}", ncols=100):
            x_vars, y_cons = x_vars.to(device), y_cons.to(device)

            with torch.amp.autocast(device_type="cuda"):
                logits = seg(x_vars)
                # loss   = bce(logits, y_cons) + 0.5*(1 - dice_coeff(logits, y_cons))
                loss_jaccard = jaccard_loss(logits, y_cons)
                pred_probs = torch.sigmoid(logits) 
                # print(f"pred_probs shape : {pred_probs.shape}")
                # print(f"y_con shape : {y_cons.shape}")
                loss_procrustes = criterion(pred_probs, y_cons)
                loss = loss_jaccard + (0.1*loss_procrustes)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            tr_loss += loss.item() * x_vars.size(0)

        # ---- validation ----
        seg.eval(); val_loss = val_dice = 0.0
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            for x_vars, _, y_cons in val_dl:
                x_vars, y_cons = x_vars.to(device), y_cons.to(device)
                logits = seg(x_vars)
                pred_probs = torch.sigmoid(logits) 
                # loss   = bce(logits, y_cons) + 0.5*(1 - dice_coeff(logits, y_cons))
                loss_jaccard = jaccard_loss(logits, y_cons)
                loss_procrustes = criterion(pred_probs, y_cons)
                loss = loss_jaccard + (0.1*loss_procrustes)
                
                
                val_loss += loss.item() * x_vars.size(0)
                val_dice += dice_coeff(logits, y_cons) * x_vars.size(0)

        # ---- epoch metrics ----
        tr_loss /= n_train
        val_loss /= n_val
        val_dice /= n_val
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} |"
              f" val {val_loss:.4f} | val-Dice {val_dice:.4f}")

    #── save checkpoint ───────────────────────────────────────────────────────
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(seg.state_dict(), out_dir / "unet_Jaccard_PL.pt")
    

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

