import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import path
from torch.optim import Adam

# MONAI's DiceLoss or your custom version
from monai.losses import DiceLoss
# from climatenet.modules import CGNetModule  # your CGNet model
from climatenet.utils.metrics import get_cm, get_iou_perClass
from climatenet.utils.utils import Config
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from climatenet.utils.utils import Config
from os import path
import pathlib
from monai.losses import HausdorffDTLoss, DiceLoss, GeneralizedDiceLoss, DiceFocalLoss
from monai.networks.utils import one_hot

# -------------------------
# 1) Surface/Boundary Loss
# -------------------------
class SurfaceLoss:
    """
    'BoundaryLoss' from your GitHub snippet.
    Multiplies predicted probabilities by signed distance maps.
    """
    def __init__(self, idc):
        """
        idc: list of class indices for which to compute boundary loss.
             e.g. [1, 2] to skip background=0 and focus on foreground classes.
        """
        self.idc = idc
        print(f"Initialized {self.__class__.__name__} with idc={idc}")

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        probs: (B, C, H, W) - predicted probabilities from softmax
        dist_maps: (B, C, H, W) - signed distance maps (neg inside, pos outside)
        """
        # Filter the relevant channels
        pc = probs[:, self.idc, ...]  # shape (B, len(idc), H, W)
        dc = dist_maps[:, self.idc, ...]
        multiplied = pc * dc  # elementwise
        return multiplied.mean()


BoundaryLoss = SurfaceLoss


# -------------------------
# 2) Example DiceFocalLoss
# -------------------------
class DiceFocalLoss(nn.Module):
    """
    Example combination: here we just wrap MONAI's DiceLoss for demonstration.
    If you truly have a FocalDice or something else, adapt accordingly.
    """
    def __init__(self, softmax=True, to_onehot_y=True, include_background=False, reduction='mean'):
        super().__init__()
        self.dice = DiceLoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            include_background=include_background,
            reduction=reduction
        )

    def forward(self, logits, labels):
        # For multi-class, 'labels' is integer, 'logits' is raw (B, C, H, W).
        # This calls the underlying MONAI DiceLoss
        return self.dice(logits, labels)


# -------------------------
# 3) CGNet High-Level Class
# -------------------------
class CGNet:
    """
    The high-level CGNet class. 
    This manages:
      - Model creation/loading
      - Training with a combined boundary + region loss
      - Evaluation
      - Prediction
    """
    def __init__(self, config: Config = None, model_path: str = None):
        # 1. Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Config or Model Path
        if config is not None and model_path is not None:
            raise ValueError("Cannot specify both config and model_path simultaneously.")

        if config is not None:
            # Create new model
            self.config = config
            self.network = CGNetModule(
                classes=len(self.config.labels), 
                channels=len(list(self.config.fields))
            ).to(self.device)
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = CGNetModule(
                classes=len(self.config.labels), 
                channels=len(list(self.config.fields))
            ).to(self.device)
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError("You must provide either a config or a model path.")

        # 3. Optimizer
        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)

    # -------------------------
    # 3a) TRAIN METHOD
    # -------------------------
    def train(self, dataset: ClimateDatasetLabeled):
        """
        Train using a combined region-based loss (DiceFocalLoss) + boundary-based loss (SurfaceLoss).
        
        We assume the dataset returns (features, labels, dist_map):
          - features: (B, V, H, W) float
          - labels:   (B, H, W) integer
          - dist_map: (B, C, H, W) with negative inside, positive outside
        """
        self.network.train()
        loader = DataLoader(
            dataset, 
            batch_size=self.config.train_batch_size,
            collate_fn=dataset.collate,
            num_workers=2,
            shuffle=True
        )
    
        # Region-based loss
        dice_loss_fn = DiceFocalLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            reduction="mean"
        )

        # Boundary loss: skip background => idc=[1,2] if 3 classes
        boundary_loss_fn = SurfaceLoss(idc=[1,2])
        
        alpha = 0.01  # weighting factor for boundary loss

        for epoch in range(1, self.config.epochs + 1):
            print(f"Epoch {epoch}:")
            epoch_loader = tqdm(loader)
            aggregate_cm = np.zeros((3, 3))  # 3 classes: BG=0, TC=1, AR=2 (example)
        
            for features_t, labels_t, dist_t in epoch_loader:
                # Move to device
                features_t = features_t.to(self.device)
                labels_t   = labels_t.to(self.device)
                dist_t     = dist_t.to(self.device)
                labels_t_unsq = labels_t.unsqueeze(1)  # => shape (B, 1, H, W)

                # Forward
                logits = self.network(features_t)  # shape (B, C, H, W)
                
                # 1) Region loss (DiceFocal)
                region_loss = dice_loss_fn(logits, labels_t_unsq)
                
                # 2) Boundary loss
                pred_probs = F.softmax(logits, dim=1)  # (B, C, H, W)
                boundary_loss = boundary_loss_fn(pred_probs, dist_t)

                total_loss = region_loss + alpha * boundary_loss

                # Update confusion matrix
                predictions = torch.argmax(logits, dim=1)  # (B, H, W)
                aggregate_cm += get_cm(predictions, labels_t, 3)

                epoch_loader.set_description(f"Loss: {total_loss.item():.4f}")
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # alpha = alpha + 0.01
        
            print("Epoch stats:")
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)
            print("IOUs:", ious, ", mean:", ious.mean())
            print(alpha)

    # -------------------------
    # 3b) PREDICT METHOD
    # -------------------------
    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        """
        Make predictions for the given dataset and return them as xr.DataArray
        """
        self.network.eval()
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size,
                            collate_fn=dataset.collate)
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values).to(self.device)
        
            with torch.no_grad():
                outputs = F.softmax(self.network(features), dim=1)  # (B, C, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            print(preds.shape)

            coords = batch.coords
            # remove 'variable' coordinate if it exists
            if 'variable' in coords:
                del coords['variable']
            
            dims = [dim for dim in batch.dims if dim != "variable"]
            predictions.append(
                xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs)
            )

        return xr.concat(predictions, dim='time')

    # -------------------------
    # 3c) EVALUATE METHOD
    # -------------------------
    def evaluate(self, dataset: ClimateDatasetLabeled):
        """
        Evaluate on a labeled dataset and return statistics.
        Summarizes confusion matrix and prints IOU metrics.
        """
        self.network.eval()
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size,
                            collate_fn=dataset.collate, num_workers=2)
        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3, 3))  # again, 3 classes

        for features, labels, _dist in epoch_loader:
            # we ignore dist_map here if returned
            features = features.to(self.device)
            labels = labels.to(self.device)
                
            with torch.no_grad():
                outputs = F.softmax(self.network(features), dim=1)
            predictions = torch.argmax(outputs, dim=1)
            aggregate_cm += get_cm(predictions, labels, 3)

        print("Evaluation stats:")
        print(aggregate_cm)
        ious = get_iou_perClass(aggregate_cm)
        print("IOUs:", ious, ", mean:", ious.mean())

    # -------------------------
    # 3d) SAVE & LOAD MODEL
    # -------------------------
    def save_model(self, save_path: str):
        """
        Save model weights and config to a directory.
        """
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        self.config.save(path.join(save_path, 'config_new.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights_new.pth'))

    def load_model(self, model_path: str):
        """
        Load a model. 
        """
        self.config = Config(path.join(model_path, 'config_new.json'))
        self.network = CGNetModule(
            classes=len(self.config.labels), 
            channels=len(list(self.config.fields))
        ).to(self.device)
        self.network.load_state_dict(
            torch.load(path.join(model_path, 'weights_new.pth'), map_location=self.device),
            strict=True
        )

class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """
    def __init__(self, classes=19, channels=4, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(channels, 32, 3, 2)      # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)                          
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)      

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injection, factor=4

        self.b1 = BNPReLU(32 + channels)
        
        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 + channels, 64, dilation_rate=2, reduction=8)  
        self.level2 = nn.ModuleList()
        for i in range(0, M-1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)
        
        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + channels, 128, dilation_rate=4, reduction=16) 
        self.level3 = nn.ModuleList()
        for i in range(0, N-1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)) # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have dropout layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        # Initialize weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        
        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # Upsample segmentation map to the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear', align_corners=False)
        return out
