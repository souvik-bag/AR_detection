import numpy as np
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config


import torch
import numpy as np
from functools import partial
from typing import Tuple, Callable
from torchvision import transforms

from scipy.ndimage import distance_transform_edt

def eucl_distance(binary_mask: np.ndarray, sampling=None):
    """
    Basic Euclidean distance transform. If 'sampling' is provided,
    you can scale distances accordingly. For simplicity, we ignore it.
    """
    return distance_transform_edt(binary_mask)

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, ...] = None,
                 dtype=None) -> np.ndarray:
    """
    Convert a one-hot encoded segmentation 'seg' into a signed distance transform per class.
      - seg shape: (K, ...) where K is the number of classes
      - inside object => negative distance
      - outside object => positive distance
      - boundary ~ 0
    """
    K = seg.shape[0]
    if dtype is None:
        dtype = np.float32
    res = np.zeros_like(seg, dtype=dtype)

    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            dist_out = eucl_distance(negmask, sampling=resolution)
            dist_in  = eucl_distance(posmask,  sampling=resolution)
            # Signed distance: outside is positive, inside is negative
            # shift boundary by 1 so boundary ~ 0
            res[k] = dist_out * negmask - (dist_in - 1) * posmask

    return res

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Example transform that:
      1) Takes an integer label Tensor (shape: e.g. (H, W)) 
      2) Converts it to one-hot shape (K, H, W).
    Adjust for multi-dimensional/time if needed.
    """
    def _transform(label_t: torch.Tensor) -> torch.Tensor:
        # label_t is shape (H, W) or possibly (time, lat, lon). 
        # We'll assume 2D for simplicity.
        # One-hot encode with K classes:
        #   shape -> (H, W) => one_hot => (H, W, K) => permute => (K, H, W)
        one_hot = torch.nn.functional.one_hot(label_t.long(), num_classes=K)
        # one_hot_2d = torch.nn.functional.one_hot(label_2d.long(), num_classes=K)  
        # one_hot_2d = one_hot_2d.permute(2, 0, 1).float() 
        one_hot = one_hot.permute(-1, 0, 1).float()  # => (K, H, W)
        return one_hot ## Change it to one_hot
    return _transform

def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Compose a transform pipeline that:
      1) one-hot encodes the label
      2) converts to numpy
      3) applies one_hot2dist
      4) converts back to torch float32
    """
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])



from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
import torch

# We assume you've defined:
#   dist_map_transform, gt_transform, one_hot2dist, etc. above
# from the snippet.

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 
    Uses config.fields for normalization (mean/std per variable).
    '''
    def __init__(self, data_path: str, config):
        self.path = data_path
        self.fields = config.fields
        self.num_files = sorted(f for f in listdir(self.path) if f.endswith('.nc'))
        self.length = len(self.num_files)

    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        # features shape: (variable, time, lat, lon)
        for var_name, stats in self.fields.items():
            var_data = features.sel(variable=var_name).values
            var_data -= stats['mean']
            var_data /= stats['std']

    def get_features(self, ds: xr.Dataset) -> xr.DataArray:
        # load the variables we want from config.fields
        vars_to_load = list(self.fields.keys())
        features = ds[vars_to_load].to_array(dim='variable')  # (variable, time, lat, lon)
        self.normalize(features)
        # reorder to (time, variable, lat, lon)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int) -> xr.DataArray:
        file_path = path.join(self.path, self.num_files[idx])
        ds = xr.load_dataset(file_path)
        return self.get_features(ds)

    @staticmethod
    def collate(batch):
        # e.g., concat along 'time'
        return xr.concat(batch, dim='time')

class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class for segmentation.
    Returns: (features, labels, dist_map).
    '''
    def __init__(self, data_path: str, config, resolution=(1.0, 1.0)):
        super().__init__(data_path, config)
        self.K = config.num_classes 
        # self.K = getattr(config, 'num_classes', num_classes)   # or read from config
        self.resolution = resolution
        # Build the transform pipeline for distance maps
        self.dist_transform = dist_map_transform(resolution=self.resolution, K=self.K)

    def __getitem__(self, idx: int):
        file_path = path.join(self.path, self.num_files[idx])
        ds = xr.load_dataset(file_path)

    # 1) get normalized features
        features = self.get_features(ds)  # xarray DataArray, shape (time, variable, lat, lon)
        features_np = features.values     # e.g. (T, V, H, W)
        features_t  = torch.tensor(features_np, dtype=torch.float32)

    # 2) get labels from ds['LABELS']
        labels_xr = ds['LABELS']  # shape might be (H, W) or (T, H, W)
        labels_t = torch.tensor(labels_xr.values, dtype=torch.long)
        # print("labels_t.shape:", labels_t.shape)

    # Decide if 2D or 3D
        if labels_t.dim() == 2:
        # shape (H, W)
        # Directly transform
            dist_map_2d = self.dist_transform(labels_t)  # => (K, H, W)
            dist_map_2d = dist_map_2d.unsqueeze(0)       # => (1, K, H, W)
            labels_2d   = labels_t.unsqueeze(0)          # => (1, H, W)

            dist_tensor   = dist_map_2d
            labels_tensor = labels_2d
        elif labels_t.dim() == 3:
        # shape (T, H, W)
        # Loop over time dimension
            dist_maps = []
            labels_list = []
            for t_idx in range(labels_t.shape[0]):
                label_2d = labels_t[t_idx]   # (H, W)
                dist_2d  = self.dist_transform(label_2d)  # => (K, H, W)
                dist_maps.append(dist_2d.unsqueeze(0))    # => (1, K, H, W)
                labels_list.append(label_2d.unsqueeze(0)) # => (1, H, W)

            dist_tensor   = torch.cat(dist_maps, dim=0)   # => (T, K, H, W)
            labels_tensor = torch.cat(labels_list, dim=0) # => (T, H, W)
        else:
            raise ValueError(f"Unexpected label shape: {labels_t.shape}")

        return features_t, labels_tensor, dist_tensor

    @staticmethod
    def collate(batch):
        # batch is list of (features_t, labels_t, dist_t)
        # we can either just stack them or handle xarray merges
        feats, labs, dists = map(list, zip(*batch))
        # each feats is shape (T, V, H, W)
        # We can cat along dimension 0 (time)
        feats_cat = torch.cat(feats, dim=0)
        labs_cat  = torch.cat(labs,  dim=0)
        dists_cat = torch.cat(dists, dim=0)
        return feats_cat, labs_cat, dists_cat
