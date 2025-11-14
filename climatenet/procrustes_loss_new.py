from __future__ import annotations
import warnings
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss
import multiprocessing
from monai.networks import one_hot
from monai.utils import LossReduction
import os
import numpy as np
import skimage.measure
import skimage.draw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt
def label_and_extract_boundaries(binary_mask):
    """
    Given a 2D binary mask, label connected components and
    extract the boundary points for each labeled object.
    
    Returns a list of (label_id, boundary_points), 
    where boundary_points is an (N,2) array of [row,col].
    """
    labeled = skimage.measure.label(binary_mask, connectivity=2)
    labeled = labeled.squeeze()
    print(f'labeled shape : {labeled.shape}')
    boundaries = []
    for region_id in range(1, labeled.max() + 1):
        # A mask for just this object
        obj_mask = (labeled == region_id)
        # Find contours returns list of arrays (N,2) in y,x
        contour_list = skimage.measure.find_contours(obj_mask, level=0.5)
        # For simplicity, pick the largest contour
        if len(contour_list) > 0:
            largest_contour = max(contour_list, key=len)
            boundaries.append((region_id, largest_contour)) 
    return boundaries

def label_and_extract_boundaries_batch(
        masks: np.ndarray
) -> List[List[Tuple[int, np.ndarray]]]:
    """
    Parameters
    ----------
    masks : np.ndarray
        Binary mask with shape
            (B, H, W)      ––or––     (B, 1, H, W)

    Returns
    -------
    List[List[(label_id, boundary_pts)]]
        Outer list length = B (images);
        inner list = result of label_and_extract_boundaries per image.
    """
    if masks.ndim == 4:                    # (B,1,H,W)  → squeeze channel
        masks = masks[:, 0]

    assert masks.ndim == 3, "Expect (B,H,W) after squeeze"

    batch_boundaries = []
    for b in range(masks.shape[0]):
        boundaries = label_and_extract_boundaries(masks[b])
        batch_boundaries.append(boundaries)

    return batch_boundaries


############################################
# 2. Simple Cost for GT-Pred Object Matching
############################################

def compute_centroid(contour):
    """
    Return the centroid (mean row,col) of the boundary points.
    """
    return contour.mean(axis=0)  # shape (2,)

def match_objects(gt_boundaries, pred_boundaries):
    """
    Assign each GT object to a predicted object 
    by building a cost matrix of centroid distances
    and using Hungarian assignment.
    
    gt_boundaries: list of (gt_id, gt_points)
    pred_boundaries: list of (pred_id, pred_points)
    
    Returns list of (gt_id, pred_id, cost).
    """
    n_gt = len(gt_boundaries)
    n_pred = len(pred_boundaries)
    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
    
    for i, (gt_id, gt_pts) in enumerate(gt_boundaries):
        c_gt = compute_centroid(gt_pts)
        for j, (pr_id, pr_pts) in enumerate(pred_boundaries):
            c_pr = compute_centroid(pr_pts)
            dist = np.linalg.norm(c_gt - c_pr)  # Euclidian distance
            cost_matrix[i, j] = dist
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignments = []
    for i, j in zip(row_ind, col_ind):
        gt_id, gt_pts = gt_boundaries[i]
        pr_id, pr_pts = pred_boundaries[j]
        cost = cost_matrix[i, j]
        assignments.append((gt_id, pr_id, cost))
    
    return assignments

############################################
# 3. Procrustes allignment
############################################


def procrustes_align(X, Y, allow_scaling=True):
    """
    Align Y to X via Procrustes. 
    X, Y: shape (N,2) arrays of boundary points [row,col].
    Returns aligned_Y of shape (N,2), plus transform dict.
    """
    # Convert to float
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Centering (Optional: if you want pure rotation + scaling)
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY
    
    # SVD to find best rotation
    A = Y0.T @ X0
    U, s, Vt = np.linalg.svd(A)
    R = (U @ Vt).T  # shape (2,2)
    
    # Optional scaling
    if allow_scaling:
        scale = s.sum() / (np.sum(Y0 ** 2) + 1e-8)
    else:
        scale = 1.0
    
    # Apply rotation and scaling
    aligned_Y = (scale * (Y0 @ R)) + muX
    
    # Compute Procrustes distance (L2 norm / Frobenius norm)
    distance = np.linalg.norm(X - aligned_Y)
    
    return X, aligned_Y, distance


def fill_polygon(boundary_pts, shape):
    """
    Fill the polygon given by boundary_pts (N,2) in row,col
    to produce a binary mask of size 'shape' (rows,cols).
    """
    # boundary is (row, col)
    # we can use skimage.draw.polygon to fill
    rr, cc = skimage.draw.polygon(boundary_pts[:,0], boundary_pts[:,1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask

def resample_contour(contour, n_points=200):
    """
    contour: shape (M,2) array of (row,col)
    n_points: desired number of output points
    returns shape (n_points,2)
    """
    # 1) compute cumulative arc length
    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_length[-1]

    # 2) sample arc lengths at n_points
    desired = np.linspace(0, total_length, n_points)
    
    # 3) for each desired arc length, find where it fits in 'arc_length'
    resampled = []
    idx = 0
    for d in desired:
        # find segment
        while idx < len(arc_length)-1 and not (arc_length[idx] <= d <= arc_length[idx+1]):
            idx += 1
        if idx >= len(arc_length)-1:
            resampled.append(contour[-1])
            continue
        
        # linear interpolation
        seg_ratio = (d - arc_length[idx]) / (arc_length[idx+1] - arc_length[idx])
        pt = contour[idx] + seg_ratio*(contour[idx+1] - contour[idx])
        resampled.append(pt)
    
    return np.array(resampled)



def resample_pair(X, Y):
    """
    Resample both X and Y contours to the minimum number of points between them.
    Returns resampled_X, resampled_Y
    """
    min_points = 10 #min(len(X), len(Y))
    resampled_X = resample_contour(X, n_points=min_points)
    resampled_Y = resample_contour(Y, n_points=min_points)
    return resampled_X, resampled_Y

def _worker_distance_field_true(mask_slice_np: np.ndarray) -> np.ndarray:
    """Processes a single mask for distance_field_true."""
    H, W = mask_slice_np.shape
    field_slice = np.zeros((1, H, W), dtype=np.float32) # Ensure it's (1,H,W) for concatenation
    fg = mask_slice_np > 0.5
    if fg.any():
        field_slice[0] = edt(fg) + edt(~fg)
    return field_slice

def _worker_distance_field_procrustes(args_tuple) -> np.ndarray:
    """Processes a single pair of (gt_mask, pred_prob) for distance_field_procrustes."""
    gt_mask_slice, pr_prob_slice, penalty_val, shape_h, shape_w = args_tuple

    # --- A) binarise -------------------------------------
    true_mask = gt_mask_slice # Already binarized or raw if needed by label_and_extract
    pred_mask_binary = pr_prob_slice >= 0.5

    # --- B) boundaries -----------------------------------
    # These helper functions must be globally defined/imported
    gt_bnd  = label_and_extract_boundaries(true_mask > 0.5) # Ensure input is binary if expected
    pr_bnd  = label_and_extract_boundaries(pred_mask_binary)

    # --- C) Hungarian assignment -------------------------
    assigns = match_objects(gt_bnd, pr_bnd)

    # --- D-E-F) build final prediction mask --------------
    aligned_mask      = np.zeros_like(true_mask, dtype=bool)
    unassigned_pred   = np.zeros_like(true_mask, dtype=bool)
    taken_pr_ids      = set()

    # Handle cases where assigns might be empty or objects are missing
    if assigns:
        for (gt_id, pr_id, _) in assigns:
            taken_pr_ids.add(pr_id)
            try:
                gt_pts = next(arr for (gid, arr) in gt_bnd if gid == gt_id)
                pr_pts = next(arr for (pid, arr) in pr_bnd if pid == pr_id)
            except StopIteration:
                # Object ID not found, skip this assignment
                print(f"Warning: Object ID not found in gt_bnd or pr_bnd. gt_id={gt_id}, pr_id={pr_id}. PID: {os.getpid()}")
                continue

            if gt_pts.size == 0 or pr_pts.size == 0: # Skip if no points
                continue

            gt_pts_resampled, pr_pts_resampled = resample_pair(gt_pts, pr_pts)
            _, aligned_pr_pts, _ = procrustes_align(gt_pts_resampled, pr_pts_resampled,
                                                    allow_scaling=True)
            if aligned_pr_pts.size > 0: # Ensure there are points to fill
                 aligned_mask |= fill_polygon(aligned_pr_pts, true_mask.shape)


    for (pid, boundary) in pr_bnd:
        if pid not in taken_pr_ids:
            if boundary.size > 0: # Ensure there are points to fill
                unassigned_pred |= fill_polygon(boundary, true_mask.shape)

    final_pred = aligned_mask | unassigned_pred   # (H,W) bool

    # --- G-H) distance transform with penalty ------------
    dt_final = edt(~final_pred)
    dt_final[unassigned_pred] = penalty_val    # high penalty
    
    return dt_final.reshape(1, shape_h, shape_w) # Ensure (1,H,W) for concat


















# class ProcrustesLossBag(nn.Module):
#     """
#     Hausdorff-style loss with Procrustes alignment.
#     Expects pred_prob & true_mask in (B,1,H,W) or (B,1,D,H,W).
#     """

#     def __init__(self, alpha: float = 2.0, penalty_val: float = 500.0):
#         super().__init__()
#         self.alpha        = alpha
#         self.penalty_val  = penalty_val          # used in step H

#     # ------------------------  A. True DT  ------------------------- #
#     @torch.no_grad()
#     def distance_field_true(self, masks: np.ndarray) -> np.ndarray:
#         """
#         masks : (B,1,H,W) or (B,H,W) NumPy array, values 0/1
#         Returns : (B,1,H,W) float32 distance field
#         """
#         if masks.ndim == 4:          # (B,1,H,W) → squeeze channel
#             masks = masks[:, 0]

#         B, H, W = masks.shape
#         field = np.zeros((B, 1, H, W), dtype=np.float32)

#         for b in range(B):
#             fg = masks[b] > 0.5
#             if fg.any():
#                 field[b, 0] = edt(fg) + edt(~fg)

#         return field

#     # ------------------------  B. Procrustes DT  ------------------- #
#     @torch.no_grad()
#     def distance_field_procrustes(
#         self,
#         gt_masks:  np.ndarray,      # (B,1,H,W)  or (B,H,W)
#         pr_probs:  np.ndarray       # same shape, values in [0,1]
#     ) -> np.ndarray:

#         if gt_masks.ndim == 4:
#             gt_masks = gt_masks[:, 0]
#         if pr_probs.ndim == 4:
#             pr_probs = pr_probs[:, 0]

#         B, H, W = gt_masks.shape
#         procrustes_field = np.zeros((B, 1, H, W), dtype=np.float32)

#         for b in range(B):
#             # --- A) binarise -------------------------------------
#             true_mask = gt_masks[b]
#             pred_mask = pr_probs[b] >= 0.5

#             # --- B) boundaries -----------------------------------
#             gt_bnd  = label_and_extract_boundaries(true_mask)
#             pr_bnd  = label_and_extract_boundaries(pred_mask)

#             # --- C) Hungarian assignment -------------------------
#             assigns = match_objects(gt_bnd, pr_bnd)

#             # --- D-E-F) build final prediction mask --------------
#             aligned_mask      = np.zeros_like(true_mask, dtype=bool)
#             unassigned_pred   = np.zeros_like(true_mask, dtype=bool)
#             taken_pr_ids      = set()

#             for (gt_id, pr_id, _) in assigns:
#                 taken_pr_ids.add(pr_id)
#                 gt_pts = next(arr for (gid, arr) in gt_bnd if gid == gt_id)
#                 pr_pts = next(arr for (pid, arr) in pr_bnd if pid == pr_id)
#                 gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
#                 _, aligned_pr, _ = procrustes_align(gt_pts, pr_pts,
#                                                     allow_scaling=True)
#                 aligned_mask |= fill_polygon(aligned_pr, true_mask.shape)

#             for (pid, boundary) in pr_bnd:
#                 if pid not in taken_pr_ids:
#                     unassigned_pred |= fill_polygon(boundary, true_mask.shape)

#             final_pred = aligned_mask | unassigned_pred   # (H,W) bool

#             # --- G-H) distance transform with penalty ------------
#             dt_final = edt(~final_pred)
#             dt_final[unassigned_pred] = self.penalty_val    # high penalty
#             procrustes_field[b, 0] = dt_final

#         return procrustes_field

#     # ------------------------  C. Forward ------------------------- #
#     def forward(
#         self,
#         pred_prob: torch.Tensor,   # (B,1,H,W) probs in [0,1]
#         true_mask: torch.Tensor,   # (B,1,H,W) binary
#     ) -> torch.Tensor:

#         assert pred_prob.shape == true_mask.shape, \
#                "Prediction and target must share shape"

#         device, dtype = pred_prob.device, pred_prob.dtype

#         # ---- pre-compute distance fields on CPU -----------------
#         with torch.no_grad():
#             true_dt_np = self.distance_field_true(
#                 true_mask.detach().cpu().numpy())
#             pred_dt_np = self.distance_field_procrustes(
#                 true_mask.detach().cpu().numpy(),
#                 pred_prob.detach().cpu().numpy())

#         true_dt = torch.from_numpy(true_dt_np).to(device=device, dtype=dtype)
#         pred_dt = torch.from_numpy(pred_dt_np).to(device=device, dtype=dtype)

#         # ---- loss ----------------------------------------------
#         err      = (pred_prob - true_mask).pow(2)          # squared error
#         distance = pred_dt.pow(self.alpha) + true_dt.pow(self.alpha)
#         loss     = (err * distance).mean()

#         return loss

class ProcrustesLossBag(nn.Module):
    def __init__(self, alpha: float = 2.0, penalty_val: float = 500.0, num_workers: int = 8):
        super().__init__()
        self.alpha        = alpha
        self.penalty_val  = penalty_val
        # Determine number of workers: 0 means main process, >0 for multiprocessing
        if num_workers <= 0:
            self.num_workers = 0 # Use 0 to signify no multiprocessing
        else:
            self.num_workers = min(num_workers, multiprocessing.cpu_count())
        print(f"ProcrustesLossBag initialized with {self.num_workers if self.num_workers > 0 else 'single-process'} worker(s) for CPU tasks.")


    @torch.no_grad()
    def distance_field_true(self, masks: np.ndarray) -> np.ndarray:
        if masks.ndim == 4:          # (B,1,H,W) → squeeze channel
            masks_squeezed = masks[:, 0] # (B, H, W)
        else:
            masks_squeezed = masks

        B, H, W = masks_squeezed.shape
        
        if self.num_workers > 0 and B > 1: # Use multiprocessing if enabled and batch size > 1
            # Prepare data for multiprocessing: list of single mask slices
            tasks = [masks_squeezed[b] for b in range(B)]
            
            # Use a context manager for the pool
            with multiprocessing.Pool(processes=min(B, self.num_workers)) as pool:
                results = pool.map(_worker_distance_field_true, tasks)
            
            # Stack results: results is a list of (1,H,W) arrays
            field = np.concatenate(results, axis=0) # Resulting shape (B, H, W)
        else: # Fallback to sequential processing
            field_list = []
            for b in range(B):
                field_list.append(_worker_distance_field_true(masks_squeezed[b]))
            field = np.concatenate(field_list, axis=0)

        return field.reshape(B, 1, H, W) # Reshape to (B,1,H,W)


    @torch.no_grad()
    def distance_field_procrustes(
        self,
        gt_masks:  np.ndarray,      # (B,1,H,W)  or (B,H,W)
        pr_probs:  np.ndarray       # same shape, values in [0,1]
    ) -> np.ndarray:

        if gt_masks.ndim == 4:
            gt_masks_squeezed = gt_masks[:, 0]
        else:
            gt_masks_squeezed = gt_masks
            
        if pr_probs.ndim == 4:
            pr_probs_squeezed = pr_probs[:, 0]
        else:
            pr_probs_squeezed = pr_probs

        B, H, W = gt_masks_squeezed.shape
        
        if self.num_workers > 0 and B > 1: # Use multiprocessing if enabled and batch size > 1
            # Prepare tasks: list of tuples, each (gt_mask_slice, pr_prob_slice, penalty_val, H, W)
            tasks = [(gt_masks_squeezed[b], pr_probs_squeezed[b], self.penalty_val, H, W) for b in range(B)]
            
            with multiprocessing.Pool(processes=min(B, self.num_workers)) as pool:
                results = pool.map(_worker_distance_field_procrustes, tasks)
            
            procrustes_field = np.concatenate(results, axis=0) # Resulting shape (B,H,W)
        else: # Fallback to sequential processing
            field_list = []
            for b in range(B):
                args = (gt_masks_squeezed[b], pr_probs_squeezed[b], self.penalty_val, H, W)
                field_list.append(_worker_distance_field_procrustes(args))
            procrustes_field = np.concatenate(field_list, axis=0)
            
        return procrustes_field.reshape(B, 1, H, W) # Reshape to (B,1,H,W)

    def forward(
        self,
        pred_prob: torch.Tensor,   # (B,1,H,W) probs in [0,1]
        true_mask: torch.Tensor,   # (B,1,H,W) binary
    ) -> torch.Tensor:

        assert pred_prob.shape == true_mask.shape, \
               "Prediction and target must share shape"

        device, dtype = pred_prob.device, pred_prob.dtype

        # ---- pre-compute distance fields on CPU -----------------
        # Convert to NumPy on CPU once
        true_mask_np = true_mask.detach().cpu().numpy()
        pred_prob_np = pred_prob.detach().cpu().numpy()
        
        with torch.no_grad(): # Redundant here as methods also have it, but fine
            true_dt_np = self.distance_field_true(true_mask_np)
            pred_dt_np = self.distance_field_procrustes(true_mask_np, pred_prob_np)

        true_dt = torch.from_numpy(true_dt_np).to(device=device, dtype=dtype)
        pred_dt = torch.from_numpy(pred_dt_np).to(device=device, dtype=dtype)

        # ---- loss ----------------------------------------------
        err      = (pred_prob - true_mask).pow(2)          # squared error
        distance = pred_dt.pow(self.alpha) + true_dt.pow(self.alpha)
        loss     = (err * distance).mean()

        return loss