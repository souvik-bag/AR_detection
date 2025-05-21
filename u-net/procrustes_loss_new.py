from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.transforms.utils import distance_transform_edt
from monai.utils import LossReduction

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
    # print(f'labeled shape : {labeled.shape}')
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
    min_points = min(len(X), len(Y))
    resampled_X = resample_contour(X, n_points=min_points)
    resampled_Y = resample_contour(Y, n_points=min_points)
    return resampled_X, resampled_Y


# class ProcrustesDTLoss(_Loss):
#     """
#     Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
#     multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
#     (BNHW[D]).

#     Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
#     must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
#     can be 1 or N (one-hot format).

#     The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
#     Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
#     """

#     def __init__(
#         self,
#         alpha: float = 2.0,
#         include_background: bool = False,
#         to_onehot_y: bool = False,
#         sigmoid: bool = False,
#         softmax: bool = False,
#         other_act: Callable | None = None,
#         reduction: LossReduction | str = LossReduction.MEAN,
#         batch: bool = False,
#     ) -> None:
#         """
#         Args:
#             include_background: if False, channel index 0 (background category) is excluded from the calculation.
#                 if the non-background segmentations are small compared to the total image size they can get overwhelmed
#                 by the signal from the background so excluding it in such cases helps convergence.
#             to_onehot_y: whether to convert the ``target`` into the one-hot format,
#                 using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
#             sigmoid: if True, apply a sigmoid function to the prediction.
#             softmax: if True, apply a softmax function to the prediction.
#             other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
#                 ``other_act = torch.tanh``.
#             reduction: {``"none"``, ``"mean"``, ``"sum"``}
#                 Specifies the reduction to apply to the output. Defaults to ``"mean"``.

#                 - ``"none"``: no reduction will be applied.
#                 - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
#                 - ``"sum"``: the output will be summed.
#             batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
#                 Defaults to False, a loss value is computed independently from each item in the batch
#                 before any `reduction`.

#         Raises:
#             TypeError: When ``other_act`` is not an ``Optional[Callable]``.
#             ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
#                 Incompatible values.

#         """
#         super(ProcrustesDTLoss, self).__init__(reduction=LossReduction(reduction).value)
#         if other_act is not None and not callable(other_act):
#             raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
#         if int(sigmoid) + int(softmax) > 1:
#             raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

#         self.alpha = alpha
#         self.include_background = include_background
#         self.to_onehot_y = to_onehot_y
#         self.sigmoid = sigmoid
#         self.softmax = softmax
#         self.other_act = other_act
#         self.batch = batch

#     @torch.no_grad()
#     def distance_field(self, img: torch.Tensor) -> torch.Tensor:
#         """Generate distance transform.

#         Args:
#             img (np.ndarray): input mask as NCHWD or NCHW.

#         Returns:
#             np.ndarray: Distance field.
#         """
#         field = torch.zeros_like(img)

#         for batch_idx in range(len(img)):
#             fg_mask = img[batch_idx] > 0.5

#             # For cases where the mask is entirely background or entirely foreground
#             # the distance transform is not well defined for all 1s,
#             # which always would happen on either foreground or background, so skip
#             if fg_mask.any() and not fg_mask.all():
#                 fg_dist: torch.Tensor = distance_transform_edt(fg_mask)  # type: ignore
#                 bg_mask = ~fg_mask
#                 bg_dist: torch.Tensor = distance_transform_edt(bg_mask)  # type: ignore

#                 field[batch_idx] = fg_dist + bg_dist

#         return field
    
#     @torch.no_grad()
#     def distance_field_procrustes(self, gt_mask: torch.Tensor, pred_proba: torch.tensor) -> torch.Tensor:
#         """Generate distance transform.

#         Args:
#             img (np.ndarray): input mask as NCHWD or NCHW.

#         Returns:
#             np.ndarray: Distance field.
#         """
        
#         procrustes_field = torch.zeros_like(gt_mask)
        
#         for batch_idx in range(len(gt_mask)):
        
#             # A) Binarize predicted probability
#             pred_mask = (pred_proba[batch_idx] >= 0.5)
#             true_mask = gt_mask[batch_idx]
            
#             if pred_mask.is_cuda:
#                 pred_mask = pred_mask.cpu().numpy()
#             if true_mask.is_cuda:
#                 true_mask = true_mask.cpu().numpy()
            
        
        
#         # B) Extract boundaries
#             gt_boundaries = label_and_extract_boundaries(true_mask)
#             pred_boundaries = label_and_extract_boundaries(pred_mask)
        
#         # C) Match objects between ground truth and prediction
#             assignments = match_objects(gt_boundaries, pred_boundaries)
        
#         # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
#             aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
#             assigned_pred_ids = set()
#             for (gt_id, pr_id, cost) in assignments:
#                 assigned_pred_ids.add(pr_id)
#                 gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
#                 pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
#                 gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
#                 std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=self.allow_scaling)
#                 aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
#                 aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
        
#         # E) For unassigned predicted objects (false positives), keep their original shape.
#             unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
#             for (pid, boundary) in pred_boundaries:
#                 if pid not in assigned_pred_ids:
#                     obj_mask = fill_polygon(boundary, gt_mask.shape)
#                     unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
#         # F) Construct final prediction mask.
#             final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
#         # G) Compute global distance transforms.
#             dt_gt = distance_transform_edt(~gt_mask.astype(bool))
#             dt_final = distance_transform_edt(~final_pred_mask)
        
#         # H) Customize the prediction distance transform: assign high penalty in unassigned regions.
#             if self.penalty_constant is None:
#                 penalty_value = dt_final.max()
#             else:
#                 penalty_value = self.penalty_constant
#             dt_final_custom = dt_final.copy()
#             dt_final_custom[unassigned_pred_mask] = penalty_value
        
#             procrustes_field[batch_idx] = dt_final_custom
            
            
#         procrustes_field = torch.from_numpy(procrustes_field).to('cuda')
            
        
#         return(procrustes_field)
        
        
        
        
    
  

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: the shape should be BNHW[D], where N is the number of classes.
#             target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

#         Raises:
#             ValueError: If the input is not 2D (NCHW) or 3D (NCHWD).
#             AssertionError: When input and target (after one hot transform if set)
#                 have different shapes.
#             ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

#         Example:
#             >>> import torch
#             >>> from monai.losses.hausdorff_loss import HausdorffDTLoss
#             >>> from monai.networks.utils import one_hot
#             >>> B, C, H, W = 7, 5, 3, 2
#             >>> input = torch.rand(B, C, H, W)
#             >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
#             >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
#             >>> self = HausdorffDTLoss(reduction='none')
#             >>> loss = self(input, target)
#             >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
#         """
#         if input.dim() != 4 and input.dim() != 5:
#             raise ValueError("Only 2D (NCHW) and 3D (NCHWD) supported")

#         if self.sigmoid:
#             input = torch.sigmoid(input)

#         n_pred_ch = input.shape[1]
#         if self.softmax:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `softmax=True` ignored.")
#             else:
#                 input = torch.softmax(input, 1)

#         if self.other_act is not None:
#             input = self.other_act(input)

#         if self.to_onehot_y:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
#             else:
#                 target = one_hot(target, num_classes=n_pred_ch)

#         if not self.include_background:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `include_background=False` ignored.")
#             else:
#                 # If skipping background, removing first channel
#                 target = target[:, 1:]
#                 input = input[:, 1:]

#         if target.shape != input.shape:
#             raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

#         device = input.device
#         all_f = []
#         for i in range(input.shape[1]):
#             ch_input = input[:, [i]]
#             ch_target = target[:, [i]]
#             pred_dt = self.distance_field_procrustes(ch_target.detach(),ch_input.detach()).float()
#             target_dt = self.distance_field(ch_target.detach()).float()

#             pred_error = (ch_input - ch_target) ** 2
#             distance = pred_dt**self.alpha + target_dt**self.alpha

#             running_f = pred_error * distance.to(device)
#             reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
#             if self.batch:
#                 # reducing spatial dimensions and batch
#                 reduce_axis = [0] + reduce_axis
#             all_f.append(running_f.mean(dim=reduce_axis, keepdim=True))
#         f = torch.cat(all_f, dim=1)
#         if self.reduction == LossReduction.MEAN.value:
#             f = torch.mean(f)  # the batch and channel average
#         elif self.reduction == LossReduction.SUM.value:
#             f = torch.sum(f)  # sum over the batch and channel dims
#         elif self.reduction == LossReduction.NONE.value:
#             # If we are not computing voxelwise loss components at least make sure a none reduction maintains a
#             # broadcastable shape
#             broadcast_shape = list(f.shape[0:2]) + [1] * (len(ch_input.shape) - 2)
#             f = f.view(broadcast_shape)
#         else:
#             raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

#         return f
    
    
    
    
    
    

# class ProcrustesLossBag(nn.Module):
    
#     def __init__(self, alpha=2.0, **kwargs):
#         super(ProcrustesLossBag, self).__init__()
#         self.alpha = alpha
    
    
#     @torch.no_grad()
#     def distance_field_true(self, img: np.ndarray) -> np.ndarray:
#         field = np.zeros_like(img)

#         for batch in range(len(img)):
#             fg_mask = img[batch] > 0.5

#             if fg_mask.any():
#                 bg_mask = ~fg_mask

#                 fg_dist = edt(fg_mask)
#                 bg_dist = edt(bg_mask)

#                 field[batch] = fg_dist + bg_dist

#         return field 
#     @torch.no_grad()
#     def distance_field_procrustes(self, gt_mask: np.ndarray, pred_proba: np.ndarray) -> np.ndarray:
#         """Generate distance transform.

#         Args:
#             img (np.ndarray): input mask as NCHWD or NCHW.

#         Returns:
#             np.ndarray: Distance field.
#         """
#         # if pred_mask.is_cuda:
#         #     pred_mask = pred_mask.cpu().numpy()
#         # if true_mask.is_cuda:
#         #     true_mask = true_mask.cpu().numpy()
        
        
        
#         procrustes_field = np.zeros_like(gt_mask)
        
#         for batch_idx in range(len(gt_mask)):
        
#             # A) Binarize predicted probability
#             pred_mask = (pred_proba[batch_idx] >= 0.5)
#             true_mask = gt_mask[batch_idx]
            
            
            
        
        
#         # B) Extract boundaries
#             gt_boundaries = label_and_extract_boundaries(true_mask)
#             pred_boundaries = label_and_extract_boundaries(pred_mask)
        
#         # C) Match objects between ground truth and prediction
#             assignments = match_objects(gt_boundaries, pred_boundaries)
        
#         # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
#             aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
#             assigned_pred_ids = set()
#             for (gt_id, pr_id, cost) in assignments:
#                 assigned_pred_ids.add(pr_id)
#                 gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
#                 pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
#                 gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
#                 std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=True)
#                 aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
#                 aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
        
#         # E) For unassigned predicted objects (false positives), keep their original shape.
#             unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
#             for (pid, boundary) in pred_boundaries:
#                 if pid not in assigned_pred_ids:
#                     obj_mask = fill_polygon(boundary, gt_mask.shape)
#                     unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
#         # F) Construct final prediction mask.
#             final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
#         # G) Compute global distance transforms.
#             dt_gt = distance_transform_edt(~gt_mask.astype(bool))
#             dt_final = distance_transform_edt(~final_pred_mask)
        
#         # H) Customize the prediction distance transform: assign high penalty in unassigned regions.
#             # if self.penalty_constant is None:
#             #     penalty_value = dt_final.max()
#             # else:
#             #     penalty_value = 500
#             dt_final_custom = dt_final.copy()
#             dt_final_custom[unassigned_pred_mask] = 500 #penalty_value
        
#             procrustes_field[batch_idx] = dt_final_custom
            
            
#         # procrustes_field = torch.from_numpy(procrustes_field).to('cuda')
            
        
#         return(procrustes_field)
            
#     def forward(
#         self, pred_prob: torch.Tensor, true_mask: torch.Tensor, debug=False
#     ) -> torch.Tensor:
#         """
#         Uses one binary channel: 1 - fg, 0 - bg
#         pred: (b, 1, x, y, z) or (b, 1, x, y)
#         target: (b, 1, x, y, z) or (b, 1, x, y)
#         """
#         assert pred_prob.dim() == 4 or pred_prob.dim() == 5, "Only 2D and 3D supported"
#         assert (
#             pred_prob.dim() == true_mask.dim()
#         ), "Prediction and target need to be of same dimension"

#         # pred = torch.sigmoid(pred)

#         true_dt = torch.from_numpy(self.distance_field_true(true_mask.detach().cpu().numpy())).float()
#         pred_dt = torch.from_numpy(self.distance_field_procrustes(true_mask.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())).float()

#         pred_error = (pred_prob - true_mask) ** 2
#         distance = pred_dt ** self.alpha + target_dt ** self.alpha

#         dt_field = pred_error * distance
#         loss = dt_field.mean()

#         if debug:
#             return (
#                 loss.cpu().numpy(),
#                 (
#                     dt_field.cpu().numpy()[0, 0],
#                     pred_error.cpu().numpy()[0, 0],
#                     distance.cpu().numpy()[0, 0],
#                     pred_dt.cpu().numpy()[0, 0],
#                     target_dt.cpu().numpy()[0, 0],
#                 ),
#             )

#         else:
#             return loss
    



class ProcrustesLossBag(nn.Module):
    """
    Hausdorff-style loss with Procrustes alignment.
    Expects pred_prob & true_mask in (B,1,H,W) or (B,1,D,H,W).
    """

    def __init__(self, alpha: float = 2.0, penalty_val: float = 500.0):
        super().__init__()
        self.alpha        = alpha
        self.penalty_val  = penalty_val          # used in step H

    # ------------------------  A. True DT  ------------------------- #
    @torch.no_grad()
    def distance_field_true(self, masks: np.ndarray) -> np.ndarray:
        """
        masks : (B,1,H,W) or (B,H,W) NumPy array, values 0/1
        Returns : (B,1,H,W) float32 distance field
        """
        if masks.ndim == 4:          # (B,1,H,W) → squeeze channel
            masks = masks[:, 0]

        B, H, W = masks.shape
        field = np.zeros((B, 1, H, W), dtype=np.float32)

        for b in range(B):
            fg = masks[b] > 0.5
            if fg.any():
                field[b, 0] = edt(fg) + edt(~fg)

        return field

    # ------------------------  B. Procrustes DT  ------------------- #
    @torch.no_grad()
    def distance_field_procrustes(
        self,
        gt_masks:  np.ndarray,      # (B,1,H,W)  or (B,H,W)
        pr_probs:  np.ndarray       # same shape, values in [0,1]
    ) -> np.ndarray:

        if gt_masks.ndim == 4:
            gt_masks = gt_masks[:, 0]
        if pr_probs.ndim == 4:
            pr_probs = pr_probs[:, 0]

        B, H, W = gt_masks.shape
        procrustes_field = np.zeros((B, 1, H, W), dtype=np.float32)

        for b in range(B):
            # --- A) binarise -------------------------------------
            true_mask = gt_masks[b]
            pred_mask = pr_probs[b] >= 0.5

            # --- B) boundaries -----------------------------------
            gt_bnd  = label_and_extract_boundaries(true_mask)
            pr_bnd  = label_and_extract_boundaries(pred_mask)

            # --- C) Hungarian assignment -------------------------
            assigns = match_objects(gt_bnd, pr_bnd)

            # --- D-E-F) build final prediction mask --------------
            aligned_mask      = np.zeros_like(true_mask, dtype=bool)
            unassigned_pred   = np.zeros_like(true_mask, dtype=bool)
            taken_pr_ids      = set()

            for (gt_id, pr_id, _) in assigns:
                taken_pr_ids.add(pr_id)
                gt_pts = next(arr for (gid, arr) in gt_bnd if gid == gt_id)
                pr_pts = next(arr for (pid, arr) in pr_bnd if pid == pr_id)
                gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
                _, aligned_pr, _ = procrustes_align(gt_pts, pr_pts,
                                                    allow_scaling=True)
                aligned_mask |= fill_polygon(aligned_pr, true_mask.shape)

            for (pid, boundary) in pr_bnd:
                if pid not in taken_pr_ids:
                    unassigned_pred |= fill_polygon(boundary, true_mask.shape)

            final_pred = aligned_mask | unassigned_pred   # (H,W) bool

            # --- G-H) distance transform with penalty ------------
            dt_final = edt(~final_pred)
            dt_final[unassigned_pred] = self.penalty_val    # high penalty
            procrustes_field[b, 0] = dt_final

        return procrustes_field

    # ------------------------  C. Forward ------------------------- #
    def forward(
        self,
        pred_prob: torch.Tensor,   # (B,1,H,W) probs in [0,1]
        true_mask: torch.Tensor,   # (B,1,H,W) binary
    ) -> torch.Tensor:

        assert pred_prob.shape == true_mask.shape, \
               "Prediction and target must share shape"

        device, dtype = pred_prob.device, pred_prob.dtype

        # ---- pre-compute distance fields on CPU -----------------
        with torch.no_grad():
            true_dt_np = self.distance_field_true(
                true_mask.detach().cpu().numpy())
            pred_dt_np = self.distance_field_procrustes(
                true_mask.detach().cpu().numpy(),
                pred_prob.detach().cpu().numpy())

        true_dt = torch.from_numpy(true_dt_np).to(device=device, dtype=dtype)
        pred_dt = torch.from_numpy(pred_dt_np).to(device=device, dtype=dtype)

        # ---- loss ----------------------------------------------
        err      = (pred_prob - true_mask).pow(2)          # squared error
        distance = pred_dt.pow(self.alpha) + true_dt.pow(self.alpha)
        loss     = (err * distance).mean()

        return loss