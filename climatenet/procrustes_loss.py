import numpy as np
import skimage.measure
import skimage.draw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

############################################
# 1. Get Objects & Boundaries
############################################

def label_and_extract_boundaries(binary_mask):
    """
    Given a 2D binary mask, label connected components and
    extract the boundary points for each labeled object.
    
    Returns a list of (label_id, boundary_points), 
    where boundary_points is an (N,2) array of [row,col].
    """
    labeled = skimage.measure.label(binary_mask, connectivity=2)
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


# class ProcrustesLoss:
#     def __init__(self, threshold=0.8, allow_scaling=True, penalty_constant=None):
#         """
#         Initialize the custom loss function.
        
#         Parameters:
#          - threshold: Threshold to binarize the prediction probabilities.
#          - allow_scaling: Whether to allow scaling during Procrustes alignment.
#          - penalty_constant: If provided, use this constant penalty value instead of dt_final.max()
#         """
#         self.threshold = threshold
#         self.allow_scaling = allow_scaling
#         self.penalty_constant = penalty_constant

#     def __call__(self, gt_mask, pred_proba):
#         """
#         Compute the loss given a ground truth mask and predicted probability map.
        
#         The loss is computed as:
        
#             loss = mean( (gt_mask - pred_proba)^2 * (dt_gt^2 + dt_final_custom^2) )
        
#         where:
#          - dt_gt: distance transform of the ground truth mask.
#          - dt_final_custom: customized distance transform for the prediction where pixels 
#                             inside unassigned objects are given a high penalty.
        
#         Returns:
#          - loss: Mean loss value (scalar).
#          - loss_map: Pixel-wise loss map.
#         """
#         # A) Binarize predicted probability
#         pred_mask = (pred_proba >= self.threshold)
        
#         # B) Extract boundaries for ground truth and prediction
#         gt_boundaries = label_and_extract_boundaries(gt_mask)
#         pred_boundaries = label_and_extract_boundaries(pred_mask)
        
#         # C) Match objects between ground truth and prediction
#         assignments = match_objects(gt_boundaries, pred_boundaries)
        
#         # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
#         aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
#         assigned_pred_ids = set()
        
#         for (gt_id, pr_id, cost) in assignments:
#             assigned_pred_ids.add(pr_id)
            
#             # Get boundaries for current objects
#             gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
#             pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
            
#             # Resample points to the same number if necessary.
#             gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
            
#             # Procrustes alignment: get transformed predicted points.
#             std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=self.allow_scaling)
            
#             # Rasterize the aligned predicted boundary.
#             aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
            
#             aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
            
#             # # Optional: visualize alignment (can be commented out in production)
#             # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#             # axes[0].scatter(gt_pts[:, 1], gt_pts[:, 0], color='blue', s=5)
#             # axes[0].imshow(gt_mask, cmap='gray', alpha=0.3)
#             # axes[0].set_title('GT Boundary')
#             # axes[0].invert_yaxis()

#             # axes[1].scatter(pr_pts[:, 1], pr_pts[:, 0], color='red', s=5)
#             # axes[1].imshow(pred_mask, cmap='gray', alpha=0.3)
#             # axes[1].set_title('Predicted Boundary')
#             # axes[1].invert_yaxis()

#             # axes[2].scatter(aligned_pr_pts[:, 1], aligned_pr_pts[:, 0], color='green', s=5)
#             # axes[2].imshow(gt_mask, cmap='gray', alpha=0.3)
#             # axes[2].set_title('Aligned Predicted Boundary')
#             # axes[2].invert_yaxis()

#             # for ax in axes:
#             #     ax.axis('equal')
#             #     ax.set_xticks([])
#             #     ax.set_yticks([])
#             # plt.tight_layout()
#             # plt.show()
        
#         # E) For unassigned predicted objects (false positives), keep their original shape.
#         unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
#         for (pid, boundary) in pred_boundaries:
#             if pid not in assigned_pred_ids:
#                 obj_mask = fill_polygon(boundary, gt_mask.shape)
#                 unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
#         # F) Construct final prediction mask by combining aligned assigned objects and unassigned objects.
#         final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
#         # G) Compute global distance transforms.
#         gt_mask_bool = gt_mask.astype(bool)
#         dt_gt = distance_transform_edt(~gt_mask_bool)
#         dt_final = distance_transform_edt(~final_pred_mask)
        
#         # H) Customize the prediction distance transform:
#         # For pixels inside unassigned objects, override with a high penalty.
#         if self.penalty_constant is None:
#             penalty_value = dt_final.max()
#         else:
#             penalty_value = self.penalty_constant
#         dt_final_custom = dt_final.copy()
#         dt_final_custom[unassigned_pred_mask] = penalty_value
        
#         # Compute pixel-wise error between ground truth and prediction probabilities.
#         error = (gt_mask.astype(np.float32) - pred_proba)**2
        
#         # Use squared distance transforms as weights.
#         weight = dt_gt**2 + dt_final_custom**2
        
#         # Compute the loss map and mean loss.
#         loss_map = error * weight
#         loss = loss_map.mean()
        
#         # # Optionally, you can also plot the distance transforms.
#         # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#         # im0 = axes[0].imshow(dt_gt, cmap='viridis')
#         # axes[0].set_title('GT Distance Transform')
#         # axes[0].invert_yaxis()
#         # fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
#         # im1 = axes[1].imshow(dt_final_custom, cmap='viridis')
#         # axes[1].set_title('Final Prediction DT (Custom)')
#         # axes[1].invert_yaxis()
#         # fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
#         # plt.tight_layout()
#         # plt.show()
        
#         return loss, loss_map

# # If run as a script, you can include a test block:
# if __name__ == "__main__":
#     # Dummy example data
#     gt_mask = np.zeros((100, 100), dtype=bool)
#     gt_mask[30:70, 30:70] = True  # ground truth square
    
#     pred_proba = np.zeros((100, 100), dtype=np.float32)
#     pred_proba[35:75, 35:75] = 0.9  # high probability for an overlapping square
    
#     # Instantiate the loss function
#     loss_fn = ProcrustesLoss(threshold=0.8, allow_scaling=True, penalty_constant=1000)
    
#     loss_value, loss_map = loss_fn(gt_mask, pred_proba)
#     print("Loss:", loss_value)

# class ProcrustesLoss(nn.Module):
#     def __init__(self, threshold=0.8, allow_scaling=True, penalty_constant=None):
#         """
#         Initialize the custom Procrustes-based loss function.
        
#         Parameters:
#          - threshold: Threshold to binarize the prediction probabilities.
#          - allow_scaling: Whether to allow scaling during Procrustes alignment.
#          - penalty_constant: If provided, use this constant penalty value instead of dt_final.max()
#         """
#         super(ProcrustesLoss, self).__init__()
#         self.threshold = threshold
#         self.allow_scaling = allow_scaling
#         self.penalty_constant = penalty_constant

#     @torch.no_grad()
#     def compute_loss_field(self, gt_mask: np.ndarray, pred_proba: np.ndarray) -> np.ndarray:
#         """
#         Compute the loss field for a batch.
#         gt_mask, pred_proba: numpy arrays of shape (B, H, W)
#         Returns:
#          - loss_field: numpy array of shape (B, H, W)
#         """
#         batch = gt_mask.shape[0]
#         loss_field = np.zeros_like(gt_mask, dtype=np.float32)
#         for i in range(batch):
#             _, loss_map = self._procrustes_loss(gt_mask[i], pred_proba[i])
#             loss_field[i] = loss_map
#         return loss_field

#     def _procrustes_loss(self, gt_mask, pred_proba):
#         """
#         Compute the Procrustes-based loss for one image.
#         gt_mask and pred_proba are numpy arrays of shape (H, W).
#         Returns:
#          - loss: scalar loss value.
#          - loss_map: pixel-wise loss map.
#         """
#         # A) Binarize predicted probability
#         pred_mask = (pred_proba >= self.threshold)
        
#         # B) Extract boundaries
#         gt_boundaries = label_and_extract_boundaries(gt_mask)
#         pred_boundaries = label_and_extract_boundaries(pred_mask)
        
#         # C) Match objects between ground truth and prediction
#         assignments = match_objects(gt_boundaries, pred_boundaries)
        
#         # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
#         aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
#         assigned_pred_ids = set()
        
#         for (gt_id, pr_id, cost) in assignments:
#             assigned_pred_ids.add(pr_id)
#             gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
#             pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
#             gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
#             std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=self.allow_scaling)
#             aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
#             aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
        
#         # E) For unassigned predicted objects (false positives), keep their original shape.
#         unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
#         for (pid, boundary) in pred_boundaries:
#             if pid not in assigned_pred_ids:
#                 obj_mask = fill_polygon(boundary, gt_mask.shape)
#                 unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
#         # F) Construct final prediction mask by combining assigned and unassigned.
#         final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
#         # G) Compute global distance transforms.
#         dt_gt = distance_transform_edt(~gt_mask.astype(bool))
#         dt_final = distance_transform_edt(~final_pred_mask)
        
#         # H) Customize the prediction distance transform: assign high penalty in unassigned regions.
#         if self.penalty_constant is None:
#             penalty_value = dt_final.max()
#         else:
#             penalty_value = self.penalty_constant
#         dt_final_custom = dt_final.copy()
#         dt_final_custom[unassigned_pred_mask] = penalty_value
        
#         # Compute pixel-wise squared error.
#         error = (gt_mask.astype(np.float32) - pred_proba)**2
#         # Weight error with the squared distance transforms.
#         weight = dt_gt**2 + dt_final_custom**2
        
#         loss_map = error * weight
#         loss = loss_map.mean()
#         return loss, loss_map

#     def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
#         """
#         Compute the loss.
        
#         Parameters:
#           pred: Tensor of shape [B, 1, H, W] with predicted probabilities (after softmax/sigmoid if needed).
#           target: Tensor of shape [B, 1, H, W] with binary ground truth (0 for background, 1 for AR).
        
#         Returns:
#           - loss: A scalar torch.Tensor with the mean loss.
#           - (optionally, if debug=True, returns additional debug information).
#         """
#         # Convert to numpy arrays; remove channel dimension.
#         pred_np = pred.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
#         target_np = target.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
        
#         # Compute the loss field over the batch.
#         loss_field = self.compute_loss_field(target_np, pred_np)
#         loss_value = loss_field.mean()
        
#         if debug:
#             # Return loss and loss map for the first image in the batch for debugging.
#             return torch.tensor(loss_value, device=pred.device), torch.tensor(loss_field[0], device=pred.device)
#         else:
#             return torch.tensor(loss_value, device=pred.device)
        
        
        
class ProcrustesLoss(nn.Module):
    def __init__(self, threshold=0.5, allow_scaling=True, penalty_constant=None):
        """
        Initialize the custom Procrustes-based loss function.
        """
        super(ProcrustesLoss, self).__init__()
        self.threshold = threshold
        self.allow_scaling = allow_scaling
        self.penalty_constant = penalty_constant
        
    @torch.no_grad()
    def _procrustes_loss(self, gt_mask, pred_proba, debug=False):
        """
        Compute the Procrustes-based loss for one image.
        gt_mask and pred_proba are numpy arrays of shape (H, W).
        Returns:
         - loss: scalar loss value.
         - loss_map: pixel-wise loss map.
         If debug is True, also return dt_gt and dt_final_custom.
        """
        # A) Binarize predicted probability
        pred_mask = (pred_proba >= self.threshold)
        
        # B) Extract boundaries
        gt_boundaries = label_and_extract_boundaries(gt_mask)
        pred_boundaries = label_and_extract_boundaries(pred_mask)
        
        # C) Match objects between ground truth and prediction
        assignments = match_objects(gt_boundaries, pred_boundaries)
        
        # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
        aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
        assigned_pred_ids = set()
        for (gt_id, pr_id, cost) in assignments:
            assigned_pred_ids.add(pr_id)
            gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
            pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
            gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
            std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=self.allow_scaling)
            aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
            aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
        
        # E) For unassigned predicted objects (false positives), keep their original shape.
        unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
        for (pid, boundary) in pred_boundaries:
            if pid not in assigned_pred_ids:
                obj_mask = fill_polygon(boundary, gt_mask.shape)
                unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
        # F) Construct final prediction mask.
        final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
        # G) Compute global distance transforms.
        dt_gt = distance_transform_edt(~gt_mask.astype(bool))
        dt_final = distance_transform_edt(~final_pred_mask)
        
        # H) Customize the prediction distance transform: assign high penalty in unassigned regions.
        if self.penalty_constant is None:
            penalty_value = dt_final.max()
        else:
            penalty_value = self.penalty_constant
        dt_final_custom = dt_final.copy()
        dt_final_custom[unassigned_pred_mask] = penalty_value
        
        # Compute pixel-wise squared error.
        error = (gt_mask.astype(np.float32) - pred_proba)**2
        # Weight error with the squared distance transforms.
        weight = dt_gt+ dt_final_custom #dt_gt**2 + dt_final_custom**2
        
        loss_map = error * weight
        loss = loss_map.mean()
        
        if debug:
            return loss, loss_map, dt_gt, dt_final_custom
        else:
            return loss, loss_map

    def compute_loss_field(self, gt_mask_batch: np.ndarray, pred_proba_batch: np.ndarray, debug=False):
        """
        Compute the loss field for a batch.
        Returns loss_field of shape (B, H, W) and, if debug=True, also dt_gt and dt_final_custom for the first image.
        """
        batch = gt_mask_batch.shape[0]
        loss_field = np.zeros_like(gt_mask_batch, dtype=np.float32)
        extra_debug = None
        for i in range(batch):
            if debug and i == 0:
                loss_val, loss_map, dt_gt, dt_final_custom = self._procrustes_loss(gt_mask_batch[i], pred_proba_batch[i], debug=True)
                extra_debug = {'dt_gt': dt_gt, 'dt_final_custom': dt_final_custom}
                loss_field[i] = loss_map
            else:
                loss_val, loss_map = self._procrustes_loss(gt_mask_batch[i], pred_proba_batch[i])
                loss_field[i] = loss_map
        return loss_field, extra_debug

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Compute the loss.
        
        Parameters:
          pred: Tensor of shape [B, 1, H, W] with predicted probabilities.
          target: Tensor of shape [B, 1, H, W] with binary ground truth (0: background, 1: AR).
        
        Returns:
          - loss: A scalar torch.Tensor with the mean loss.
          - if debug=True, returns a dict containing extra debug info.
        """
        # Convert to numpy arrays; remove channel dimension.
        pred_np = pred.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
        target_np = target.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
        
        loss_field, extra_debug = self.compute_loss_field(target_np, pred_np, debug=debug)
        loss_value = loss_field.mean()
        
        loss_tensor = torch.tensor(loss_value, device=pred.device)
        if debug:
            debug_info = {
                'loss_field': torch.tensor(loss_field, device=pred.device),
                'dt_gt': torch.tensor(extra_debug['dt_gt'], device=pred.device),
                'dt_final_custom': torch.tensor(extra_debug['dt_final_custom'], device=pred.device)
            }
            return loss_tensor, debug_info
        else:
            return loss_tensor
        
        
class ProcrustesLoss_metric(nn.Module):
    def __init__(self, threshold=0.5, allow_scaling=True, penalty_constant=None):
        """
        Initialize the custom Procrustes-based loss function.
        """
        super(ProcrustesLoss_metric, self).__init__()
        self.threshold = threshold
        self.allow_scaling = allow_scaling
        self.penalty_constant = penalty_constant
        
    @torch.no_grad()
    def _procrustes_loss(self, gt_mask, pred_mask, debug=False):
        """
        Compute the Procrustes-based loss for one image.
        gt_mask and pred_proba are numpy arrays of shape (H, W).
        Returns:
         - loss: scalar loss value.
         - loss_map: pixel-wise loss map.
         If debug is True, also return dt_gt and dt_final_custom.
        """
        # A) Binarize predicted probability
        # pred_mask = (pred_proba >= self.threshold)
        
        # B) Extract boundaries
        gt_boundaries = label_and_extract_boundaries(gt_mask)
        pred_boundaries = label_and_extract_boundaries(pred_mask)
        
        # C) Match objects between ground truth and prediction
        assignments = match_objects(gt_boundaries, pred_boundaries)
        
        # D) Process assigned objects: align using Procrustes and accumulate aligned masks.
        aligned_assigned_mask = np.zeros_like(gt_mask, dtype=bool)
        assigned_pred_ids = set()
        for (gt_id, pr_id, cost) in assignments:
            assigned_pred_ids.add(pr_id)
            gt_pts = next(arr for (gid, arr) in gt_boundaries if gid == gt_id)
            pr_pts = next(arr for (pid, arr) in pred_boundaries if pid == pr_id)
            gt_pts, pr_pts = resample_pair(gt_pts, pr_pts)
            std_gt_points, aligned_pr_pts, _ = procrustes_align(gt_pts, pr_pts, allow_scaling=self.allow_scaling)
            aligned_obj_mask = fill_polygon(aligned_pr_pts, gt_mask.shape)
            aligned_assigned_mask = np.logical_or(aligned_assigned_mask, aligned_obj_mask)
        
        # E) For unassigned predicted objects (false positives), keep their original shape.
        unassigned_pred_mask = np.zeros_like(gt_mask, dtype=bool)
        for (pid, boundary) in pred_boundaries:
            if pid not in assigned_pred_ids:
                obj_mask = fill_polygon(boundary, gt_mask.shape)
                unassigned_pred_mask = np.logical_or(unassigned_pred_mask, obj_mask)
        
        # F) Construct final prediction mask.
        final_pred_mask = np.logical_or(aligned_assigned_mask, unassigned_pred_mask)
        
        # G) Compute global distance transforms.
        dt_gt = distance_transform_edt(~gt_mask.astype(bool))
        dt_final = distance_transform_edt(~final_pred_mask)
        
        # H) Customize the prediction distance transform: assign high penalty in unassigned regions.
        if self.penalty_constant is None:
            penalty_value = dt_final.max()
        else:
            penalty_value = self.penalty_constant
        dt_final_custom = dt_final.copy()
        dt_final_custom[unassigned_pred_mask] = penalty_value
        
        # Compute pixel-wise squared error.
        error = (gt_mask.astype(np.float32) - pred_proba)**2
        # Weight error with the squared distance transforms.
        weight = dt_gt+ dt_final_custom #dt_gt**2 + dt_final_custom**2
        
        loss_map = error * weight
        loss = loss_map.mean()
        
        if debug:
            return loss, loss_map, dt_gt, dt_final_custom
        else:
            return loss, loss_map

    def compute_loss_field(self, gt_mask_batch: np.ndarray, pred_proba_batch: np.ndarray, debug=False):
        """
        Compute the loss field for a batch.
        Returns loss_field of shape (B, H, W) and, if debug=True, also dt_gt and dt_final_custom for the first image.
        """
        batch = gt_mask_batch.shape[0]
        loss_field = np.zeros_like(gt_mask_batch, dtype=np.float32)
        extra_debug = None
        for i in range(batch):
            if debug and i == 0:
                loss_val, loss_map, dt_gt, dt_final_custom = self._procrustes_loss(gt_mask_batch[i], pred_proba_batch[i], debug=True)
                extra_debug = {'dt_gt': dt_gt, 'dt_final_custom': dt_final_custom}
                loss_field[i] = loss_map
            else:
                loss_val, loss_map = self._procrustes_loss(gt_mask_batch[i], pred_proba_batch[i])
                loss_field[i] = loss_map
        return loss_field, extra_debug

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Compute the loss.
        
        Parameters:
          pred: Tensor of shape [B, 1, H, W] with predicted probabilities.
          target: Tensor of shape [B, 1, H, W] with binary ground truth (0: background, 1: AR).
        
        Returns:
          - loss: A scalar torch.Tensor with the mean loss.
          - if debug=True, returns a dict containing extra debug info.
        """
        # Convert to numpy arrays; remove channel dimension.
        pred_np = pred.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
        target_np = target.detach().cpu().numpy()[:, 0, ...]  # shape: [B, H, W]
        
        loss_field, extra_debug = self.compute_loss_field(target_np, pred_np, debug=debug)
        loss_value = loss_field.mean()
        
        loss_tensor = torch.tensor(loss_value, device=pred.device)
        if debug:
            debug_info = {
                'loss_field': torch.tensor(loss_field, device=pred.device),
                'dt_gt': torch.tensor(extra_debug['dt_gt'], device=pred.device),
                'dt_final_custom': torch.tensor(extra_debug['dt_final_custom'], device=pred.device)
            }
            return loss_tensor, debug_info
        else:
            return loss_tensor