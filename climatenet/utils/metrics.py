# import numpy as np
# import torch

# def get_iou_perClass(confM):
#     """
#     Takes a confusion matrix confM and returns the IoU per class
#     """
#     unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
#     iouPerClass = np.zeros(3)
#     for i in range(0,3):
#         if unionPerClass[i] == 0:
#             iouPerClass[i] = 1
#         else:
#             iouPerClass[i] = confM.diagonal()[i] / unionPerClass[i]
#     return iouPerClass
        
# def get_cm(pred, gt, n_classes=3):
#     cm = np.zeros((n_classes, n_classes))
#     for i in range(len(pred)):
#         pred_tmp = pred[i].int()
#         gt_tmp = gt[i].int()

#         for actual in range(n_classes):
#             for predicted in range(n_classes):
#                 is_actual = torch.eq(gt_tmp, actual)
#                 is_pred = torch.eq(pred_tmp, predicted)
#                 cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))
            
#     return cm



import numpy as np
import torch

def get_iou_perClass(confM, n_classes=3):
    """
    Takes a confusion matrix `confM` and returns the IoU per class.
    
    Args:
        confM (np.ndarray): A confusion matrix of shape (n_classes, n_classes).
        n_classes (int): Number of classes.
        
    Returns:
        iouPerClass (np.ndarray): Array of IoU values for each class.
    """
    # Compute union for each class:
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - np.diag(confM)
    iouPerClass = np.zeros(n_classes)
    for i in range(n_classes):
        if unionPerClass[i] == 0:
            iouPerClass[i] = 1
        else:
            iouPerClass[i] = confM[i, i] / unionPerClass[i]
    return iouPerClass

def get_cm(pred, gt, n_classes=3):
    """
    Computes a confusion matrix for the predictions.
    
    Args:
        pred (torch.Tensor): Predicted labels of shape [B, H, W].
        gt (torch.Tensor): Ground truth labels of shape [B, H, W].
        n_classes (int): Number of classes.
        
    Returns:
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes).
    """
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        gt_tmp = gt[i].int()
        for actual in range(n_classes):
            for predicted in range(n_classes):
                # Compute how many pixels are both actual==actual and predicted==predicted.
                is_actual = (gt_tmp == actual)
                is_pred = (pred_tmp == predicted)
                cm[actual, predicted] += (is_actual & is_pred).sum().item()
    return cm
