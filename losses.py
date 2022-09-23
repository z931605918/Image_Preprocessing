from __future__ import print_function, division
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def MSELoss(pred,label):
    loss_fn = nn.MSELoss(reduce=False, size_average=True)
    EPE_map = loss_fn(pred, label)


    return EPE_map.mean()

def precies_recall(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    with torch.no_grad():
        i_flat = prediction.view(-1).cpu().numpy()
       # i_flat = np.where(i_flat < 0.5, 0, 1)
        t_flat = target.view(-1).cpu().numpy()

        intersection = (i_flat * t_flat).sum()

    return (intersection + 0.0001) / (((t_flat)**2).sum() + 0.0001), (intersection + 0.0001) / (((i_flat)**2).sum() + 0.0001)
def correlation(pre,target):

    pre_flat=pre.view(-1)
    target_flat=target.view(-1)

    dot_matr=(pre_flat*target_flat).sum()
    denominator=torch.sqrt((pre_flat**2).sum())*torch.sqrt((target_flat**2).sum())+0.00001

    return dot_matr/(denominator+0.000001)
class Correlation_loss(torch.nn.Module):
    def __init__(self):
        super(Correlation_loss,self).__init__()

    def forward(self,pre,target):

        return 1-correlation(pre,target)



