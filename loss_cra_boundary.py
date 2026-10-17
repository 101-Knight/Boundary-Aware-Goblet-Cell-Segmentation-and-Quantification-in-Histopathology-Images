import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CRABoundaryLoss(nn.Module):
    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = smp.losses.DiceLoss(
            mode="binary"
        )

    def forward(
        self,
        seg_pred,
        seg_gt,
        boundary_pred,
        boundary_gt
    ):

        seg_loss = (

            0.5 * self.bce(
                seg_pred,
                seg_gt
            )

            +

            0.5 * self.dice(
                seg_pred,
                seg_gt
            )
        )

        boundary_loss = self.bce(
            boundary_pred,
            boundary_gt
        )

        total_loss = (

            seg_loss

            +

            0.3 * boundary_loss
        )

        return total_loss
