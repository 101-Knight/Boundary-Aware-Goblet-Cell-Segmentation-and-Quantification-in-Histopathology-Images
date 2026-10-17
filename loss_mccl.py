import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

dice_loss = smp.losses.DiceLoss(
mode="binary"
)

bce_loss = nn.BCEWithLogitsLoss()

mse_loss = nn.MSELoss()

def supervised_loss(
    pred,
    target
    ):

    return (
        0.5 *
        bce_loss(pred, target)
        +
        0.5 *
        dice_loss(pred, target)
    )

def consistency_loss(
    student_pred,
    teacher_pred
    ):

    student_pred = torch.sigmoid(
        student_pred
    )

    teacher_pred = torch.sigmoid(
        teacher_pred
    )

    return mse_loss(
        student_pred,
        teacher_pred
    )