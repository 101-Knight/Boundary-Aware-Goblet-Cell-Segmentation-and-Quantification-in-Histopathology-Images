import copy

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

class MCCLModel(nn.Module):
    def __init__(self):

        super().__init__()

        self.student = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        self.teacher = copy.deepcopy(
            self.student
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward_student(self, x):

        return self.student(x)

    def forward_teacher(self, x):

        with torch.no_grad():
            return self.teacher(x)

    @torch.no_grad()
    def update_teacher(
        self,
        ema_decay=0.99
    ):

        student_dict = (
            self.student.state_dict()
        )

        teacher_dict = (
            self.teacher.state_dict()
        )

        for k in teacher_dict.keys():

            teacher_dict[k] = (
                ema_decay *
                teacher_dict[k]
                +
                (1.0 - ema_decay) *
                student_dict[k]
            )

        self.teacher.load_state_dict(
            teacher_dict
        )