from typing import Union, Tuple

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SmpUnet(nn.Module):
    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    _ACTIVATION_TABLE = {
        ACTIVATION_SIGMOID: "sigmoid",
        ACTIVATION_SOFTMAX: "softmax2d"
    }

    # ---------------------------------
    # ---- Input mode - only input
    INPUT_MODE_X = 0
    # ---- Input mode - both input and target
    INPUT_MODE_XY = 1
    # ---- Expected model input for each stage (0 - training, 1 - validation, 2 - test)
    STAGE_MODE = [INPUT_MODE_X, INPUT_MODE_X, INPUT_MODE_X]

    # ---------------------------------

    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 3,
                 backbone: str = "resnet18",
                 activation: str = ACTIVATION_SIGMOID,
                 **kwargs):
        super(SmpUnet, self).__init__()

        # ---- Here need to change from the naming convention used
        # ---- in this library and the smp.
        # ---- sigmoid -> sigmoid
        # ---- softmax -> softmax2d
        activation = activation if activation is None else self._ACTIVATION_TABLE[activation]
        if 'encoder_weights' not in kwargs:
            kwargs['encoder_weights'] = 'imagenet'

        self.stage = 0
        self.smp_net = smp.Unet(encoder_name=backbone,
                                classes=num_classes,
                                in_channels=in_channels,
                                activation=activation,
                                **kwargs)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.smp_net(x)
        return x

    def set_stage(self, stage_id: int):
        if stage_id in [0, 1, 2]:
            self.stage = stage_id
