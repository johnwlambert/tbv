"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

import torch
from torch import nn

import tbv.models.resnet_factory as resnet_factory
from tbv.models.resnet_factory import ResNetConvBackbone


class LateFusionSiameseCEResnet(nn.Module):
    """Late Fusion Model"""

    def __init__(
        self, num_layers: int, pretrained: bool, num_classes: int, late_fusion_operator: str = "concat"
    ) -> None:
        """
        Args:
            num_layers: number of desired ResNet layers, i.e. network depth.
            pretrained: whether to load Imagenet-pretrained network weights.
            num_classes:
            late_fusion_operator: either 'concat' or 'elementwise_add' or 'elementwise_multiply'
        """
        super(LateFusionSiameseCEResnet, self).__init__()

        if late_fusion_operator not in ["concat", "elementwise_add", "elementwise_multiply"]:
            raise RuntimeError("Unknown late fusion operator")
        self.late_fusion_operator = late_fusion_operator

        self.net = ResNetConvBackbone(num_layers, pretrained)
        assert num_classes > 1

        feature_dim = resnet_factory.get_resnet_feature_dim(
            num_layers, late_fusion=True, late_fusion_operator=late_fusion_operator
        )
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, xstar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (N,C,H,W) representing batch of images.
            xstar: tensor of shape (N,C,H,W) representing batch of corresponding maps.

        Returns:
            logits: tensor of shape (N,K) where K is the number of classes.
        """
        x = self.net(x)
        xstar = self.net(xstar)

        if self.late_fusion_operator == "concat":
            feat = torch.cat([x, xstar], dim=1)

        elif self.late_fusion_operator == "elementwise_add":
            feat = x + xstar

        elif self.late_fusion_operator == "elementwise_multiply":
            feat = torch.mul(x, xstar)

        logits = self.fc(feat)
        return logits
