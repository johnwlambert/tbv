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

from typing import Optional

import torch
from torch import nn
from torchvision import models


def get_resnet_feature_dim(num_layers: int, late_fusion: bool, late_fusion_operator: Optional[str] = None) -> int:
    """

    Args:
        num_layers: number of desired ResNet layers, i.e. network depth.
        late_fusion: whether the network architecture will use late fusion of features.
        late_fusion_operator: either 'concat' or 'elementwise_add' or 'elementwise_multiply'
    """
    if num_layers in [18, 34]:
        feature_dim = 512 * 1  # expansion factor 1
    elif num_layers in [50, 101, 152]:
        feature_dim = 512 * 4  # expansion factor 4, so get 2048
    else:
        raise RuntimeError("unexplored feature dim")

    # late fusion with 'elementwise_add', 'elementwise_multiply' wont increase feature dim
    if late_fusion and late_fusion_operator == "concat":
        feature_dim *= 2  # concatenating embeddings together before FC

    return feature_dim


def get_vanilla_resnet_model(num_layers: int, pretrained: bool) -> nn.Module:
    """
    Args:
        num_layers: number of desired ResNet layers, i.e. network depth.
        pretrained: whether to load Imagenet-pretrained network weights.

    Returns:
        resnet: ResNet model architecture and weights.
    """
    assert num_layers in [18, 34, 50, 101, 152]
    if num_layers == 18:
        resnet = models.resnet18(pretrained=pretrained)
    elif num_layers == 34:
        resnet = models.resnet34(pretrained=pretrained)
    elif num_layers == 50:
        resnet = models.resnet50(pretrained=pretrained)
    elif num_layers == 101:
        resnet = models.resnet101(pretrained=pretrained)
    elif num_layers == 152:
        resnet = models.resnet152(pretrained=pretrained)
    return resnet


class ResNetConvBackbone(nn.Module):
    """ """

    def __init__(self, num_layers: int, pretrained: bool) -> None:
        """ """
        super(ResNetConvBackbone, self).__init__()
        self.resnet = get_vanilla_resnet_model(num_layers, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through conv layers.

        Args:
            x: tensor of shape (N,C,H,W)

        Returns:
            x: tensor of shape (N, 512)
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # pool (N, 512, 7, 7) into (N, 512, 1, 1)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
