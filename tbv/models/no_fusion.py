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
import torch.nn as nn

from tbv.models.resnet_factory import get_vanilla_resnet_model, get_resnet_feature_dim, ResNetConvBackbone


class SingleModalityCEResnet(nn.Module):
    """See only one modality"""

    def __init__(self, num_layers: int, pretrained: bool, num_classes: int) -> None:
        super(SingleModalityCEResnet, self).__init__()
        assert num_classes > 1

        self.net = ResNetConvBackbone(num_layers, pretrained)
        feature_dim = get_resnet_feature_dim(num_layers, late_fusion=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sensor or map stream only as x"""
        x = self.net(x)
        logits = self.fc(x)

        # must not return a scalar for dataparallel to work
        return logits


class SingleModalityLabelmapCEResnet(nn.Module):
    """Early Fusion Model for image + map data (pair)"""

    def __init__(self, num_layers: int, pretrained: bool, num_classes: int) -> None:
        super(SingleModalityLabelmapCEResnet, self).__init__()
        assert num_classes > 1

        self.inplanes = 64

        num_inchannels = 5
        # resnet with more channels in first layer (5 instead of 3)
        self.conv1 = nn.Conv2d(num_inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = get_vanilla_resnet_model(num_layers, pretrained)
        feature_dim = get_resnet_feature_dim(num_layers, late_fusion=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concat channels immediately"""
        xxs = self.conv1(x)

        xxs = self.resnet.bn1(xxs)
        xxs = self.resnet.relu(xxs)
        xxs = self.resnet.maxpool(xxs)

        xxs = self.resnet.layer1(xxs)
        xxs = self.resnet.layer2(xxs)
        xxs = self.resnet.layer3(xxs)
        xxs = self.resnet.layer4(xxs)
        # pool (N, 512, 7, 7) into (N, 512, 1, 1)
        xxs = self.resnet.avgpool(xxs)
        xxs = torch.flatten(xxs, 1)

        logits = self.fc(xxs)
        return logits
