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

from typing import List, Tuple

import torch
from torch import nn

import tbv.models.resnet_factory as resnet_factory

NUM_RGB_CHANNELS = 3
NUM_MAP_CHANNELS = 3
NUM_LABELMAP_EGOVIEW_CHANNELS = 5  # semantic, as per-category binary masks.
NUM_LABELMAP_BEV_CHANNELS = 3  # semantic, as RGB colormap on categories.


class EarlyFusionCEResnet(nn.Module):
    """Early Fusion Model for image + map data (pair) w/ a single prediction head.

    Accepts 2 args in forward()
    """

    def __init__(
        self, num_layers: int, pretrained: bool, num_classes: int, viewpoint: str, fusion_modalities: List[str]
    ) -> None:
        """
        Args:
            num_layers: number of desired ResNet layers, i.e. network depth.
            pretrained: whether to load Imagenet-pretrained network weights.
            num_classes:
            viewpoint: "egoview" or "bev"
        """
        super(EarlyFusionCEResnet, self).__init__()
        assert num_classes > 1

        resnet = resnet_factory.get_vanilla_resnet_model(num_layers, pretrained)
        self.inplanes = 64

        if viewpoint == "egoview":
            num_labelmap_channels = NUM_LABELMAP_EGOVIEW_CHANNELS
        else:
            num_labelmap_channels = NUM_LABELMAP_BEV_CHANNELS

        if set(fusion_modalities) == set(["sensor", "map"]):
            num_inchannels = NUM_RGB_CHANNELS + NUM_MAP_CHANNELS
        elif set(fusion_modalities) == set(["semantics", "map"]):
            num_inchannels = num_labelmap_channels + NUM_MAP_CHANNELS
        else:
            raise RuntimeError("Incompatible fusion modalities")

        # resnet with more channels in first layer (6 instead of 3)
        self.conv1 = nn.Conv2d(num_inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = resnet
        feature_dim = resnet_factory.get_resnet_feature_dim(num_layers, late_fusion=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, xstar: torch.Tensor):
        """Concat channels of 2 inputs immediately

        Args:
            x: tensor of shape (N,C,H,W) representing batch of images.
            xstar: tensor of shape (N,C,H,W) representing batch of corresponding maps.

        Returns:
            logits: tensor of shape (N,K) where K is the number of classes.
        """
        xxs = torch.cat([x, xstar], dim=1)
        xxs = self.conv1(xxs)

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


class EarlyFusionCEResnetWLabelMap(nn.Module):
    """Early Fusion Model for image + map + semantics data (triplet) w/ a single prediction head.

    Accepts 3 args in forward().
    """

    def __init__(self, num_layers: int, pretrained: bool, num_classes: int, viewpoint: str) -> None:
        """
        Args:
            num_layers: number of desired ResNet layers, i.e. network depth.
            pretrained: whether to load Imagenet-pretrained network weights.
            num_classes:
            viewpoint: "egoview" or "bev"
        """
        super(EarlyFusionCEResnetWLabelMap, self).__init__()
        assert num_classes > 1

        resnet = resnet_factory.get_vanilla_resnet_model(num_layers, pretrained)
        self.inplanes = 64

        if viewpoint == "egoview":
            num_labelmap_channels = NUM_LABELMAP_EGOVIEW_CHANNELS
        else:
            num_labelmap_channels = NUM_LABELMAP_BEV_CHANNELS

        num_inchannels = NUM_RGB_CHANNELS + num_labelmap_channels + NUM_MAP_CHANNELS
        # resnet with more channels in first layer (9 instead of 3 or 6)
        self.conv1 = nn.Conv2d(num_inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = resnet
        feature_dim = resnet_factory.get_resnet_feature_dim(num_layers, late_fusion=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, xstar: torch.Tensor, labelmap: torch.Tensor):
        """Concat channels of 3 inputs immediately

        Args:
            x: tensor of shape (N,C,H,W) representing batch of images.
            xstar: tensor of shape (N,C,H,W) representing batch of corresponding maps.
            label map: tensor of shape ... representing batch of corresponding semantic segmentation label maps.

        Returns:
            logits: tensor of shape (N,K) where K is the number of classes.
        """
        xxs = torch.cat([x, xstar, labelmap], dim=1)
        xxs = self.conv1(xxs)

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


class EarlyFusionTwoHeadResnet(nn.Module):
    """
    Early-fusion architecture w/ ResNet backbone, but branches into two heads at
    the end to predict `is_match` and specific type of change category separately.
    """

    def __init__(self, num_layers: int, pretrained: bool, num_classes: int) -> None:
        super(EarlyFusionTwoHeadResnet, self).__init__()
        assert num_classes > 1

        resnet = resnet_factory.get_vanilla_resnet_model(num_layers, pretrained)
        self.inplanes = 64

        # resnet with more channels in first layer (6 instead of 3)
        self.conv1 = nn.Conv2d(3 + 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = resnet
        feature_dim = resnet_factory.get_resnet_feature_dim(num_layers, late_fusion=False)
        self.fc_ismatch = nn.Linear(feature_dim, out_features=2)
        self.fc_class = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        xstar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concat channels immediately

        Args:
            x: tensor of shape (N,C,H,W) representing batch of images.
            xstar: tensor of shape (N,C,H,W) representing batch of corresponding maps.

        Returns:
            is_match_logits: tensor of shape (N,2) for the `no_match` and `match` categories.
            class_logits: tensor of shape (N,K) where K is the number of classes.
        """
        xxs = torch.cat([x, xstar], dim=1)
        xxs = self.conv1(xxs)

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

        xxs_copy = xxs.clone()
        is_match_logits = self.fc_ismatch(xxs_copy)
        class_logits = self.fc_class(xxs)

        return is_match_logits, class_logits
