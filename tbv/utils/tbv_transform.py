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

#!/usr/bin/python3


"""
Provides a set of Pytorch transforms that use OpenCV instead of PIL (Pytorch default)
for image manipulation.

The data augmentation objects symmetrically perturb both the image and map data.

Pair is for (sensor 3ch, map 3ch)
Triplet is for (sensor 3ch, map 3ch, labelmap 1ch)

Reference: See https://github.com/hszhao/semseg/blob/master/util/transform.py
"""

import collections
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numbers
import numpy as np
import torch


class ComposePair(object):
    """
    Composes transforms together into a chain of operations
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert pair of HWC images (numpy arrays) into CHW tensors"""
        for t in self.transforms:
            image1, image2 = t(image1, image2)
        return image1, image2


class ComposeTriplet(object):
    """
    Composes transforms together into a chain of operations
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform series of transforms sequentially to 3 inputs, preserving pixel-to-pixel alignment.

        Args:
            image1: array of shape (H,W,C) representing sensor image.
            image2: array of shape (H,W,C) representing HD map rendering.
            labelmap: array of shape (H,W) representing semantic segmentation label map.

        Returns:
            image1: tensor of shape (C,H,W) representing sensor_img, after normalization
                and pre-processing.
            image2: tensor of shape (C,H,W) representing HD map rendering, after normalization
                and pre-processing.
            labelmap: tensor of shape (C2,H,W), where C2 is the number of stacked binary masks.
        """
        for t in self.transforms:
            image1, image2, labelmap = t(image1, image2, labelmap)
        return image1, image2, labelmap


class CropBottomSquarePair:
    """Crop a square from the image, from the bottom of the image"""

    def __init__(self):
        """ """
        pass

    def __call__(self, image1: np.ndarray, image2: np.ndarray):
        """ """
        h, w = image1.shape[:2]
        assert h > w, "Only valid for portrait-mode photos"
        return image1[-w:, :], image2[-w:, :]


class CropBottomSquareTriplet:
    """Crop a square from the image, from the bottom of the image"""

    def __init__(self):
        """ """
        pass

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image1: HWC
            image2: HWC
            labelmap: HW
        """
        h, w = image1.shape[:2]
        assert h > w, "Only valid for portrait-mode photos"
        return image1[-w:, :], image2[-w:, :], labelmap[-w:, :]


class ToTensorPair(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray [eg: data read by cv2.imread()].\n")
        if image1.ndim not in [2, 3]:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")
        if len(image1.shape) == 2:
            image1 = np.expand_dims(image1, axis=2)

        if image2.ndim not in [2, 3]:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")
        if len(image2.shape) == 2:
            image2 = np.expand_dims(image2, axis=2)

        # convert from HWC to CHW for collate/batching into NCHW
        image1 = torch.from_numpy(image1.transpose((2, 0, 1)))
        if not isinstance(image1, torch.FloatTensor):
            image1 = image1.float()

        image2 = torch.from_numpy(image2.transpose((2, 0, 1)))
        if not isinstance(image2, torch.FloatTensor):
            image2 = image2.float()

        return image1, image2


class ToTensorTriplet(object):
    def __init__(self, grayscale_labelmap: bool):
        """ """
        self.grayscale_labelmap = grayscale_labelmap

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts HWC to CHW

        Args:
            image1: HWC array
            image2: HWC array
            labelmap: HW array

        Returns:
            image1: CHW torch.FloatTensor (3,H,W)
            image2: CHW torch.FloatTensor (3,H,W)
            labelmap: CHW torch.FloatTensor (N,H,W)
        """
        if (
            not isinstance(image1, np.ndarray)
            or not isinstance(image2, np.ndarray)
            or not isinstance(labelmap, np.ndarray)
        ):
            raise RuntimeError(
                "tbv_transform.ToTensorTriplet() only handle np.ndarray [eg: data read by cv2.imread()].\n"
            )

        if (image1.ndim != 3) or (image2.ndim != 3):
            raise RuntimeError("tbv_transform.ToTensor() wrong dimensions.\n")

        if self.grayscale_labelmap and (labelmap.ndim != 2):
            # should be grayscale
            raise RuntimeError("tbv_transform.ToTensor() wrong dimensions.\n")

        elif not self.grayscale_labelmap and (labelmap.ndim != 3):
            # should be 3-channel
            raise RuntimeError("tbv_transform.ToTensor() wrong dimensions.\n")

        # convert from HWC to CHW for collate/batching into NCHW
        image1 = torch.from_numpy(image1.transpose((2, 0, 1)))
        if not isinstance(image1, torch.FloatTensor):
            image1 = image1.float()

        image2 = torch.from_numpy(image2.transpose((2, 0, 1)))
        if not isinstance(image2, torch.FloatTensor):
            image2 = image2.float()

        if self.grayscale_labelmap:
            labelmap = torch.from_numpy(labelmap).unsqueeze(0)
        else:
            labelmap = torch.from_numpy(labelmap.transpose((2, 0, 1)))
        if not isinstance(labelmap, torch.FloatTensor):
            labelmap = labelmap.float()

        return image1, image2, labelmap


class NormalizePair(object):
    """
    Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    """

    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: check if better to have separate mean/std for map channels"""
        if self.std is None:
            for t, m in zip(image1, self.mean):
                t.sub_(m)
            for t, m in zip(image2, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image1, self.mean, self.std):
                t.sub_(m).div_(s)
            for t, m, s in zip(image2, self.mean, self.std):
                t.sub_(m).div_(s)

        return image1, image2


class NormalizeTriplet(object):
    """
    Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    """

    def __init__(self, normalize_labelmap: bool, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.normalize_labelmap = normalize_labelmap
        self.mean = mean
        self.std = std

    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor, labelmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO: check if better to have separate mean/std for map channels

        Args:
            image1
            image2
            labelmap

        Returns:
            image1: (C,H,W)
            image2: (C,H,W)
            labelmap: (1,H,W)
        """
        if self.std is None:
            raise RuntimeError("Provide a valid standard deviation as `std` ")

        # normalize each channel separate by looping over C dim
        for t, m, s in zip(image1, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(image2, self.mean, self.std):
            t.sub_(m).div_(s)

        if self.normalize_labelmap:
            for t, m, s in zip(labelmap, self.mean, self.std):
                t.sub_(m).div_(s)

        # labelmap_mean = [32]
        # labelmap_std = [30]
        # # TODO: try with different std and mean
        # for t, m, s in zip(labelmap, labelmap_mean, labelmap_std):
        # 	t.sub_(m).div_(s)

        return image1, image2, labelmap


class BinarizeLabelMapTriplet(object):
    def __init__(self):
        """Taxonomy and ordering from seamseg"""
        self.CROSSWALK_ZEBRA_IDX = 35  # 'marking--crosswalk-zebra'
        self.MARKING_GENERAL_IDX = 16  # 'marking--general'
        self.BIKE_LANE_IDX = 5  # 'construction--flat--bike-lane'
        self.ROAD_IDX = 10  # 'construction--flat--road'
        self.CROSSWALK_PLAIN_IDX = 30  # 'construction--flat--crosswalk-plain'

    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor, labelmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a semantic segmentation labelmap (defined over N classes) to 5 stacked binary masks.

        Args:
            image1
            image2
            labelmap: tensor of shape (H,W)

        Returns:
            image1: tensor of shape (H,W,C)
            image2: tensor of shape (H,W,C)
            labelmap: tensor of shape (5, H, W), representing 5 stacked binary masks, corresponding
               to 5 semantic classes:zebra crosswalk, general lane marking, bike lane, road surface, and plain crosswalk
        """
        # now in CHW order
        _, h, w = image1.shape

        semantic_idxs = [
            self.CROSSWALK_ZEBRA_IDX,
            self.MARKING_GENERAL_IDX,
            self.BIKE_LANE_IDX,
            self.ROAD_IDX,
            self.CROSSWALK_PLAIN_IDX,
        ]

        num_masks = len(semantic_idxs)
        labelmap_masks = torch.zeros((num_masks, h, w))
        for i, semantic_idx in enumerate(semantic_idxs):
            labelmap_masks[i] = labelmap == semantic_idx

        labelmap = labelmap_masks
        return image1, image2, labelmap


class ResizePair(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        image1 = cv2.resize(image1, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image1, image2


class ResizeTriplet(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image1: array of shape (H,W,C) representing sensor image.
            image2: array of shape (H,W,C) representing HD map rendering.
            labelmap: array of shape (H,W) representing semantic segmentation label map.
        """
        image1 = cv2.resize(image1, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        labelmap = cv2.resize(labelmap, self.size[::-1], interpolation=cv2.INTER_NEAREST)

        return image1, image2, labelmap


class ResizeLabelMapTriplet(object):
    def __init__(self):
        pass

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize label map to be the same size as image1 and image2"""
        h, w, _ = image1.shape
        labelmap = cv2.resize(labelmap, (w, h), interpolation=cv2.INTER_NEAREST)

        return image1, image2, labelmap


class CropPair(object):
    """Crops the given ndarray image (H,W,C or H,W)."""

    def __init__(self, size, crop_type: str = "center", padding=None):
        """
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is made.
            crop_type: either 'rand' or 'center'
            padding:
        """
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif (
            isinstance(size, collections.Iterable)
            and len(size) == 2
            and isinstance(size[0], int)
            and isinstance(size[1], int)
            and size[0] > 0
            and size[1] > 0
        ):
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")
        if crop_type not in ["center", "rand"]:
            raise RuntimeError("crop type error: rand | center\n")
        self.crop_type = crop_type

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")

        self.map_padding = [0.0, 0.0, 0.0]  # color padded areas as pitch black

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crop both inputs to specified width and height, either via random or center crop"""
        h, w, c = image1.shape
        assert image1.shape == image2.shape

        # may need to pad image if input images are smaller than desired crop size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("tbv_transform.CropPair() need padding while padding argument is None\n"))
            image1 = cv2.copyMakeBorder(
                image1,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding,
            )
            image2 = cv2.copyMakeBorder(
                image2,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.map_padding,
            )

        # get shape again, may have changed if padded
        h, w, c = image1.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        image1 = image1[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image2 = image2[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        return image1, image2


class CropTriplet(object):
    """Crops the given ndarray image (H,W,C or H,W)."""

    def __init__(self, size, crop_type: str = "center", padding=None, grayscale_labelmap: bool = True):
        """
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is made.
            crop_type: either 'rand' or 'center'
            padding:
        """
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif (
            isinstance(size, collections.Iterable)
            and len(size) == 2
            and isinstance(size[0], int)
            and isinstance(size[1], int)
            and size[0] > 0
            and size[1] > 0
        ):
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")
        if crop_type not in ["center", "rand"]:
            raise RuntimeError("crop type error: rand | center\n")
        self.crop_type = crop_type

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")

        self.map_padding = [0.0, 0.0, 0.0]  # color padded areas as pitch black
        self.grayscale_labelmap = grayscale_labelmap

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        labelmap: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop both inputs to specified width and height, either via random or center crop.

        Args:
            image1: array of shape (H,W,C) representing sensor image.
            image2: array of shape (H,W,C) representing HD map rendering.
            labelmap: array of shape (H,W) representing semantic segmentation label map.
        
        Returns:
            image1: array of shape (crop_h, crop_w, C) representing sensor image.
            image2: array of shape (crop_h, crop_w, C) representing HD map rendering.
            labelmap: array of shape (crop_h, crop_w) representing semantic segmentation label map.
        """
        h, w, c = image1.shape
        assert image1.shape == image2.shape
        if self.grayscale_labelmap:
            assert image1.shape[:2] == labelmap.shape
        else:
            assert image1.shape == labelmap.shape  # 3-channel RGB labelmap for BEV

        # may need to pad image if input images are smaller than desired crop size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("tbv_transform.CropTriplet() need padding while padding argument is None\n"))
            image1 = cv2.copyMakeBorder(
                image1,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding,
            )
            image2 = cv2.copyMakeBorder(
                image2,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.map_padding,
            )
            labelmap = cv2.copyMakeBorder(
                labelmap,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.map_padding,
            )

        # get shape again, may have changed if padded
        h, w, c = image1.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        image1 = image1[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        image2 = image2[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        labelmap = labelmap[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        return image1, image2, labelmap


class RandRotatePair(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding: Tuple[float, float, float], p: float = 0.5):
        """
        Args:
            rotate:
            padding: R,G,B padding values
            p: probability of symmetrically applying random rotation to both inputs
        """
        assert isinstance(rotate, collections.Iterable) and len(rotate) == 2
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3

        if not all([isinstance(i, numbers.Number) for i in padding]):
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        self.padding = padding
        self.p = p
        self.map_padding = [0.0, 0.0, 0.0]  # color padded areas as pitch black

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w, c = image1.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image1 = cv2.warpAffine(
                image1, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding
            )
            image2 = cv2.warpAffine(
                image2,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.map_padding,
            )

        return image1, image2


class RandomHorizontalFlipPair(object):
    def __init__(self, p: float = 0.5) -> None:
        """p is the probability of applying a random horizontal flip to both inputs"""
        self.p = p

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
        return image1, image2


class RandomHorizontalFlipTriplet(object):
    def __init__(self, p: float = 0.5) -> None:
        """p is the probability of applying a random horizontal flip to both inputs"""
        self.p = p

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image1: array of shape (H,W,C) representing sensor image.
            image2: array of shape (H,W,C) representing HD map rendering.
            labelmap: array of shape (H,W) representing semantic segmentation label map.
        """
        if random.random() < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            labelmap = cv2.flip(labelmap, 1)
        return image1, image2, labelmap


class RandomVerticalFlipPair(object):
    def __init__(self, p: float = 0.5) -> None:
        """p is the probability of applying a random vertical flip to both inputs"""
        self.p = p

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
        return image1, image2


class RandomVerticalFlipTriplet(object):
    def __init__(self, p: float = 0.5) -> None:
        """p is the probability of applying a random vertical flip to both inputs"""
        self.p = p

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
            labelmap = cv2.flip(labelmap, 0)
        return image1, image2, labelmap


class RandomGaussianBlurPair(object):
    def __init__(self, radius: int = 5):
        """radius is Gaussian kernel size"""
        self.radius = radius

    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image1 = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
            # image2 = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image1, image2


class RandomGaussianBlurTriplet(object):
    def __init__(self, radius: int = 5):
        """radius is Gaussian kernel size"""
        self.radius = radius

    def __call__(
        self, image1: np.ndarray, image2: np.ndarray, labelmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image1 = cv2.GaussianBlur(image1, (self.radius, self.radius), 0)
            # image2 = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image1, image2, labelmap


class RandomModalityDropoutWithoutReplacementTriplet(object):
    def __init__(self, modalities_for_random_dropout: List[str], p: float = 0.5) -> None:
        """
        Args:
            modalities_for_random_dropout:
            p: probability of dropping out a modality (which could be ANY one of the
                requested modalities above)
        """
        possible_modalities = ["sensor", "semantics", "map"]
        assert all([m in possible_modalities for m in modalities_for_random_dropout])

        self.modalities_for_random_dropout = modalities_for_random_dropout
        self.p = p

    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor, labelmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """At most one modality will be dropped

        Args:
            image1: sensor image
            image2: map
            labelmap: semantic label map
        """
        # leave intact with (1-p) probability
        # with probability p, drop one of the modalities!
        if random.random() < self.p:

            modality_to_dropout = np.random.choice(self.modalities_for_random_dropout)

            if modality_to_dropout == "sensor":
                image1 = torch.zeros_like(image1).type(image1.dtype)

            elif modality_to_dropout == "map":
                image2 = torch.zeros_like(image2).type(image2.dtype)

            elif modality_to_dropout == "semantics":
                labelmap = torch.zeros_like(labelmap).type(labelmap.dtype)

        return image1, image2, labelmap


class RandomMapDropoutTriplet(object):
    def __init__(self, p: float = 0.5) -> None:
        """ """
        self.p = p

    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor, labelmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        if random.random() < self.p:
            # image2 is the map
            image2 = torch.zeros_like(image2).type(image2.dtype)

        return image1, image2, labelmap


class RandomSemanticsDropoutTriplet(object):
    def __init__(self, p: float = 0.5) -> None:
        """ """
        self.p = p

    def __call__(
        self, image1: torch.Tensor, image2: torch.Tensor, labelmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        if random.random() < self.p:
            labelmap = torch.zeros_like(labelmap).type(labelmap.dtype)

        return image1, image2, labelmap


def unnormalize_img(
    input: torch.Tensor, mean: Tuple[float, float, float], std: Optional[Tuple[float, float, float]] = None
):
    """Pass in by reference Torch tensor, and normalize its values.
    Args:
        input: Torch tensor of shape (3,M,N), must be in this order, and of type float (necessary).
        mean: mean values for each RGB channel.
        std: standard deviation values for each RGB channel.
    """
    if std is None:
        for t, m in zip(input, mean):
            t.add_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.mul_(s).add_(m)
