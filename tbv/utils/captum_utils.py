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

"""
Utility that applies Captum to the image classification task, to understand which pixels and regions
contribute to the labeling of a particular class. GradCAM and GuidedGradCAM are used.

See reference:
    https://captum.ai/tutorials/Segmentation_Interpret
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam, LayerGradCam, LayerAttribution
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std

from tbv.utils.tbv_transform import unnormalize_img


def get_gradcam_results(
    log_id: str,
    timestamp: int,
    args,
    rgb_img: np.ndarray,
    map_rgb_img: np.ndarray,
    normalized_inputs: List[torch.Tensor],
    model: nn.Module,
    feature_module: nn.Module,
    target_layer_names: List[str],
    use_cuda: bool,
) -> None:
    """Analyze model output by visualizing image saliency via GradCAM.

    Args:
        log_id:
        timestamp:
        args:
        rgb_img: array of shape ()
        map_rgb_img: array of shape ()
        normalized_inputs: list of tensors, each of shape (), representing normalized ...
        model:
        feature_module:
        target_layer_names:
        use_cuda: whether to use the GPU.
    """
    sensor_img, map_img, semantic_img = normalized_inputs
    sensor_img = sensor_img.requires_grad_(True)
    map_img = map_img.requires_grad_(True)
    semantic_img = semantic_img.requires_grad_(True)

    if args.model_name in ["EarlyFusionCEResnetWLabelMap"]:
        inputs = (sensor_img, map_img, semantic_img)

    elif args.model_name in ["EarlyFusionCEResnet"] and set(args.fusion_modalities) == set(["semantics", "map"]):
        inputs = (semantic_img, map_img)

    elif args.model_name in ["SingleModalityCEResnet"] and args.fusion_modalities == ["map"]:
        inputs = map_img

    else:
        raise RuntimeError

    use_guided_gradcam = True
    if use_guided_gradcam:
        lgc = GuidedGradCam(model=model, layer=feature_module)
        gc_attr = lgc.attribute(inputs=inputs, target=0, attribute_to_layer_input=True)

        if args.model_name in ["EarlyFusionCEResnetWLabelMap"]:
            sensor_heatmap = gc_attr[0][0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)
            map_heatmap = gc_attr[1][0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)
            labelmap_heatmap = gc_attr[2][0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)

        elif args.model_name in ["EarlyFusionCEResnet"]:
            labelmap_heatmap = gc_attr[0][0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)
            map_heatmap = gc_attr[1][0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)

            mean, std = get_imagenet_mean_std()
            semantic_rgb_img = semantic_img.squeeze().clone().detach()
            unnormalize_img(semantic_rgb_img, mean, std)
            semantic_rgb_img = semantic_rgb_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

            sensor_heatmap = np.ones_like(labelmap_heatmap)  # (224, 224, 1)

        elif args.model_name in ["SingleModalityCEResnet"] and args.fusion_modalities == ["map"]:
            map_heatmap = gc_attr[0].permute(1, 2, 0).detach().cpu().numpy().sum(axis=2, keepdims=True)

            sensor_heatmap = np.ones_like(map_heatmap)
            labelmap_heatmap = np.ones_like(map_heatmap)
            semantic_rgb_img = rgb_img

    else:
        lgc = LayerGradCam(forward_func=model, layer=feature_module)
        gc_attr = lgc.attribute(inputs=inputs, target=0, attribute_to_layer_input=True)
        # We'd like to understand this better by overlaying it on the original image. In order to do
        # this, we need to upsample the layer GradCAM attributions to match the input size.
        upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, rgb_img.shape[:2])
        heatmap = upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy()

        sensor_heatmap = heatmap.copy()
        map_heatmap = heatmap.copy()

    # We can now visualize the Layer GradCAM attributions using the Captum visualization utilities.
    # viz.visualize_image_attr(attr=gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(), sign="all")
    # fig = viz.visualize_image_attr(attr=gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(), sign="positive")

    # plt.savefig(save_fpath, dpi=400)
    # #plt.show()
    # plt.close('all')

    # fig, ax = viz.visualize_image_attr_multiple(
    #     attr=heatmap,
    #     original_image=rgb_img,
    #     signs=["all", "positive", "negative"],
    #     methods=["original_image", "blended_heat_map", "blended_heat_map"]
    # )

    num_rows = 2  # 1
    num_columns = 3  # 2

    default_heatmap_start_idx = 1

    fig = plt.figure()

    if num_rows == 2:
        plt.subplot(num_rows, num_columns, 1)
        plt.imshow(rgb_img)
        plt.axis("off")

        plt.subplot(num_rows, num_columns, 2)
        plt.imshow(map_rgb_img)
        plt.axis("off")

        plt.subplot(num_rows, num_columns, 3)
        plt.imshow(semantic_rgb_img)
        plt.axis("off")
        default_heatmap_start_idx = 4

    plt.subplot(num_rows, num_columns, default_heatmap_start_idx)
    plt.axis("off")
    render_unnormalized_heatmap_on_img(rgb_img, sensor_heatmap)

    plt.subplot(num_rows, num_columns, default_heatmap_start_idx + 1)
    plt.axis("off")
    render_unnormalized_heatmap_on_img(map_rgb_img, map_heatmap)

    if num_columns == 3:
        plt.subplot(num_rows, num_columns, default_heatmap_start_idx + 2)
        plt.axis("off")
        render_unnormalized_heatmap_on_img(semantic_rgb_img, labelmap_heatmap)

    # plt.show()
    save_dir = "guided_captum_redheatmap_BEV_maponly"  # output_pt8overlay'
    os.makedirs(save_dir, exist_ok=True)
    save_fpath = f"{save_dir}/{log_id}_{timestamp}.jpg"
    plt.savefig(save_fpath, dpi=400)
    # plt.show()
    plt.axis("off")
    plt.close("all")

    # labelmap_heatmap = viz._normalize_image_attr(
    #     attr=labelmap_heatmap,
    #     sign="positive",
    #     outlier_perc=2,
    # )
    # plt.imshow(labelmap_heatmap)
    # plt.show()
    # plt.close('all')


def render_unnormalized_heatmap_on_img(rgb_img: np.ndarray, heatmap: np.ndarray, cmap: str = "Reds") -> None:
    """Color areas of an RGB image according to a heatmap, with bright red indicating high heatmap values.

    We use the "Reds" matplotlib colormap by default, but other possible colormaps are reasonable,
    such as "Greens" or "gray".

    Args:
        rgb_img: array of shape (H,W,C) representing ...
        heatmap: array of shape (H,W) representing ...
    """
    heatmap = viz._normalize_image_attr(
        attr=heatmap,
        sign="positive",
        outlier_perc=2,
    )
    heatmap = 1 - heatmap
    alpha_overlay = 0.8
    # alpha_overlay: float = 0.5

    vmin, vmax = 0, 1

    plt.imshow(np.mean(rgb_img, axis=2), cmap="gray")
    heat_map = plt.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay)
