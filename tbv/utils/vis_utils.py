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

import os
from pathlib import Path
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mseg.utils.cv2_utils import add_text_cv2

from tbv.utils.z1_egovehicle_mask_utils import get_z1_ring_front_center_mask


def hstack_imgs_w_border(imgs: List[np.ndarray], border_sz: int = 30) -> np.ndarray:
    """Stack images horizontally, with white border between them.

    Args:
        imgs: list of N images, each an array with identical shape (H,W,C) representing an RGB image.
        border_sz: width of border, in pixels.

    Returns:
        hstack_img: array of shape (H,W * N, C) representing horizontally stacked images.
    """
    num_imgs = len(imgs)
    if len(imgs) < 1:
        raise RuntimeError

    img_h, img_w, _ = imgs[0].shape

    hstack_img = np.ones((img_h, img_w * num_imgs + (num_imgs - 1) * border_sz, 3), dtype=np.uint8) * 255

    for i in range(num_imgs):
        h_start = i * img_w + (i) * border_sz
        h_end = h_start + img_w
        hstack_img[:, h_start:h_end, :] = imgs[i]

    return hstack_img


def form_vstacked_imgs(img_list: List[np.ndarray], border_sz: int = 30) -> np.ndarray:
    """Concatenate images along a vertical axis and save them.
    
    Args:
        img_list: list of Numpy arrays representing different RGB visualizations of same image,
            must all be of same shape
        hstack_save_fpath: string, representing file path

    Returns:
        hstack_img: Numpy array representing RGB image, containing vertically stacked images as tiles.
    """
    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # width and number of channels must match
    assert all(img.shape[1] == img_w for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)
    all_heights = [img.shape[0] for img in img_list]
    vstack_img = np.ones((sum(all_heights) + (num_imgs - 1) * border_sz, img_w, 3), dtype=np.uint8) * 255

    running_h = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_h
        end = start + h
        vstack_img[start:end, :, :] = img
        running_h += h
        running_h += border_sz

    return vstack_img


def blend_imgs_preserve_contrast(img1: np.ndarray, img2: np.ndarray, alpha1: float = 0.4):
    """
    Interpolate between (multiplying the images) and taking their weighted average

    Args:
        img1
        img2
        alpha1

    Returns:
        blended_img
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    blended_img2 = img1 * alpha1 + img2 * (1.0 - alpha1)

    # blended_img = 0.5*( (img1 + 100) + (img2) )
    blended_img1 = img1 * img2

    blended_img1 -= np.amin(blended_img1)
    blended_img1 /= np.amax(blended_img1)
    blended_img1 *= 255

    # blended_img2 = img1 * (img2 + 30)

    blended_img = 0.5 * (blended_img1 + blended_img2)

    blended_img = np.clip(blended_img, a_min=0, a_max=255)
    return blended_img.astype(np.uint8)


def blend_imgs(img1: np.ndarray, img2: np.ndarray, alpha1: float = 0.4):
    """
    Visualizes a single binary mask by coloring the region inside a binary mask
    as a specific color, and then blending it with an RGB image.

    Args:
        img: Numpy array, representing RGB image with values in the [0,255] range
        mask: Numpy integer array, with values in [0,1] representing mask region
        col: color, tuple of integers in [0,255] representing RGB values
        alpha: blending coefficient (higher alpha shows more of mask,
            lower alpha preserves original image)

    Returns:
        image: Numpy array, representing an RGB image, representing a blended image
            of original RGB image and specified colors in mask region.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    blended_img = img1 * alpha1 + img2 * (1.0 - alpha1)
    return blended_img.astype(np.uint8)


def remove_foreground_front_center(sensor_img: np.ndarray) -> np.ndarray:
    """

    Args:
        sensor_img

    Returns:
        sensor_img:
    """
    foreground_mask = get_z1_ring_front_center_mask()

    sensor_img = sensor_img.astype(np.float32)
    for i in range(3):
        sensor_img[:, :, i] *= 1 - foreground_mask.astype(np.float32)

    sensor_img = sensor_img.astype(np.uint8)
    return sensor_img


def save_egoview_sensor_map_semantics_triplet(
    is_match: bool,
    img_fname: str,
    sensor_img_fpath: str,
    map_img_fpath: str,
    labelmap_fpath: str,
    save_dir: str,
    render_text: bool,
) -> None:
    """

    Args:
        is_match:
        img_fname:
        sensor_img_fpath:
        map_img_fpath:
        labelmap_fpath:
        save_dir:
        render_text:
    """
    sensor_img = imageio.imread(sensor_img_fpath)
    no_change_img = imageio.imread(map_img_fpath)
    if Path(labelmap_fpath).exists():
        semantic_img = imageio.imread(labelmap_fpath)
    else:
        print(f"Label map file missing, so using dummy values instead: {labelmap_fpath}")
        h, w, c = sensor_img.shape
        semantic_img = np.zeros((h, w), dtype=np.uint8)

    # apply foregound mask to sensor_img

    sensor_img = remove_foreground_front_center(sensor_img)
    h, w, _ = sensor_img.shape

    # blended_img = blend_imgs(img1=sensor_img, img2=no_change_img, alpha1=0.7)

    blended_img = blend_imgs_preserve_contrast(sensor_img, no_change_img)

    if render_text:
        # cv2 add text to blended img
        if is_match:
            text = "MATCH"
            x = w // 5
            font_scale = 10
            font_color = (34, 139, 34)  # forest green
        else:
            text = "MISMATCH"
            x = 100
            font_scale = 7
            font_color = (255, 0, 0)
        y = h // 3

        blended_img = add_text_cv2(
            blended_img, text, coords_to_plot_at=(x, y), font_color=font_color, font_scale=font_scale, thickness=10
        )

        sensor_img = add_text_cv2(
            sensor_img,
            text="ONLINE IMAGERY",
            coords_to_plot_at=(300, h - 100),
            font_color=(255, 255, 255),
            font_scale=3,
            thickness=5,
        )

        no_change_img = add_text_cv2(
            no_change_img,
            text="ONBOARD MAP",
            coords_to_plot_at=(300, h - 100),
            font_color=(255, 255, 255),
            font_scale=3,
            thickness=5,
        )

    triplet_img = hstack_imgs_w_border([sensor_img, no_change_img, blended_img], border_sz=60)

    # plt.figure(figsize=(15,10))
    # plt.imshow(triplet_img)
    # plt.show()

    os.makedirs(save_dir, exist_ok=True)

    imageio.imwrite(f"{save_dir}/{img_fname}_ismatch{is_match}.jpg", triplet_img)
    #
    # plt.axis('off')
    #
    # assert len(mc_events) == 1
    # plt.title(f'is_match: {is_match}, @{mc_events[0].rel_start_s} s, {log_id[:5]} ')
    # print(log_id)
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(f'test_examples/{log_id_ts}.jpg', dpi=500)
    # plt.close('all')
