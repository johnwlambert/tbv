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
Interface to the MSeg taxonomy and to semantic segmentation label maps dumped by a model, in a canonical
folder structure. 
   See the MSeg paper: http://vladlen.info/papers/MSeg.pdf
   or the MSeg repo: https://github.com/mseg-dataset/mseg-api, https://github.com/mseg-dataset/mseg-semantic
"""

import glob
import logging
import os
import shutil
from pathlib import Path

import mseg.utils.names_utils as names_utils
import numpy as np
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from mseg_semantic.tool.universal_demo import run_universal_demo


def get_mseg_label_map_fpath_from_image_info(
    label_maps_dir: str, log_id: str, camera_name: str, img_fname_stem: str
) -> str:
    """
    Given canonical storage location for MSeg label maps, provide the path to the semantic segmentation label map
    associated with this image.

    Assumes MSeg-3M-480p model was used at single scale, with 358 px resolution input.

    Args:
        label_maps_dir: string representing path to directory ...
        log_id: string representing unique ID for a TbV log/scenario.
        camera_name: string representing name of a ring camera.
        img_fname_stem: 

    Returns:
        label_map_path: file path to where label map should be stored on disk.
    """
    label_map_path = f"{label_maps_dir}/mseg-3m-480_{log_id}_{camera_name}_universal_ss/358/gray/{img_fname_stem}.png"
    return label_map_path


def filter_by_semantic_classes(label_map: np.ndarray, uv: np.ndarray, render_road_only: bool = False) -> np.ndarray:
    """Provide a mask to filter away pixels corresponding to non-ground semantic classes, when generating ground
    imagery.

    Compare (x,y) coords against semantic label map ground classes.

    Args:
        label_map: array of shape (H,W)
        uv: array of shape (N,2)
        render_road_only: whether to consider the `ground` surface to include only road, or other possible classes
            as well.

    Returns:
        logicals: boolean array of shape (N,) indicating which uv pixel locations correspond to the ground class.
    """
    u_classnames = names_utils.get_universal_class_names()
    if render_road_only:
        ground_classes = ["road"]
    else:
        ground_classes = ["road", "sidewalk_pavement", "terrain", "gravel", "railroad"]
    ground_uids = [u_classnames.index(g_class) for g_class in ground_classes]
    valid_semantic_mask = np.logical_or.reduce([label_map == uid for uid in ground_uids])

    y = uv[:, 1]
    x = uv[:, 0]
    logicals = valid_semantic_mask[y, x] != 0
    return logicals


def run_semantic_segmentation(log_id: str, slice_extraction_dir: str, mseg_semantic_repo_root: str) -> None:
    """Run MSeg model at single-scale over all images in a log.

    Single scale is actually sufficient (multi-scale inference not required for good results).

    Args:
        log_id
        slice_extraction_dir
        mseg_semantic_repo_root:
    """
    model_name = "mseg-3m"
    model_path = f"{mseg_semantic_repo_root}/pretrained-models/mseg-3m.pth"
    config_fpath = f"{mseg_semantic_repo_root}/mseg_semantic/config/test/default_config_360_ss.yaml"

    model_name = "mseg-3m-480"
    model_path = f"{mseg_semantic_repo_root}/pretrained-models/mseg-3m-480p.pth"
    config_fpath = f"{mseg_semantic_repo_root}/mseg_semantic/config/test/default_config_360_ss_473x473.yaml"

    use_gpu = True
    for camera_name in RING_CAMERA_LIST:

        input_file = f"{slice_extraction_dir}/{camera_name}"
        # move the predicted label maps into the log's own directory
        dirname = f"mseg-3m-480_{log_id}_{camera_name}_universal_ss"
        src_dir = f"{mseg_semantic_repo_root}/temp_files/{dirname}"
        dst_dir = f"{slice_extraction_dir}/{dirname}"

        # if subsampled_label_maps_exist(
        #   f'{mseg_semantic_repo_root}/temp_files', # parent dir
        #   log_id,
        #   input_file,
        #   camera_name
        # ) or subsampled_label_maps_exist(
        #   slice_extraction_dir, # parent dir
        #   log_id,
        #   input_file,
        #   camera_name
        # ):
        #   logging.info(f'All label maps already were precomputed for {log_id}')

        #   # copy to final location, if not copied yet
        #   copy_label_map_dir_contents(src_dir, dst_dir)
        #   continue

        logging.info(f"Semantic segmentation of {camera_name} for {log_id}")

        args = config.load_cfg_from_cfg_file(config_fpath)
        args.model_path = model_path
        args.model_name = model_name
        args.input_file = input_file
        args.workers = 1
        assert isinstance(args.model_name, str)
        assert isinstance(args.model_path, str)
        if args.dataset == "default":
            args.dataset = "_".join(Path(args.input_file).parts[-2:])

        check_mkdir(src_dir)
        check_mkdir(dst_dir)
        # RE-USE THE OLD ONES
        # copy_label_map_dir_contents(dst_dir, src_dir)

        run_universal_demo(args, use_gpu)

        # save to cache now
        copy_label_map_dir_contents(src_dir, dst_dir)


def copy_label_map_dir_contents(src_dir: str, dst_dir: str) -> None:
    """ """
    if not Path(dst_dir).exists():
        shutil.move(src_dir, dst_dir)
    else:
        # cannot move if already exists
        img_fpaths = glob.glob(f"{src_dir}/358/gray/*.png")
        check_mkdir(f"{dst_dir}/358/gray")
        for img_fpath in img_fpaths:
            fname_stem = Path(img_fpath).stem
            dst_fpath = f"{dst_dir}/358/gray/{fname_stem}.png"
            shutil.copyfile(img_fpath, dst_fpath)


def test_copy_label_map_dir_contents():
    """ """

    src_dir = "tests/test_data/move_label_maps/src"
    dst_dir = "tests/test_data/move_label_maps/dst"
    copy_label_map_dir_contents(src_dir, dst_dir)

    # clean up for next unit test run
    for f in glob.glob(f"{dst_dir}/**/*.png", recursive=True):
        os.remove(f)
