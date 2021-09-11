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

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import argoverse.utils.interpolate as interp_utils
from argoverse.map_representation.map_api_v2 import ArgoverseStaticMapV2, LaneSegment, Polyline
from argoverse.utils.datetime_utils import generate_datetime_string
from argoverse.utils.polyline_density import get_polyline_length

import tbv.synthetic_generation.synthetic_crosswalk_generator as synthetic_crosswalk_generator
import tbv.utils.infinity_norm_utils as infinity_norm_utils
from tbv.rendering.bev_vector_map_rendering_utils import bev_render_window
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig
from tbv.rendering.map_rendering_classes import LocalLaneMarking, LocalVectorMap, LocalPedCrossing
from tbv.utils.dir_utils import check_mkdir

WPT_INFTY_NORM_INTERP_NUM = 50

MARKING_DELETION_PATH_LENGTH = 3
MARKING_COLOR_CHANGE_PATH_LENGTH = 3
BIKE_LANE_DESIRED_PATH_LENGTH = 5


"""Utilities for applying synthetic perturbations to a map.

---- From the paper appendix ----
Our main observations from studying real-world map changes are that lane changes generally occur over a chain of lane
segments, with combined length often over tens or hundreds of meters, although at times the combined length is far
shorter. Accordingly, we use the directed lane graph to sample random connected sequences of lane segments, respecting
valid successors. We then manipulate either the left or the right boundary only (not both) of this lane sequence.

Our general procedure is to start this sequence at a random lane well-within the field of view of the BEV image.
As before, we ensure that the sampled marking is not entirely contained within the outermost 1/8 of pixels along
any border of the rendered map image (i.e. within 15 m according to l-infinity norm from the egovehicle).

When deleting lane boundaries, we sample only painted yellow or white lane boundary markings.  When changing the
color or structure of lane boundaries, we sample lane boundary markings of any color (including those that are
implicit). When adding a bike lane, we sample a sequence of 5 lane segments. For marking deletion and changes to
lane marking color and structure, we sample a sequence of length 3.

We render these boundaries as colored polylines; we use red for implicit boundaries, and yellow and white for
lane markings of their respective color. Lane boundary markings are deleted by simply not rendering them in the
rasterized image. 

Bike lanes generally represent the rightmost lane in the United States. Accordingly, we synthesize a valid location
for a new bike lane by iterating through the lane graph until there is no right neighbor; by dividing this rightmost
lane into half, we can create two half-width lanes in place of one. We use solid white lines to represent their
boundaries.


Many more possible map perturbations are possible, although we do not explore them, e.g.
- insert trapezoid paint on the right side of the road
- change number of lanes in road (change subdivision)
- add parking spots (orthogonal lines)
- insert bollards in intersection, around curve, white dots
- jitter chained lane boundaries (shift to the left or the right several chained boundaries),
   e.g. jitter in the direction of its normal, by at least a large threshold.
- add a turn lane
"""

# Dictionary dictates allow changes for "CHANGE_LANE_MARKING_COLOR"
ALT_LANE_MARK_COLOR_DICT = {
    "SOLID_WHITE": ["SOLID_YELLOW", "NONE"],
    "DASHED_WHITE": ["DASHED_YELLOW", "NONE"],
    "DOUBLE_SOLID_WHITE": ["DOUBLE_SOLID_YELLOW"],
    "DOUBLE_DASH_WHITE": ["DOUBLE_DASH_YELLOW"],
    "DOUBLE_DASH_YELLOW": ["DOUBLE_DASH_WHITE"],
    "DASHED_YELLOW": ["DASHED_WHITE", "NONE"],
    "DOUBLE_SOLID_YELLOW": ["DOUBLE_SOLID_WHITE"],
    "SOLID_YELLOW": ["SOLID_WHITE", "NONE"],
    "SOLID_DASH_YELLOW": ["DOUBLE_DASH_WHITE"],  # no clear alternative, hack
    "DASH_SOLID_YELLOW": ["DOUBLE_DASH_WHITE"],  # no clear alternative, hack
    "NONE": [
        "SOLID_WHITE",
        "SOLID_DASH_YELLOW",
        "DOUBLE_SOLID_WHITE",
        "DOUBLE_DASH_YELLOW",
        "SOLID_YELLOW",
        "DOUBLE_DASH_WHITE",
        "DASHED_YELLOW",
        "DASH_SOLID_YELLOW",
        "DOUBLE_SOLID_YELLOW",
        "DASHED_WHITE",
    ],
}


# Dictionary dictates allowed changes for "CHANGE_LANE_BOUNDARY_DASH_SOLID"
# keep color, just change structure
# TODO: differentiate between dash_solid and dashed
ALT_LANE_MARK_STRUCTURE_DICT = {
    "SOLID_WHITE": ["DASHED_WHITE", "DOUBLE_SOLID_WHITE", "DOUBLE_DASH_WHITE"],
    "SOLID_DASH_YELLOW": ["SOLID_YELLOW", "DOUBLE_DASH_YELLOW", "DOUBLE_SOLID_YELLOW"],
    "DOUBLE_SOLID_WHITE": ["DASHED_WHITE"],
    "DOUBLE_SOLID_YELLOW": ["DASHED_YELLOW", "SOLID_YELLOW"],
    "DOUBLE_DASH_WHITE": ["SOLID_WHITE", "DASHED_WHITE"],
    "DOUBLE_DASH_YELLOW": ["SOLID_YELLOW", "DOUBLE_SOLID_YELLOW"],
    "SOLID_YELLOW": ["DASHED_YELLOW", "DOUBLE_DASH_YELLOW", "DOUBLE_SOLID_YELLOW"],
    "DASHED_YELLOW": ["SOLID_YELLOW", "DOUBLE_SOLID_YELLOW"],
    "DASHED_WHITE": ["SOLID_WHITE"],
    "DASH_SOLID_YELLOW": ["SOLID_YELLOW", "DOUBLE_SOLID_YELLOW"],
}


class SyntheticChangeType(str, Enum):
    """
    Types of changes:
    - switch solid line for dashed line, or vice versa
    - switch white line to yellow line, or vice versa
    -
    - insert crosswalk randomly across lane segment, or in front of stop line
    - delete drawn polyline for some lane (colored -> none)
    - add bike lane (very thin lane),
    """

    CHANGE_LANE_BOUNDARY_DASH_SOLID: str = "CHANGE_LANE_BOUNDARY_DASH_SOLID"
    CHANGE_LANE_MARKING_COLOR: str = "CHANGE_LANE_MARKING_COLOR"
    DELETE_CROSSWALK: str = "DELETE_CROSSWALK"
    INSERT_CROSSWALK: str = "INSERT_CROSSWALK"
    DELETE_LANE_MARKING: str = "DELETE_LANE_MARKING"
    ADD_BIKE_LANE: str = "ADD_BIKE_LANE"


def render_perturbed_bev(
    avm: ArgoverseStaticMapV2,
    timestamp: int,
    dirname: str,
    log_id: str,
    config: BevRenderingConfig,
    ego_center: np.ndarray,
    change_type: Union[str, SyntheticChangeType] = "no_change",
) -> None:
    """

    Args:
        avm: local map for TbV log/scenario.
        timestamp:
        dirname:
        log_id: string representing unique identifier for TbV log/scenario to render.
        config: specification of rendering parameters for BEV data.
        ego_center:
        change_type: type of synthetic change to apply to the local vector map.
    """
    lvm, save_fpath = create_and_perturb_local_vector_map(
        avm=avm,
        timestamp=timestamp,
        dirname=dirname,
        log_id=log_id,
        config=config,
        ego_center=ego_center,
        change_type=change_type,
    )
    if lvm is None:
        return
    # returns a copy of the BEV image, but it is already saved to disk, so we discard the return value.
    _ = bev_render_window(lvm, save_fpath, log_id, ego_center, config.resolution, dilation=config.dilation)


def create_and_perturb_local_vector_map(
    avm: ArgoverseStaticMapV2,
    timestamp: int,
    dirname: str,
    log_id: str,
    config: Union[BevRenderingConfig, EgoviewRenderingConfig],
    ego_center: np.ndarray,
    change_type: Union[str, SyntheticChangeType] = "no_change",
) -> Tuple[Optional[LocalVectorMap], Optional[str]]:
    """Synthetically manipulate a local vector map, for either egoview or BEV rendering.

    Args:
        avm: local map for TbV log/scenario.
        timestamp:
        dirname
        log_id: string representing unique identifier for TbV log/scenario to render.
        config: specification of rendering parameters for BEV or egoview data.
        ego_center:
        change_type: type of synthetic change to apply to the local vector map.

    Returns:
        lvm: (returns None upon failure)
        save_fpath: (returns None upon failure)
    """
    window_sz = config.dilation
    ego_center = ego_center.reshape(1, 2)

    print(f"Generating (and optionally perturbing) local map for {change_type}")

    lvm = LocalVectorMap(avm, ego_center, config.dilation)
    lpcs = avm.get_scenario_ped_crossings()  # ego_center) # or could put a distance-based search into the local map.

    # now, place flag for which local ped crossing are strictly nearby
    marked_ped_crossings = []
    for lpc in lpcs:
        edge1, edge2 = lpc.edge1.xyz, lpc.edge2.xyz

        # could have very long segment, with endpoints and all waypoints outside of radius
        edge1_interp = interp_utils.interp_arc(t=WPT_INFTY_NORM_INTERP_NUM, px=edge1[:, 0], py=edge1[:, 1])
        edge2_interp = interp_utils.interp_arc(t=WPT_INFTY_NORM_INTERP_NUM, px=edge2[:, 0], py=edge2[:, 1])
        is_nearby = infinity_norm_utils.has_pts_in_infty_norm_radius(
            edge1_interp, ego_center, config.max_dist_to_del_crosswalk
        ) or infinity_norm_utils.has_pts_in_infty_norm_radius(
            edge2_interp, ego_center, config.max_dist_to_del_crosswalk
        )
        # now, is marked whether it is closeby
        mpc = LocalPedCrossing(edge1, edge2, is_nearby)
        marked_ped_crossings.append(mpc)

    lvm.ped_crossing_edges = marked_ped_crossings

    vector_lane_segments = avm.get_nearby_lane_segments(ego_center, window_sz)

    for vls in vector_lane_segments:
        lvm.nearby_lane_segment_dict[vls.id] = vls
        lvm.nearby_lane_markings += [vls.get_left_lane_marking()]
        lvm.nearby_lane_markings += [vls.get_right_lane_marking()]

        # lane_segment['has_traffic_control']
        # stopline occurs at start of an intersection lane segment
        if vls.is_intersection:
            stopline = np.vstack([vls.right_lane_boundary.xyz[0, :2], vls.left_lane_boundary.xyz[0, :2]])
            lvm.stoplines += [stopline]

    # currently, only one change at a time
    if change_type == SyntheticChangeType.DELETE_CROSSWALK:
        num_crosswalks = len(lvm.ped_crossing_edges)
        ped_crossings_are_nearby = np.array([pce.is_nearby for pce in lvm.ped_crossing_edges])
        num_crosswalks_nearby = ped_crossings_are_nearby.sum()
        if num_crosswalks_nearby < 1:
            # cannot perform the desired augmentation, abort
            return None, None

        idx_choices = np.where(ped_crossings_are_nearby)[0]
        crosswalk_idx = np.random.choice(a=idx_choices)
        # TODO: just set as non-renderable, instead of deleting anything
        # removes the item at a specific index

        lvm.changed_points = np.vstack(
            [lvm.ped_crossing_edges[crosswalk_idx].edge1.copy(), lvm.ped_crossing_edges[crosswalk_idx].edge2.copy()]
        )

        del lvm.ped_crossing_edges[crosswalk_idx]
        assert len(lvm.ped_crossing_edges) == num_crosswalks - 1
        # print(f'from {num_crosswalks} to {len(lvm.ped_crossing_edges)}')

    elif change_type == SyntheticChangeType.INSERT_CROSSWALK:
        # insert crosswalk randomly across lane segment, or in front of stop line

        try:
            lvm = synthetic_crosswalk_generator.insert_synthetic_crosswalk(lvm, ego_center, window_sz)
        except Exception as e:
            logging.exception(f"Synthetic crosswalk generation failed for {log_id}")
            return None, None  # likely some lane segment was not interpolate-able

    elif change_type == SyntheticChangeType.CHANGE_LANE_BOUNDARY_DASH_SOLID:
        # switch solid line for dashed line, or vice versa

        # ignore the implicit boundaries for now
        noticeable_bnd_idxs = lvm.get_painted_nonfringe_marking_idxs()
        if len(noticeable_bnd_idxs) == 0:
            # abort, no real markings to delete
            logging.info("Skipping CHANGE_LANE_BOUNDARY_DASH_SOLID: There were no yellow or white markings to change")
            print("Skipping CHANGE_LANE_BOUNDARY_DASH_SOLID: There were no yellow or white markings to change")
            return None, None

        lvm = change_lane_marking(lvm, "structure")

    elif change_type == SyntheticChangeType.CHANGE_LANE_MARKING_COLOR:  # switch white line to yellow line

        # date_str = generate_datetime_string()
        # fig_save_fpath = f'color_changes/{date_str}_A.jpg'
        # plot_lane_colored_geometry(fig_save_fpath, lvm.nearby_lane_segment_dict)

        lvm = change_lane_marking(lvm, "color")

        # fig_save_fpath = f'color_changes/{date_str}_B.jpg'
        # plot_lane_colored_geometry(fig_save_fpath, lvm.nearby_lane_segment_dict)

    elif change_type == SyntheticChangeType.DELETE_LANE_MARKING:

        noticeable_bnd_idxs = lvm.get_painted_nonfringe_marking_idxs()

        if len(noticeable_bnd_idxs) == 0:
            # abort, no real markings to delete
            logging.info("There were no yellow or white markings to delete")
            # print("There were no yellow or white markings to delete")
            return None, None

        # date_str = generate_datetime_string()
        # fig_save_fpath = f'delete_changes/{date_str}_A.jpg'
        # plot_lane_colored_geometry(fig_save_fpath, lvm.nearby_lane_segment_dict)

        lvm = delete_lane_marking(lvm)

        # fig_save_fpath = f'delete_changes/{date_str}_B.jpg'
        # plot_lane_colored_geometry(fig_save_fpath, lvm.nearby_lane_segment_dict)

    elif change_type == SyntheticChangeType.ADD_BIKE_LANE:  # add bike lane (very thin lane)

        # lane types can be 'BIKE', 'VEHICLE', 'BUS', or "NON_VEHICLE"
        if any([vls.lane_type == "BIKE" for vls in lvm.nearby_lane_segment_dict.values()]):
            # Don't synthetically generate if there is already one here...
            # TODO: check to see if heuristic finds the same one, only then quit
            return None, None

        lvm = add_bike_lane(lvm)

    # if 'no_change', just pass through cases above
    dirname += f"/{change_type.lower()}"

    im_fname = f"{log_id}_{timestamp}_vectormap.jpg"
    check_mkdir(f"{config.rendered_dataset_dir}/{dirname}")
    save_fpath = f"{config.rendered_dataset_dir}/{dirname}/{im_fname}"

    # plt.axis('equal')
    # plt.show()
    return lvm, save_fpath


def delete_lane_marking(lvm: LocalVectorMap) -> LocalVectorMap:
    """Delete drawn polyline for some lane boundary in the local vector map.

    Args:
        lvm: local vector map, before a lane marking is deleted.

    Returns:
        lvm: local vector map, after a lane marking is deleted.
    """
    # ignore the implicit boundaries for now
    noticeable_bnd_idxs = lvm.get_painted_nonfringe_marking_idxs()

    # choose a random lane segment to delete
    del_marking_idx = np.random.choice(a=noticeable_bnd_idxs)
    random_marking: LocalLaneMarking = lvm.nearby_lane_markings[del_marking_idx]

    print(f"Deleting {del_marking_idx} with bnd {random_marking.mark_type}")
    # must also change its corresponding left/right neighbor (if exists)
    # to prevent just blending the two colors together

    start_lane_id = random_marking.src_lane_id
    traj_lane_ids = lvm.get_random_lane_id_sequence(start_lane_id, path_length=MARKING_DELETION_PATH_LENGTH)

    lvm = get_changed_polyline(lvm, traj_lane_ids, bnd_side_to_change=random_marking.bound_side)

    # mark lane IDs as not to be rendered
    for traj_lane_id in traj_lane_ids:
        left_neighbor_id = lvm.nearby_lane_segment_dict[traj_lane_id].left_neighbor_id
        right_neighbor_id = lvm.nearby_lane_segment_dict[traj_lane_id].right_neighbor_id

        # must account for same or opposite direction travel
        # mark lane IDs as not to be rendered
        if random_marking.bound_side == "left":
            lvm.nearby_lane_segment_dict[traj_lane_id].render_l_bound = False
            if left_neighbor_id is not None and lvm.lane_is_in_map(left_neighbor_id):
                if lvm.nearby_lane_segment_dict[left_neighbor_id].right_neighbor_id == traj_lane_id:
                    lvm.nearby_lane_segment_dict[left_neighbor_id].render_r_bound = False
                else:
                    lvm.nearby_lane_segment_dict[left_neighbor_id].render_l_bound = False
        else:
            lvm.nearby_lane_segment_dict[traj_lane_id].render_r_bound = False
            if right_neighbor_id is not None and lvm.lane_is_in_map(right_neighbor_id):
                if lvm.nearby_lane_segment_dict[right_neighbor_id].left_neighbor_id == traj_lane_id:
                    lvm.nearby_lane_segment_dict[right_neighbor_id].render_l_bound = False
                else:
                    # this case should only occur in countries where one drives on the left side of road
                    # e.g. U.K. not the U.S.
                    lvm.nearby_lane_segment_dict[right_neighbor_id].render_r_bound = False
    return lvm


def get_changed_polyline(lvm: LocalVectorMap, traj_lane_ids: List[int], bnd_side_to_change: str) -> LocalVectorMap:
    """Get sequence of coordinates correspondencing to changed lane graph path.

    In the ego-view, we use these changed points to determine whether or not a synthetic change
    is visible in an image.

    Args:
        lvm:
        traj_lane_ids: path/trajectory of chained, consecutive lane segments.
        bnd_side_to_change: side to change lane marking on (either "left" or "right").

    Returns:
        lvm:
    """
    lvm.changed_points = []

    # mark lane IDs as not to be rendered
    for traj_lane_id in traj_lane_ids:

        if bnd_side_to_change == "left":
            lvm.changed_points += [lvm.nearby_lane_segment_dict[traj_lane_id].left_lane_boundary]
        else:
            lvm.changed_points += [lvm.nearby_lane_segment_dict[traj_lane_id].right_lane_boundary]

    lvm.changed_points = np.vstack(lvm.changed_points)
    return lvm


def change_lane_marking(lvm: LocalVectorMap, change_type: str) -> LocalVectorMap:
    """Sample a random right or left boundary, and change the color of its lane markings and for its successors.

    Either:
        preserve the structure, just change color
            if white, make it yellow or none
            if yellow, make it white or none
            if none, make it white or yellow
        or change structure, keep color

    Args:
        lvm: local vector map, before a lane marking is changed.

    Returns:
        lvm: local vector map, after a lane marking is changed.
    """
    # must also change its corresponding left/right neighbor (if exists)
    # to prevent just blending the two colors together

    # polyline_len = get_polyline_length(boundary_polyline)
    # print(f'Will modify {polyline_len} meters of lane boundary.')

    # choose a random lane segment
    if change_type == "color":
        rand_marking_idx = np.random.choice(a=lvm.get_nonfringe_marking_idxs())
    elif change_type == "structure":
        # ignore the implicit boundaries for now
        noticeable_bnd_idxs = lvm.get_painted_nonfringe_marking_idxs()
        assert len(noticeable_bnd_idxs) > 0  # already checked previously
        rand_marking_idx = np.random.choice(a=noticeable_bnd_idxs)

    random_marking: LocalLaneMarking = lvm.nearby_lane_markings[rand_marking_idx]
    # print(f"Changing {rand_marking_idx} with marking {random_marking.mark_type}")
    mark_type = random_marking.mark_type
    if change_type == "color":
        new_mark_type = np.random.choice(ALT_LANE_MARK_COLOR_DICT[mark_type])
    elif change_type == "structure":
        new_mark_type = np.random.choice(ALT_LANE_MARK_STRUCTURE_DICT[mark_type])

    start_lane_id = random_marking.src_lane_id
    traj_lane_ids = lvm.get_random_lane_id_sequence(start_lane_id, path_length=MARKING_COLOR_CHANGE_PATH_LENGTH)

    # print(f"Change {change_type} from {mark_type} to {new_mark_type}")

    lvm = get_changed_polyline(lvm, traj_lane_ids, random_marking.bound_side)

    # mark lane IDs as not to be rendered
    for traj_lane_id in traj_lane_ids:
        left_neighbor_id = lvm.nearby_lane_segment_dict[traj_lane_id].left_neighbor_id
        right_neighbor_id = lvm.nearby_lane_segment_dict[traj_lane_id].right_neighbor_id

        # must account for same or opposite direction travel
        if random_marking.bound_side == "left":
            lvm.nearby_lane_segment_dict[traj_lane_id].left_mark_type = new_mark_type
            if left_neighbor_id is not None and lvm.lane_is_in_map(left_neighbor_id):
                if lvm.nearby_lane_segment_dict[left_neighbor_id].right_neighbor_id == traj_lane_id:
                    lvm.nearby_lane_segment_dict[left_neighbor_id].right_mark_type = new_mark_type
                else:
                    lvm.nearby_lane_segment_dict[left_neighbor_id].left_mark_type = new_mark_type
        else:
            lvm.nearby_lane_segment_dict[traj_lane_id].right_mark_type = new_mark_type
            if right_neighbor_id is not None and lvm.lane_is_in_map(right_neighbor_id):
                if lvm.nearby_lane_segment_dict[right_neighbor_id].left_neighbor_id == traj_lane_id:
                    lvm.nearby_lane_segment_dict[right_neighbor_id].left_mark_type = new_mark_type
                else:
                    # this case should only occur in countries where one drives on the left side of road
                    # e.g. U.K. not the U.S.
                    lvm.nearby_lane_segment_dict[right_neighbor_id].right_mark_type = new_mark_type

    return lvm


def add_bike_lane(lvm: LocalVectorMap) -> LocalVectorMap:
    """Add a synthetic bike lane to the local vector map.

    Args:
        lvm: local vector map, before a bike lane is added.

    Returns:
        lvm: local vector map, after a bike lane is added.
    """
    # find the rightmost lane
    query_lane_id = np.random.choice(lvm.get_nonfringe_lane_ids())
    rightmost_lane_id = lvm.get_rightmost_neighbor(query_lane_id)

    # then get a straight trajectory following it
    traj_lane_ids = lvm.get_random_lane_id_sequence(rightmost_lane_id, path_length=BIKE_LANE_DESIRED_PATH_LENGTH)

    # no longer render the rightmost lane
    # instead add a new lane segment to the graph, with a random lane ID
    DUMMY_LANE_ID = -500

    lvm.changed_points = []

    for i, traj_lane_id in enumerate(traj_lane_ids):
        traj_lane = lvm.nearby_lane_segment_dict[traj_lane_id]

        # do the linear interpolation directly in 3d
        lane_centerline, _ = interp_utils.compute_midpoint_line(
            left_ln_bnds=traj_lane.left_lane_boundary.xyz,
            right_ln_bnds=traj_lane.right_lane_boundary.xyz,
            num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        )
        lvm.changed_points += [lane_centerline]

        # add a new bike lane
        new_lane_id = DUMMY_LANE_ID + i
        lvm.nearby_lane_segment_dict[new_lane_id] = LaneSegment(
            id=new_lane_id,
            is_intersection=traj_lane.is_intersection,
            lane_type="BIKE",
            right_lane_boundary=Polyline.from_array(traj_lane.right_lane_boundary.xyz),
            left_lane_boundary=Polyline.from_array(lane_centerline),
            right_mark_type="SOLID_WHITE",  # could use multiple colors later
            left_mark_type="SOLID_WHITE",  # could use multiple colors later
            right_neighbor_id=traj_lane.right_neighbor_id,
            left_neighbor_id=traj_lane.left_neighbor_id,
            successors=[],  # irrelevant, no more graph operations will be performed
        )

    lvm.changed_points = np.vstack(lvm.changed_points)
    return lvm


def plot_lane_colored_geometry(fig_save_fpath: str, nearby_lane_segment_dict: Dict[int, LaneSegment]):
    """Save image w/ lane configuration against a black background.

    Args:
        fig_save_fpath
        nearby_lane_segment_dict
    """
    plt.rcParams["savefig.facecolor"] = "black"
    plt.rcParams["figure.facecolor"] = "black"
    plt.rcParams["axes.facecolor"] = "black"
    fig = plt.figure(figsize=(15, 15))
    plt.axis("off")
    ax = fig.add_subplot(111)
    ax.set_facecolor((0, 0, 0))

    # Use the lane marking information ("mark_type")
    for vls in nearby_lane_segment_dict.values():
        lane_polylines = []
        lane_mark_types = []

        if vls.render_l_bound:
            lane_polylines += [vls.left_lane_boundary]
            lane_mark_types += [vls.left_mark_type]

        if vls.render_r_bound:
            lane_polylines += [vls.right_lane_boundary]
            lane_mark_types += [vls.right_mark_type]

        for lane_polyline, mark_type in zip(lane_polylines, lane_mark_types):
            if "yellow" in mark_type.lower():
                color = "y"
            elif "white" in mark_type.lower():
                color = "w"
            elif "none" in mark_type.lower():
                color = "r"
            else:
                raise RuntimeError
            ax.plot(lane_polyline[:, 0], lane_polyline[:, 1], "--", color=color)

    # ax.set_facecolor((0, 0, 0))
    ax.set_facecolor("k")
    ax.axis("equal")
    # plt.show()
    fig.tight_layout()
    plt.savefig(fig_save_fpath, dpi=400)

    plt.close("all")
