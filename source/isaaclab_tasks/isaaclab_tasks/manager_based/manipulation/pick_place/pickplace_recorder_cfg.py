# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder configuration for A2 pick-place mimic data generation."""

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
    CameraRecorderCfg,
)
from isaaclab.utils import configclass

from .mdp.success_condition_recorder import PickPlaceSuccessConditionRecorderCfg


@configclass
class PickPlaceRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder plus pick-place success sub-condition labels."""

    # record_failure_reasons inherited from ActionStateRecorderManagerCfg → FailureReasonRecorderCfg
    record_pickplace_success_conditions = PickPlaceSuccessConditionRecorderCfg()

    # Camera frames for dataset generation (requires --enable_cameras).
    record_head_cam = CameraRecorderCfg(
        sensor_names=["head_camera"],
        data_types=["rgb", "distance_to_image_plane"],
    )
    record_chest_left_cam = CameraRecorderCfg(
        sensor_names=["chest_left_camera"],
        data_types=["rgb"],
    )
    record_chest_right_cam = CameraRecorderCfg(
        sensor_names=["chest_right_camera"],
        data_types=["rgb"],
    )
