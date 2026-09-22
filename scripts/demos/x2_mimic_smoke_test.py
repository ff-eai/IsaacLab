"""Smoke test for Isaac-PickPlace-X2-Mimic-v0.

Checks the pieces Isaac Mimic relies on before you spend a teleop session on them:
the env builds and steps, the 46-dim action layout round-trips through the
pose/hand split, the subtask signal fires on a lifted-and-grasped can, and the
recorder terms instantiate (they resolve X2 body and joint names).

Runs headless with a GPU; no headset or CloudXR runtime needed.

    ./isaaclab.sh -p scripts/demos/x2_mimic_smoke_test.py --headless
"""

from __future__ import annotations

import argparse
import traceback

import pinocchio as _pin_preload  # noqa: F401
from pinocchio.robot_wrapper import RobotWrapper as _RW_preload  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_mimic.envs  # noqa: F401, E402  (registers the mimic envs)

from isaaclab_mimic.envs.pinocchio_envs.pickplace_x2_mimic_env import (  # noqa: E402
    GRASP_CLOSED_THRESHOLD,
    GRASP_JOINT_NAMES,
    GRASP_PROXIMITY_M,
    LIFT_MARGIN_M,
)
from isaaclab_mimic.envs.pinocchio_envs.pickplace_x2_mimic_env_cfg import (  # noqa: E402
    PickPlaceX2MimicEnvCfg,
)

TASK_ID = "Isaac-PickPlace-X2-Mimic-v0"
EXPECTED_ACTION_DIM = 46


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"[smoke] {'PASS' if ok else 'FAIL'}  {label}{f'  ({detail})' if detail else ''}")
        if not ok:
            failures.append(label)

    check("task registered", TASK_ID in gym.envs.registry, TASK_ID)
    if TASK_ID not in gym.envs.registry:
        return 1

    # Pink IK swallows solver failures and returns the current joint positions, so a
    # missing or ABI-mismatched QP backend shows up only as an arm that never moves
    # (fingers keep working -- they bypass IK). Solve a trivial QP with the solver
    # the env is configured to use, so that failure is loud here instead.
    solver_name = PickPlaceX2MimicEnvCfg().actions.upper_body_ik.controller.solver
    try:
        from qpsolvers import solve_qp

        solution = solve_qp(np.eye(2), np.array([1.0, 1.0]), solver=solver_name)
        check(f"IK solver '{solver_name}' usable", solution is not None, str(solution))
    except Exception as exc:  # noqa: BLE001 - any failure here means teleop is dead
        check(f"IK solver '{solver_name}' usable", False, f"{type(exc).__name__}: {exc}")

    # Two envs, so anything that mixes up per-env frames shows up here rather than
    # during a 4-env generation run.
    cfg = PickPlaceX2MimicEnvCfg()
    cfg.scene.num_envs = 2
    check("idle_action dim", len(cfg.idle_action) == EXPECTED_ACTION_DIM, f"{len(cfg.idle_action)}")
    check("subtask arms", set(cfg.subtask_configs) == {"left", "right"}, str(list(cfg.subtask_configs)))
    check("cameras off by default", cfg.scene.rgbd_head_front is None)

    env = gym.make(TASK_ID, cfg=cfg)
    u = env.unwrapped
    env.reset()
    check("action space", env.action_space.shape[-1] == EXPECTED_ACTION_DIM, str(env.action_space.shape))

    idle = cfg.idle_action.unsqueeze(0).repeat(u.num_envs, 1).to(u.device)
    for _ in range(12):
        env.step(idle)
    check("env steps", True)

    # Action layout: [left pose 7][right pose 7][left hand 16][right hand 16].
    gripper = u.actions_to_gripper_actions(idle)
    check(
        "hand slices",
        gripper["left"].shape[-1] == 16 and gripper["right"].shape[-1] == 16,
        f"L{tuple(gripper['left'].shape)} R{tuple(gripper['right'].shape)}",
    )
    poses = u.action_to_target_eef_pose(idle)
    rebuilt = u.target_eef_pose_to_action(
        {"left": poses["left"][0], "right": poses["right"][0]},
        {"left": gripper["left"][0], "right": gripper["right"][0]},
    )
    err = (rebuilt - idle[0]).abs().max().item()
    check("action round-trip", rebuilt.shape[0] == EXPECTED_ACTION_DIM and err == 0.0, f"max err {err:.2e}")

    # idle_right must stay low at rest and while the can is high but out of reach,
    # and fire only when both conditions hold.
    at_rest = u.get_subtask_term_signals()["idle_right"]
    check("idle_right false at rest", not bool(at_rest.any()))

    # Lifted but the hand never closed. This is the case proximity alone cannot
    # reject: with IK working, the idle pose leaves the wrist 0.15-0.18 m from the
    # can, straddling any useful proximity threshold. Finger closure is what makes
    # it decisive.
    baseline = u._can_rest_z.clone()
    u._can_rest_z = baseline - 0.20
    lifted_only = u.get_subtask_term_signals()["idle_right"]
    obs = u.obs_buf["policy"]
    hand_dist = torch.norm(obs["right_eef_pos"] - obs["object_pos"], dim=-1)
    robot = u.scene["robot"]
    grasp_ids, _ = robot.find_joints(list(GRASP_JOINT_NAMES), preserve_order=True)
    closure = robot.data.joint_pos[:, grasp_ids].mean(dim=-1)
    check(
        "idle_right false when lifted but hand open",
        not bool(lifted_only.any()),
        f"hand_dist={[round(v, 3) for v in hand_dist.tolist()]} (gate {GRASP_PROXIMITY_M}), "
        f"closure={[round(v, 3) for v in closure.tolist()]} (needs >{GRASP_CLOSED_THRESHOLD})",
    )
    u._can_rest_z = baseline

    # Put the can just above the right wrist and recompute observations without
    # stepping, so PhysX depenetration doesn't fling it out of the hand meshes.
    obj = u.scene["object"]
    eef_world = u.obs_buf["policy"]["right_eef_pos"] + u.scene.env_origins
    target = eef_world.clone()
    target[:, 2] += 0.5 * (LIFT_MARGIN_M + GRASP_PROXIMITY_M)
    quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=u.device).repeat(u.num_envs, 1)
    obj.write_root_pose_to_sim(torch.cat([target, quat], dim=-1))
    # Close the thumb and index, since a grasp now requires closure as well as
    # proximity and lift.
    joint_pos = robot.data.joint_pos.clone()
    joint_pos[:, grasp_ids] = GRASP_CLOSED_THRESHOLD + 0.3
    robot.write_joint_state_to_sim(joint_pos, robot.data.joint_vel.clone())
    u.obs_buf = u.observation_manager.compute()
    u._can_rest_z = (target[:, 2] - u.scene.env_origins[:, 2]) - (LIFT_MARGIN_M + 0.03)
    grasped = u.get_subtask_term_signals()["idle_right"]
    check("idle_right fires in all envs when grasped+lifted", bool(grasped.all()), str(grasped.tolist()))

    subset = u.get_subtask_term_signals(env_ids=[1])["idle_right"]
    check("env_ids indexing", tuple(subset.shape) == (1,), str(tuple(subset.shape)))

    terms = list(u.recorder_manager.active_terms)
    check("success-condition recorder active", "record_pickplace_success_conditions" in terms)

    print(f"\n[smoke] {'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    env.close()
    app.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        app.close()
        raise SystemExit(1) from None
