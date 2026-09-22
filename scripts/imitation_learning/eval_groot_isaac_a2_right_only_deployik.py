"""Closed-loop eval for the right-only GR00T policy server (172.18.1.26:5557)
that uses the **deploy_groot_a2_real_isaac.py** Pink IK stack instead of the
env's built-in PinkInverseKinematicsAction.

Why: the real-robot deploy resolves the model's world-frame EE target into joint
targets via its own Pink stack (FrameTask + DampingTask + NullSpacePostureTask)
running on A2_expanded.urdf, then publishes raw joint commands. This eval mirrors
that path inside Isaac Lab so we can validate the deploy-side IK against ground
truth physics. The env is driven by a 53-d JointPositionAction (one per
articulation joint); the right arm 7-d slot is filled with the IK solution and
the right hand 12-d slots with action.right_hands. All other joints are
held at their measured value each tick.

Server modality (queried via get_modality_config):
    state:  right_arm[7], right_hand[12], right_eef[9]
    action: right_eef[9]   (RELATIVE EEF XYZ_ROT6D, decoded server-side)
            right_arm[7]   (ABSOLUTE NON_EEF; ignored — IK output is used instead)
            right_hands[12](ABSOLUTE NON_EEF)

IK config matches deploy defaults (deploy_groot_a2_real_isaac.py):
    backend=pink, solver=daqp, dt=0.006, max_iter=8
    FrameTask(right_eef_link, pos_cost=8, ori_cost=1, lm=12, gain=0.5)
    DampingTask(cost=0.5)
    NullSpacePostureTask(cost=0.5, lm=1.0,
        joints=shoulders[1..3]_L + shoulders[1..3]_R + waist)

Usage::

    isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2_right_only_deployik.py \\
        --groot_host 172.18.1.26 --groot_port 5557 \\
        --task "place the can in the tray" --episodes 5 --max_steps 600 \\
        --use_length 16 --enable_cameras
"""

from __future__ import annotations

import argparse
import functools
import io
import os
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pinocchio as pin  # noqa: F401  (must come before AppLauncher)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Right-only GR00T eval with deploy-style Pink IK.")
parser.add_argument("--task_id", type=str, default="Isaac-PickPlace-A2-Abs-v0")
parser.add_argument("--task", type=str, default="place the can in the tray")
parser.add_argument("--groot_host", type=str, default="172.18.1.26")
parser.add_argument("--groot_port", type=int, default=5557)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--use_length", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
# --- IK options (defaults match deploy_groot_a2_real_isaac.py) ---
parser.add_argument("--urdf", type=str,
                    default="/home/wagner/code/IsaacLab/assets/A2/A2_expanded.urdf")
parser.add_argument("--right_eef_link", type=str, default="right_arm_link07")
parser.add_argument("--ee_solver", type=str, default="daqp")
parser.add_argument("--ee_position_cost", type=float, default=8.0)
parser.add_argument("--ee_orientation_cost", type=float, default=1.0)
parser.add_argument("--ee_lm_damping", type=float, default=12.0)
parser.add_argument("--ee_gain", type=float, default=0.5)
parser.add_argument("--ee_damping_cost", type=float, default=0.5)
parser.add_argument("--ee_null_space_cost", type=float, default=0.5)
parser.add_argument("--ee_null_space_lm_damping", type=float, default=1.0)
parser.add_argument("--ee_ik_dt", type=float, default=0.0,
                    help="Pink integration dt [s]. 0 (default) -> use env.sim.dt (matches "
                         "Isaac's PinkInverseKinematicsAction). Deploy uses 0.006.")
parser.add_argument("--ee_ik_max_iter", type=int, default=1,
                    help="Pink integration steps per IK call. 1 (default) matches Isaac's "
                         "PinkInverseKinematicsAction; deploy uses 8.")
parser.add_argument("--max_delta_rad", type=float, default=0.0,
                    help="Per-tick clip on right-arm joint delta vs measured. 0 (default) "
                         "disables — Isaac's PinkIK has no explicit clip, the velocity*dt "
                         "step is naturally small. Deploy uses 0.05.")
# Robot base in world frame for sim (env spawn pose).
parser.add_argument("--world_base_xyz", type=float, nargs=3, default=[0.0, 0.05, 0.93])
parser.add_argument("--world_base_quat", type=float, nargs=4, default=[0.7071, 0.0, 0.0, 0.7071])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise SystemExit("Only num_envs=1 is supported.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import msgpack
import numpy as np
import torch
import zmq

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# Joint orderings (right-only schema). Same as eval_groot_isaac_a2_right_only.py.
# ---------------------------------------------------------------------------
RIGHT_ARM_NAMES = [f"idx{20 + i:02d}_right_arm_joint{i + 1}" for i in range(7)]
RIGHT_HAND_NAMES = [
    "R_index_1_joint", "R_middle_1_joint", "R_pinky_1_joint", "R_ring_1_joint", "R_thumb_swing_joint",
    "R_index_2_joint", "R_middle_2_joint", "R_pinky_2_joint", "R_ring_2_joint", "R_thumb_1_joint",
    "R_thumb_2_joint", "R_thumb_3_joint",
]
LEFT_ARM_NAMES = [f"idx{13 + i:02d}_left_arm_joint{i + 1}" for i in range(7)]
WAIST_NAME = "waist_yaw_joint"


# ---------------------------------------------------------------------------
# msgpack <-> ndarray serialization
# ---------------------------------------------------------------------------
def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _msg_to_bytes(data) -> bytes:
    return msgpack.packb(data, default=_encode_custom)


def _msg_from_bytes(b: bytes):
    return msgpack.unpackb(b, object_hook=_decode_custom)


class GrootClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        try:
            self.socket.send(_msg_to_bytes(request))
            msg = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        if msg == b"ERROR":
            raise RuntimeError("Server error.")
        resp = _msg_from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def reset(self, options: dict | None = None):
        return self.call("reset", {"options": options})

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self.call("get_action", {"observation": observation, "options": options})
        return tuple(resp)

    def get_modality_config(self):
        return self.call("get_modality_config", requires_input=False)


# ---------------------------------------------------------------------------
# Math helpers (rot6d <-> matrix, quat conversions)
# ---------------------------------------------------------------------------
def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    R = _quat_wxyz_to_R(np.asarray(q, dtype=np.float64))
    return R[:2, :].reshape(-1).astype(np.float32)


def _rot6d_to_R(r6: np.ndarray) -> np.ndarray:
    row0 = np.asarray(r6[0:3], dtype=np.float64)
    row1 = np.asarray(r6[3:6], dtype=np.float64)
    n0 = row0 / max(np.linalg.norm(row0), 1e-12)
    row1_proj = row1 - n0 * float(np.dot(n0, row1))
    n1 = row1_proj / max(np.linalg.norm(row1_proj), 1e-12)
    n2 = np.cross(n0, n1)
    return np.stack([n0, n1, n2], axis=0)  # rows = row0,row1,row2


def _img_hwc(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------------------------
# NullSpacePostureTask — Python port of Isaac Lab's
# isaaclab.controllers.pink_ik.null_space_posture_task.NullSpacePostureTask
# (verbatim from deploy_groot_a2_real_isaac.py).
# ---------------------------------------------------------------------------
def _build_null_space_posture_task_class():
    from pink.tasks import Task as _PinkTask

    class NullSpacePostureTask(_PinkTask):
        PSEUDOINVERSE_DAMPING_FACTOR = 1e-9

        def __init__(self, cost, lm_damping=0.0, gain=1.0,
                     controlled_frames=None, controlled_joints=None):
            super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
            self.target_q = None
            self.controlled_frames = list(controlled_frames or [])
            self.controlled_joints = list(controlled_joints or [])
            self._joint_mask = None

        def _build_joint_mapping(self, configuration):
            self._joint_mask = np.zeros(configuration.model.nq, dtype=np.float64)
            joint_names = configuration.model.names.tolist()[1:]  # skip universe
            controlled = set(self.controlled_joints)
            for i, name in enumerate(joint_names):
                if name in controlled:
                    self._joint_mask[i] = 1.0

        def set_target(self, target_q):
            self.target_q = np.array(target_q, dtype=np.float64, copy=True)

        def set_target_from_configuration(self, configuration):
            self.set_target(configuration.q)

        def compute_error(self, configuration):
            if self.target_q is None:
                raise ValueError("NullSpacePostureTask: call set_target() first")
            if self._joint_mask is None:
                self._build_joint_mapping(configuration)
            err = pin.difference(configuration.model, self.target_q, configuration.q)
            return self._joint_mask * err

        def compute_jacobian(self, configuration):
            if self._joint_mask is None:
                self._build_joint_mapping(configuration)
            n = configuration.model.nq
            if not self.controlled_frames:
                return np.eye(n)
            Js = [configuration.get_frame_jacobian(f) for f in self.controlled_frames]
            J = np.concatenate(Js, axis=0)
            JJT = J @ J.T
            JJT[np.diag_indices_from(JJT)] += self.PSEUDOINVERSE_DAMPING_FACTOR ** 2
            try:
                X = np.linalg.solve(JJT, J)
                return np.eye(n) - J.T @ X
            except np.linalg.LinAlgError:
                return np.eye(n) - np.linalg.pinv(J) @ J

    return NullSpacePostureTask


# ---------------------------------------------------------------------------
# DeployIK — Pink IK wrapper that mirrors deploy_groot_a2_real_isaac.py
# (right-arm-only configuration, no waist in IK).
# ---------------------------------------------------------------------------
class DeployIK:
    def __init__(self, args):
        urdf_path = Path(args.urdf)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        # Build pin model. Use buildModelFromUrdf so the root is base_link (no
        # free flyer) — matches deploy.
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        joint_names = self.model.names.tolist()[1:]  # skip "universe"
        self._q_idx = {n: self.model.joints[self.model.getJointId(n)].idx_q
                       for n in joint_names}

        self.right_eef_frame = self.model.getFrameId(args.right_eef_link)
        if self.right_eef_frame == self.model.nframes:
            raise RuntimeError(f"right_eef_link {args.right_eef_link} not in URDF")

        # World <-> base transform (sim spawn pose).
        bx = np.asarray(args.world_base_xyz, dtype=np.float64)
        bq = np.asarray(args.world_base_quat, dtype=np.float64)  # wxyz
        R_wb = _quat_wxyz_to_R(bq)
        self._R_bw = R_wb.T  # world -> base rotation
        self._t_bw = -self._R_bw @ bx  # world -> base translation

        # Pink stack
        import pink
        from pink import solve_ik
        from pink.tasks import DampingTask, FrameTask
        try:
            from pink.exceptions import NoSolutionFound
        except ImportError:  # older pin-pink (<3.2)
            from pink.exceptions import PinkError as NoSolutionFound
        self._pink = pink
        self._solve_ik = solve_ik
        self._NoSolutionFound = NoSolutionFound

        self.right_frame_task = FrameTask(
            args.right_eef_link,
            position_cost=float(args.ee_position_cost),
            orientation_cost=float(args.ee_orientation_cost),
            lm_damping=float(args.ee_lm_damping),
            gain=float(args.ee_gain),
        )
        self.damping_task = (DampingTask(cost=float(args.ee_damping_cost))
                             if args.ee_damping_cost > 0.0 else None)

        null_cost = float(args.ee_null_space_cost)
        if null_cost > 0.0:
            null_joints = (
                LEFT_ARM_NAMES[0:3] + RIGHT_ARM_NAMES[0:3]
                + ([WAIST_NAME] if WAIST_NAME in self._q_idx else [])
            )
            NullSpacePostureTask = _build_null_space_posture_task_class()
            self.null_task = NullSpacePostureTask(
                cost=null_cost,
                lm_damping=float(args.ee_null_space_lm_damping),
                controlled_frames=[args.right_eef_link],
                controlled_joints=null_joints,
            )
        else:
            self.null_task = None

        # ik_q_mask: only right arm allowed to move (waist excluded — same as
        # deploy default --no_ee_waist_ik).
        self._right_arm_q = np.asarray(
            [self._q_idx[n] for n in RIGHT_ARM_NAMES if n in self._q_idx],
            dtype=np.int64,
        )
        if len(self._right_arm_q) != 7:
            raise RuntimeError(
                f"Could not resolve all 7 right-arm joints in URDF, got {len(self._right_arm_q)}: "
                f"{RIGHT_ARM_NAMES}"
            )
        self._ik_q_mask = np.zeros(self.model.nq, dtype=bool)
        self._ik_q_mask[self._right_arm_q] = True

        self._ee_ik_dt = float(args.ee_ik_dt)
        self._ee_ik_max_iter = int(args.ee_ik_max_iter)
        self._ee_solver = args.ee_solver

        print(f"[deployik] URDF={urdf_path}  nq={self.model.nq}  "
              f"right_arm_q_idx={self._right_arm_q.tolist()}  "
              f"R_bw=\n{self._R_bw}\n  t_bw={self._t_bw}")

    def world_eef_target_to_base_se3(self, eef9_world: np.ndarray) -> "pin.SE3":
        p_w = np.asarray(eef9_world[0:3], dtype=np.float64)
        R_w = _rot6d_to_R(eef9_world[3:9])
        R_b = self._R_bw @ R_w
        p_b = self._R_bw @ p_w + self._t_bw
        return pin.SE3(R_b, p_b)

    def _q_from_state53_subset(self, env_qpos: np.ndarray, env_to_pin: dict) -> np.ndarray:
        """Build a model-sized q vector by copying values from the env q for
        every joint name shared with the pin model. Joints unique to the URDF
        (e.g. mimicked finger joints) stay at neutral."""
        q = pin.neutral(self.model)
        for env_idx, qi in env_to_pin.items():
            q[qi] = float(env_qpos[env_idx])
        return q

    def _clip_q_to_model_limits(self, q: np.ndarray) -> None:
        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        n = min(int(self.model.nq), int(q.shape[0]))
        for i in range(n):
            if float(hi[i]) > float(lo[i]) + 1e-9:
                q[i] = float(np.clip(float(q[i]), float(lo[i]), float(hi[i])))

    def fk_right_eef_base(self, env_qpos: np.ndarray, env_to_pin: dict) -> tuple[np.ndarray, np.ndarray]:
        """FK from current measured joints, returns (pos, R) of the right EEF in base frame."""
        q = self._q_from_state53_subset(env_qpos, env_to_pin)
        self._clip_q_to_model_limits(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.right_eef_frame]
        return np.asarray(T.translation), np.asarray(T.rotation)

    def solve_right(self, env_qpos: np.ndarray, env_to_pin: dict,
                    eef9_world: np.ndarray) -> np.ndarray:
        """Run Pink IK for the right arm. Returns 7 joint targets in
        RIGHT_ARM_NAMES order."""
        r_T = self.world_eef_target_to_base_se3(eef9_world)
        q0 = self._q_from_state53_subset(env_qpos, env_to_pin)
        self._clip_q_to_model_limits(q0)
        configuration = self._pink.Configuration(self.model, self.data, q0)
        if self.null_task is not None:
            self.null_task.set_target_from_configuration(configuration)

        tasks: list = [self.right_frame_task]
        if self.damping_task is not None:
            tasks.append(self.damping_task)
        if self.null_task is not None:
            tasks.append(self.null_task)

        eps = 1e-3
        for _ in range(self._ee_ik_max_iter):
            self.right_frame_task.set_target(r_T)
            try:
                velocity = self._solve_ik(
                    configuration,
                    tasks,
                    self._ee_ik_dt,
                    self._ee_solver,
                    damping=1e-12,
                    limits=[],
                    safety_break=False,
                )
            except self._NoSolutionFound:
                break
            velocity = np.asarray(velocity, dtype=np.float64)
            velocity[~self._ik_q_mask] = 0.0
            configuration.integrate_inplace(velocity, self._ee_ik_dt)
            q_next = np.array(configuration.q, dtype=np.float64, copy=True)
            self._clip_q_to_model_limits(q_next)
            configuration.update(q_next)
            pin.forwardKinematics(self.model, self.data, q_next)
            pin.updateFramePlacements(self.model, self.data)
            if np.linalg.norm(
                pin.log6(self.data.oMf[self.right_eef_frame].actInv(r_T)).vector
            ) < eps:
                break

        q = np.array(configuration.q, dtype=np.float64, copy=True)
        self._clip_q_to_model_limits(q)

        # IK debug — populated each call so the eval can periodically print it.
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_ach = self.data.oMf[self.right_eef_frame]
        self.last_pos_target_b = np.asarray(r_T.translation)
        self.last_pos_achieved_b = np.asarray(T_ach.translation)
        self.last_pos_err_m = float(np.linalg.norm(self.last_pos_target_b - self.last_pos_achieved_b))
        self.last_q_right_arm_init = q0[self._right_arm_q].astype(np.float32)
        self.last_q_right_arm_solved = q[self._right_arm_q].astype(np.float32)

        return q[self._right_arm_q].astype(np.float32)


# ---------------------------------------------------------------------------
# Env wiring
# ---------------------------------------------------------------------------
@configclass
class _JointActOnlyCfg:
    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


def encode_obs_for_groot(env, joint_pos_env: np.ndarray,
                         right_arm_env_idx: np.ndarray,
                         right_hand_env_idx: np.ndarray,
                         task: str) -> dict:
    pol = env.unwrapped.observation_manager.compute()["policy"]
    head_rgb = _img_hwc(pol["head_camera_rgb"])
    chest_l = _img_hwc(pol["chest_left_camera_rgb"])
    chest_r = _img_hwc(pol["chest_right_camera_rgb"])

    right_arm = joint_pos_env[right_arm_env_idx].astype(np.float32)
    right_hand = joint_pos_env[right_hand_env_idx].astype(np.float32)

    re_pos = pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float32)
    re_quat = pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float32)
    re_state = np.concatenate([re_pos, _quat_wxyz_to_rot6d(re_quat)], axis=0).astype(np.float32)

    return {
        "video": {
            "cam_high": head_rgb[None, None, ...].astype(np.uint8),
            "cam_chest_left": chest_l[None, None, ...].astype(np.uint8),
            "cam_chest_right": chest_r[None, None, ...].astype(np.uint8),
        },
        "state": {
            "right_arm": right_arm[None, None, :],
            "right_hand": right_hand[None, None, :],
            "right_eef": re_state[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def main():
    cfg = parse_env_cfg(args_cli.task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.actions = _JointActOnlyCfg()
    cfg.teleop_devices.devices = {}
    if hasattr(cfg, "recorders"):
        cfg.recorders = None
    cfg.episode_length_s = max(args_cli.max_steps * 0.05 + 5.0, getattr(cfg, "episode_length_s", 0.0))

    print(f"[eval] making env {args_cli.task_id} on {args_cli.device}")
    env = gym.make(args_cli.task_id, cfg=cfg).unwrapped
    env.seed = args_cli.seed

    env_joint_names = list(env.scene["robot"].data.joint_names)
    n_joints = len(env_joint_names)
    env_act_dim = env.action_manager.total_action_dim
    print(f"[eval] env articulation has {n_joints} joints, action_dim={env_act_dim}")

    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}

    def _idx(names):
        out = np.asarray([env_name_to_idx[n] for n in names if n in env_name_to_idx],
                         dtype=np.int64)
        if len(out) != len(names):
            missing = [n for n in names if n not in env_name_to_idx]
            raise SystemExit(f"[eval] joints missing in env articulation: {missing}")
        return out

    right_arm_env_idx = _idx(RIGHT_ARM_NAMES)
    right_hand_env_idx = _idx(RIGHT_HAND_NAMES)
    print(f"[eval] right_arm env idx ({len(right_arm_env_idx)}): {right_arm_env_idx.tolist()}")
    print(f"[eval] right_hand env idx ({len(right_hand_env_idx)}): {right_hand_env_idx.tolist()}")

    # If user left --ee_ik_dt 0, fall back to env.sim.dt so we match Isaac's
    # PinkInverseKinematicsAction integration timestep.
    if args_cli.ee_ik_dt <= 0.0:
        try:
            args_cli.ee_ik_dt = float(env.sim.get_physics_dt())
        except Exception:
            args_cli.ee_ik_dt = 1.0 / 120.0
        print(f"[eval] --ee_ik_dt resolved from sim.dt = {args_cli.ee_ik_dt:.6f}s")

    # Build IK and the env<->pin name mapping.
    ik = DeployIK(args_cli)
    env_to_pin: dict[int, int] = {}
    for env_i, name in enumerate(env_joint_names):
        qi = ik._q_idx.get(name)
        if qi is not None:
            env_to_pin[env_i] = qi
    print(f"[eval] mapped {len(env_to_pin)}/{n_joints} env joints into pin model")

    print(f"[eval] connecting to GR00T at {args_cli.groot_host}:{args_cli.groot_port}")
    client = GrootClient(args_cli.groot_host, args_cli.groot_port)
    if not client.ping():
        raise SystemExit("[eval] ping to GR00T server failed")
    try:
        modality = client.get_modality_config()
        if isinstance(modality, dict):
            print(f"[eval] server modality keys: {list(modality.keys())}")
    except Exception as e:
        print(f"[eval] get_modality_config warning: {e}")

    successes, episode_lengths = 0, []
    for ep in range(args_cli.episodes):
        try:
            client.reset()
        except Exception as e:
            print(f"  client.reset warning: {e}")

        env.reset()
        chunk_eef = None  # (T, 9)  right_eef per-row, world frame
        chunk_hands = None  # (T, 12) right_hands per-row
        success = False
        _dumped_fk_check = False
        for step in range(args_cli.max_steps):
            cur_joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            payload = encode_obs_for_groot(env, cur_joint_pos,
                                           right_arm_env_idx, right_hand_env_idx,
                                           args_cli.task)
            if not _dumped_fk_check:
                _pol = env.unwrapped.observation_manager.compute()["policy"]
                _re_pos_w = _pol["right_eef_pos"][0].detach().cpu().numpy().astype(np.float64)
                _re_quat_w = _pol["right_eef_quat"][0].detach().cpu().numpy().astype(np.float64)
                _R_w = _quat_wxyz_to_R(_re_quat_w)
                _pos_b_from_env = ik._R_bw @ _re_pos_w + ik._t_bw
                _R_b_from_env = ik._R_bw @ _R_w
                _pos_b_fk, _R_b_fk = ik.fk_right_eef_base(cur_joint_pos, env_to_pin)
                _pos_err = np.linalg.norm(_pos_b_from_env - _pos_b_fk)
                _R_err = np.linalg.norm(_R_b_from_env - _R_b_fk)
                _base_pos_w = env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                _base_quat_w = env.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
                print(f"[fk_check] env right_eef pos_w={_re_pos_w.tolist()} quat_w={_re_quat_w.tolist()}")
                print(f"[fk_check] robot base pos_w={_base_pos_w.tolist()} quat_w={_base_quat_w.tolist()}")
                print(f"[fk_check] pos_b(from env→base xform)={_pos_b_from_env.tolist()}")
                print(f"[fk_check] pos_b(from pin FK on q)   ={_pos_b_fk.tolist()}")
                print(f"[fk_check] pos_err={_pos_err:.4f} m  R_err_fro={_R_err:.4f}")
                _dumped_fk_check = True

            if step % max(1, args_cli.use_length) == 0:
                t0 = time.time()
                action_dict, info = client.get_action(payload)
                if action_dict is None:
                    print(f"  step {step}: server returned None — abort")
                    break

                def _ax(name):
                    a = np.asarray(action_dict[name], dtype=np.float32)
                    return a[0] if a.ndim == 3 else a

                chunk_eef = _ax("right_eef")    # (T, 9)
                chunk_hands = _ax("right_hands")  # (T, 12)
                print(f"  step {step}: chunk eef={chunk_eef.shape} "
                      f"hands={chunk_hands.shape} infer={time.time() - t0:.2f}s")

            t_in_chunk = step % chunk_eef.shape[0]
            eef9_world = chunk_eef[t_in_chunk]
            ha = chunk_hands[t_in_chunk]

            # Run deploy-style Pink IK on the right arm.
            t_ik = time.time()
            right_arm_q = ik.solve_right(cur_joint_pos, env_to_pin, eef9_world)
            ik_dt = time.time() - t_ik

            # Per-tick rate limit on arm joints (mirrors deploy --max_delta_rad).
            if args_cli.max_delta_rad > 0.0:
                cur_right_arm = cur_joint_pos[right_arm_env_idx].astype(np.float32)
                delta = np.clip(right_arm_q - cur_right_arm,
                                -args_cli.max_delta_rad, args_cli.max_delta_rad)
                right_arm_q = cur_right_arm + delta

            # Compose 53-d JointPositionAction target: hold all measured, override
            # right arm with IK and right hand with policy hand qpos.
            target = cur_joint_pos.copy()
            target[right_arm_env_idx] = right_arm_q
            target[right_hand_env_idx] = ha

            action_t = torch.from_numpy(target).to(args_cli.device).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(action_t)
            term = bool(terminated[0]) if hasattr(terminated, "__getitem__") else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, "__getitem__") else bool(truncated)
            if step % 30 == 0:
                _t = ik.last_pos_target_b
                _a = ik.last_pos_achieved_b
                _dq = ik.last_q_right_arm_solved - ik.last_q_right_arm_init
                _eef_target_w = eef9_world[0:3]
                print(f"  step {step}: ik_dt={ik_dt*1000:.1f}ms "
                      f"target_w={_eef_target_w.tolist()} "
                      f"target_b={_t.tolist()} achieved_b={_a.tolist()} "
                      f"pos_err={ik.last_pos_err_m*1000:.1f}mm "
                      f"|dq_arm|={np.linalg.norm(_dq):.4f}")
            if term or trunc:
                success_flag = False
                done_mgr = getattr(env, "termination_manager", None)
                if done_mgr is not None:
                    s_buf = done_mgr.get_term("success") if "success" in done_mgr.active_terms else None
                    if s_buf is not None:
                        success_flag = bool(s_buf[0])
                print(f"  step {step}: terminated={term} truncated={trunc} success={success_flag}")
                success = success_flag
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(args_cli.max_steps)
            print(f"  episode {ep} hit max_steps={args_cli.max_steps}")
        if success:
            successes += 1
        print(f"[eval] episode {ep}: success={success} len={episode_lengths[-1]}")

    print(f"[eval] DONE — {successes}/{args_cli.episodes} success "
          f"(avg len {np.mean(episode_lengths):.1f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
