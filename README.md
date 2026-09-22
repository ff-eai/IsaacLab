# Isaac Lab Mimic &rarr; GR00T VLA Training Guide

Fork of [Isaac Lab](https://github.com/isaac-sim/IsaacLab) carrying the **Futurist** bimanual
place-can-in-tray pipeline: VR teleoperation capture, Isaac Mimic data multiplication, LeRobot
conversion, GR00T fine-tuning, and closed-loop evaluation.

Upstream documentation still applies for anything not covered here; see the
[Isaac Lab docs](https://isaac-sim.github.io/IsaacLab) and
[upstream README](https://github.com/isaac-sim/IsaacLab/blob/main/README.md).

---

## Overview: a six-stage pipeline

Isaac Lab Mimic takes **about ten human teleoperation demos** and multiplies them into **1,000+
successful trajectories** by segmenting each demo into subtasks, transforming those segments to new
object poses, and stitching them back together. GR00T then converts those trajectories into a
LeRobot dataset and fine-tunes a VLA policy on them.

```
teleop capture ──▶ subtask annotation ──▶ Mimic expansion ──▶ [Cosmos Transfer] ──▶
    convert to LeRobot ──▶ GR00T N1.x fine-tune ──▶ closed-loop eval in Isaac Sim
```

Every stage leaves an artifact on disk you can inspect. When something breaks, work backwards up
this table:

| Stage | Command | Artifact | Typical scale |
|---|---|---|---|
| Capture | `scripts/tools/record_demos.py` | `futurist_pickplace.hdf5` | 10–30 demos |
| Annotate | `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py` | `futurist_pickplace_annotated.hdf5` | same + subtask boundaries |
| Expand | `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` | `futurist_generated_1000.hdf5` | 1,000+ demos |
| Augment | `scripts/tools/cosmos/cosmos_prompt_gen.py` + Cosmos Transfer | multi-style frames | ×N visual variants |
| Convert | `scripts/imitation_learning/convert_annotated_ee_to_lerobot.py` | LeRobot v2.1 dataset | parquet + mp4 |
| Train | `gr00t/experiment/launch_finetune.py` | checkpoint | 10k–60k steps |

The first five commands ship with this repository; the last with Isaac-GR00T.

> **Why Futurist rather than the Franka example.** The upstream tutorial uses
> `Isaac-Stack-Cube-Franka-IK-Rel-v0`: single arm, parallel gripper, relative IK. Futurist is
> bimanual with dexterous hands and absolute Pink IK, which is the configuration most real VLA work
> needs and the one where the sharp edges live. Everything here applies to any Mimic task; the task
> IDs and dimensions change.

### Environments

| Gym ID | Hand | Role |
|---|---|---|
| `Isaac-PickPlace-A2-Abs-v0` | s6_hand (12 finger joints) | Teleop recording; the Mimic env derives from this config |
| `Isaac-PickPlace-A2-Mimic-v0` | s6_hand | Annotation and Mimic generation |
| `Isaac-PickPlace-A2OmniHand-Abs-v0` | OmniHand T2 (10 DoF driven) | Default env of the GR00T v5 eval script; **no Mimic env exists for it** |

> **Hand mismatch to decide up front.** GR00T v5's eval defaults to the OmniHand env, while Mimic
> generation only exists for the s6_hand env. Either record and train on s6_hand and change
> `--task_id` at eval, or add an OmniHand Mimic env. Do not mix: state and action widths differ.

---

## Where to get the source code

**Isaac Lab Mimic is not a separate download.** It ships inside this repository as the
`isaaclab_mimic` package (`source/isaaclab_mimic/`) plus three scripts under
`scripts/imitation_learning/isaaclab_mimic/`.

| Component | Source | Notes |
|---|---|---|
| Isaac Lab (includes Mimic) | github.com/isaac-sim/IsaacLab | The only clone needed for stages 1–3 |
| Isaac Sim | github.com/isaac-sim/IsaacSim | Source build or binary release |
| Isaac-GR00T | github.com/NVIDIA/Isaac-GR00T | Training + inference code |
| GR00T base checkpoint | `nvidia/GR00T-N1.7-3B` | ~3B params, bfloat16 |
| VLM backbone (gated) | `nvidia/Cosmos-Reason2-2B` | Must click **Agree** or downloads 401 |
| Cosmos Transfer | `nvidia/Cosmos-Transfer1-7B` | Visual augmentation, needs 80 GB VRAM |
| LeRobot | github.com/huggingface/lerobot | Dataset format used by GR00T |

```bash
# Simulation side
./isaaclab.sh --install

# Training side
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && uv sync --python 3.10
```

---

## Recommended PC configuration

Two workloads with opposite hardware profiles, and one cannot run on the other's machine.
Simulation needs an x86_64 box with a display stack; fine-tuning needs 40 GB+ of VRAM per GPU.

### Simulation workstation (stages 1–3), VR capture

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 8-core x86_64 | 16-core Threadripper Pro or better |
| RAM | 32 GB | 64 GB |
| GPU | RTX A4000 16 GB | **RTX A6000 48 GB** |
| VRAM | 16 GB | 48 GB — multi-env generation plus camera rendering |
| Storage | 500 GB NVMe | 2 TB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 or 24.04 |
| VR headset | — | Quest 3 or Pico 4 Ultra over CloudXR |
| Network | — | Dedicated Wi-Fi 6 router |
| Container stack | — | Docker 26.0+, Compose 2.25+, NVIDIA Container Toolkit |

The headset and the workstation must be IP-reachable from each other, and many institutional
wireless networks block exactly that. Use a dedicated router.

> **Multi-GPU caution.** If the workstation has two GPUs and the second is busy with another
> workload, restrict Isaac Sim to one card. With both visible it builds a multi-GPU render graph and
> deadlocks on a copy-queue barrier — the app stops responding with no further log output. Use
> `CUDA_VISIBLE_DEVICES`, or a single-GPU device reservation in the container runtime.

### Training node (stages 5–6)

| Setup | GPUs | VRAM/GPU | Global batch | Use case |
|---|---|---|---|---|
| Prototyping | 1× H100 / L40 / A100 | 40–80 GB | 32 | Demo datasets |
| Recommended | 4–8× H100 or L40 | 40–80 GB | 64–640 | Normal fine-tuning |
| Full scale | 8× RTX Pro 6000 or DGX | 96 GB | 640 | Production |

Default fine-tuning only trains the projector and diffusion action head, keeping peak VRAM under
~35 GB per GPU. Adding `--tune-llm` or `--tune-visual` pushes that past 80 GB.

### Cosmos augmentation node (stage 4)

Cosmos-Transfer1-7B needs a single GPU with **80 GB VRAM** and **must run on a node separate from
the simulation**.

---

## Prerequisites and machine split

| Stage | Machine | Why |
|---|---|---|
| Teleop capture | x86_64 workstation with a display stack | Needs GUI/VR; the Isaac Sim XR extension does not exist on aarch64 |
| Mimic annotate + expand | Same workstation, `--headless` | Pure simulation; runs unattended overnight |
| Cosmos augmentation | Separate 80 GB-VRAM node | Transfer1-7B cannot share a GPU with Isaac Sim |
| GR00T fine-tune | Multi-GPU node, 40 GB+ per GPU | 3B model with multi-camera input |
| Inference / closed-loop eval | Single GPU, 16 GB+ | Workstation or edge device |

> **Pinocchio is required.** Bimanual and dexterous-hand tasks use Pink IK, so `--enable_pinocchio`
> is needed on the capture, annotate and generate steps. Omitting it fails at env construction.

> **Check the IK solver before your first capture.** Pink IK defaults to the `daqp` QP solver. If it
> is missing or ABI-mismatched, every solve raises, the controller returns the current joint
> positions, and `show_ik_warnings=False` hides it — the arm sits perfectly still while the fingers,
> which bypass IK, keep moving.
>
> ```bash
> python -c "import numpy as np; from qpsolvers import solve_qp; \
>   print(solve_qp(np.eye(2), np.array([1.,1.]), solver='daqp'))"
> ```
>
> A `SolverNotFound`, or a `TypeError` about `primal_start`, means the installed `daqp` does not
> match the installed `qpsolvers`.

---

## Step 1: capture source demos

Record against the **non-Mimic task ID**. The Mimic variant is only used from annotation onwards.

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
   --task Isaac-PickPlace-A2-Abs-v0 \
   --teleop_device handtracking \
   --dataset_file ./datasets/futurist_pickplace.hdf5 \
   --num_demos 20 \
   --enable_pinocchio --enable_cameras \
   --xr --xr_autostart
```

`handtracking` routes through `OpenXRDevice` and needs the CloudXR runtime up and the headset
connected first. With VR the environment drives its own rate limiting via OpenXR, so `--step_hz`
behaves differently than on a desktop device.

| Flag | Default | How to choose |
|---|---|---|
| `--teleop_device` | `keyboard` | `handtracking` for VR; a custom name works if the env defines it under `teleop_devices` |
| `--num_demos` | 0 (infinite) | 10–30. Mimic does the multiplying; extra demos have sharply diminishing returns |
| `--step_hz` | 30 | Match your downstream GR00T frame rate; with `handtracking`, OpenXR paces the loop |
| `--num_success_steps` | 10 | Consecutive successful frames before a demo counts as complete |
| `--enable_pinocchio` | off | Required for bimanual / dexterous-hand tasks |
| `--xr_autostart` | off | Steps the env immediately instead of waiting for a START teleop command |

> **Hands track but the robot does not move.** In XR mode the capture loop idles until a `START`
> teleop command arrives. A browser-based CloudXR client with no teleop UI never sends one, so the
> retargeter keeps running — hands animate in the viewport — while `env.step()` is never called and
> the robot holds its reset pose. `--xr_autostart` skips that wait.

Demo quality matters far more than demo count. Mimic slices source trajectories into segments and
re-stitches them, so:

- every demo must follow the same subtask order, with no mid-demo retries;
- grasp and release moments must be clean — they are the segment boundaries, and a hesitant grasp
  makes the termination signal chatter;
- spread the initial object poses out. Twenty demos staged at the same spot will not cover the
  workspace, however many you expand them into.

Replay with `scripts/tools/replay_demos.py` afterwards and drop anything that failed or dragged.

---

## Step 2: subtask annotation

Annotation tells Mimic which frame each subtask ends on. The cut points come from binary signals in
`datagen_info`, taken at the 0→1 edge.

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
   --task Isaac-PickPlace-A2-Mimic-v0 \
   --input_file ./datasets/futurist_pickplace.hdf5 \
   --output_file ./datasets/futurist_pickplace_annotated.hdf5 \
   --auto --enable_pinocchio
```

`--auto` segments using the termination signals already defined in the environment; without it you
step through frame by frame and mark boundaries by hand, which is what new tasks need.

### How this task defines its subtasks

Config in `source/isaaclab_mimic/isaaclab_mimic/envs/pinocchio_envs/pickplace_a2_mimic_env_cfg.py`,
signal in the matching `pickplace_a2_mimic_env.py`. The task is one-handed, so the right arm carries
two subtasks and the left is held passively:

| Field | Purpose | Value |
|---|---|---|
| `object_ref` | Which object frame the subtask is transformed relative to | `object` (the can), then `tray` |
| `subtask_term_signal` | Binary signal marking the end of this segment | `idle_right`, then `None` |
| `subtask_term_offset_range` | Random jitter on the boundary, for diversity | `(0, 10)` |
| `selection_strategy` | How a source segment is picked for transformation | `nearest_neighbor_object` |
| `action_noise` | Noise added during this segment | `0.003` |
| `num_interpolation_steps` | Bridging frames between segments | `10` |

The last subtask sets `subtask_term_signal` to `None`, meaning it runs to the end of the episode.

> **Choose a discriminating termination signal.** A grasp signal built only on
> end-effector-to-object distance can be ambiguous: on a dexterous hand the wrist frame may sit as
> far from the object during a genuine grasp as at the rest pose, so the signal fires on an object
> that was merely nudged. Combine proximity with finger closure, or with a lift relative to the
> object's resting height, and keep the margin wide.

A new task must subclass `ManagerBasedRLMimicEnv`, implement the pose/action conversion methods, and
define `get_subtask_term_signals()` — that is what `--auto` reads.

---

## Step 3: Mimic expansion

For each new initial scene, Mimic pulls the matching source segment per subtask, moves the
end-effector trajectory over using the relative transform between old and new object poses,
interpolates between segments, then replays the result in simulation and judges it with the task's
own success condition. **Failed trajectories are discarded by default.**

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
   --task Isaac-PickPlace-A2-Mimic-v0 \
   --input_file ./datasets/futurist_pickplace_annotated.hdf5 \
   --output_file ./datasets/futurist_generated_1000.hdf5 \
   --generation_num_trials 1000 \
   --num_envs 10 \
   --enable_pinocchio \
   --headless
```

| Flag | Meaning |
|---|---|
| `--generation_num_trials` | How many demos you want |
| `--generation_guarantee` | On by default: retry until that many successes |
| `--num_envs` | Parallel envs; 10–20 is stable on a 48 GB card, lower with cameras enabled |
| `--headless` | Mandatory for batch generation — rendering is far slower |
| `--enable_pinocchio` | Same as the annotate step |
| `--use_skillgen` | SkillGen (cuRobo collision-aware planning); higher success rate in cluttered scenes |

`generation_keep_failed=True` stores failed trajectories so you can replay them and see which
segment stalls; `max_num_failures` caps how many failures are tolerated before giving up.

> **Cameras are off by default in the Mimic config.** `camera_enabled` defaults to `False` because
> rendering slows generation considerably. A dataset generated that way contains states but **no
> images**, and cannot train a vision policy. Set it to `True` and pass `--enable_cameras` when
> generating VLA training data. Check which camera set the config enables — a policy trained on
> wrist cameras cannot be fed a three-camera dataset.

### Reading the success rate

A healthy task lands between 50% and 80%. Below 20% the problem is upstream, not in these flags:

- subtask boundaries or the transform reference frame are wrong — re-check `object_ref`;
- target poses fall outside the arm's reachable workspace — tighten the initial-pose randomisation;
- `action_noise` is large enough to shake the grasp loose — drop it to 0.01 on grasp segments;
- the success threshold used during generation is stricter than what the demos actually achieve.

Generating 1,000 trajectories with several camera streams takes roughly a night on a 48 GB card.

---

## Step 4 (optional): Cosmos visual augmentation

Mimic multiplies **motion** diversity; every trajectory still renders in exactly the same style. To
stop GR00T's visual encoder from overfitting to simulation textures, Cosmos Transfer repaints the
appearance while holding geometry and motion fixed.

**When to skip it:** closed-loop validation inside simulation does not need it. Real-robot
deployment does.

```bash
./isaaclab.sh -p scripts/tools/cosmos/cosmos_prompt_gen.py \
   --templates_path scripts/tools/cosmos/transfer1_templates.json \
   --num_prompts 50 \
   --output_path ./datasets/cosmos_prompts.txt
```

Two hard constraints: Cosmos must run on a node separate from the simulation, and you need a
camera-equipped dataset for there to be anything to augment.

---

## Step 5: HDF5 → LeRobot dataset

Isaac Lab writes its own HDF5 layout; GR00T reads LeRobot v2.1. Upstream ships no converter, but
this repository does, under `scripts/imitation_learning/`:

| Script | State / action | Cameras emitted |
|---|---|---|
| `convert_annotated_ee_to_lerobot.py` | 67-d state, 38-d action | `cam_high`, `cam_chest_left`, `cam_chest_right`, depth |
| `convert_isaac_a2_to_lerobot_ee.py` | EE variant | same three + depth |
| `convert_annotated_to_pi07_lerobot.py` | pi0.7 layout | same three + depth, plus subtask language and quality metadata |
| `convert_isaac_a2_to_lerobot.py` | 41-DOF compact joints | three RGB; needs a joint-name JSON |

```bash
python scripts/imitation_learning/convert_annotated_ee_to_lerobot.py \
   --input_file  ./datasets/futurist_generated_1000.hdf5 \
   --repo_id     futurist_pickplace_ee_v1 \
   --task        "place the can in the tray" \
   --lerobot_home <lerobot-cache> \
   --fps 30

python scripts/imitation_learning/build_lerobot_norm_stats.py \
   --dataset_root <lerobot-cache>/futurist_pickplace_ee_v1 \
   --out          <lerobot-cache>/futurist_pickplace_ee_v1/norm_stats.json
```

State layout: `robot_joint_pos(53) + left_eef_pos(3) + left_eef_quat(4) + right_eef_pos(3) +
right_eef_quat(4) = 67`. Action: `left IK 7 + right IK 7 + hand 24 = 38`. Conversion needs
`lerobot`, `h5py` and `cv2`, but not Isaac Sim.

### Two things that must line up

`meta/modality.json` describes how the state and action vectors are sliced. Get a slice off by one
dim and training will not error; the policy simply learns nothing.

The modality config (a Python file, not JSON) tells GR00T how to use those keys:

```python
futurist_config = {
    # keys must match the "video" entries in meta/modality.json
    "video": ModalityConfig(delta_indices=[0],
                            modality_keys=["cam_high", "cam_chest_left", "cam_chest_right"]),
    "state": ModalityConfig(delta_indices=[0],
                            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"]),
    "action": ModalityConfig(delta_indices=list(range(0, 16)),   # 16-step horizon
                             modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
                             action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                                          type=ActionType.NON_EEF,
                                                          format=ActionFormat.DEFAULT)] * 4),
    "language": ModalityConfig(delta_indices=[0],
                               modality_keys=["annotation.human.task_description"]),
}
register_modality_config(futurist_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

Custom robots always use `EmbodimentTag.NEW_EMBODIMENT`; reusing an existing tag such as `GR1` or
`OXE_DROID` fails on dimension mismatch.

After converting, pull one episode and check four things: frame count, number of camera streams,
whether `task_index` is uniformly 0, and whether normalisation statistics exist. Any one of those
being wrong crashes training in the first epoch.

---

## Step 6: GR00T N1.x fine-tune

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base_model_path <models-dir>/GR00T-N1.7-3B \
  --dataset_path <lerobot-cache>/futurist_pickplace_ee_v1 \
  --modality_config_path examples/<your_robot>/config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 4 \
  --output_dir <runs-dir> \
  --experiment_name futurist_pickplace_mimic_from_base \
  --max_steps 10000 --save_steps 1000 --save_total_limit 5 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --weight_decay 1e-5 \
  --global_batch_size 320 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers 4 \
  --state_dropout_prob 0.2 \
  --use_wandb
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `--max_steps` | 10000 | Enough for a single task; 20k–60k past ~10k episodes |
| `--global_batch_size` | 320 | Stable on 4× 80 GB cards; lower this first if VRAM is tight |
| `--learning_rate` | 1e-4 | From base. Drop to 5e-5 when continuing from a checkpoint |
| `--color_jitter_params` | as above | Mandatory for pure-simulation data; partly compensates for skipping Cosmos |
| `--state_dropout_prob` | 0.2 | Stops the policy reading proprioception and ignoring the images |

Simulation data has one specific weakness: it is too clean. Every trajectory succeeded, and the noise
is identically distributed. Keep colour jitter generous, add state noise where needed, or the policy
scores perfectly in simulation and collapses on hardware.

If the simulation data is meant as pre-training: train from base on the Mimic thousands for 10k
steps, then continue from that checkpoint on a few dozen real-robot episodes by passing the
checkpoint directory as `--base_model_path`.

---

## Step 7: evaluation and replay

Three tiers, fastest first:

1. **Offline action MSE.** Score predicted against ground-truth actions on held-out episodes.
   Reasonable MSE but failing closed loop means the problem is error accumulation, not fit.
2. **Stand up an inference server.** `scripts/imitation_learning/launch_groot_v5_policy_server.sh`
   wraps this; host, port, GPU index and checkpoint are all environment overrides.
3. **Closed-loop simulation.** Drive the Isaac Lab env from that policy server: reset, query, step,
   count successes. Run 20–50 episodes before the number means anything.

```bash
./isaaclab.sh -p scripts/imitation_learning/eval_groot_isaac_a2_http_v5_5cam_abs_ee.py \
   --groot_host <server-host> --groot_port <port> \
   --task_id Isaac-PickPlace-A2-Abs-v0 \
   --task "place the can in the tray" \
   --episodes 20 --max_steps 2000 --enable_cameras
```

Evaluate in the same environment you captured in, but **change the initial-pose random seed**. A
success rate measured on the training set's own initial conditions tells you nothing.

Only go to hardware once closed-loop simulation passes. The main remaining gaps are camera
intrinsics and mounting pose, control rate, and whether actions are relative or absolute. Align all
three at conversion time, not after training.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Generation success rate < 20% | Wrong `object_ref`, mis-marked subtask boundaries, or too strict a success threshold | Set `generation_keep_failed=True` and replay failures |
| Generation hangs forever | `generation_guarantee` on with a near-zero success rate | Turn it off and run a small batch to measure the real rate |
| Hands track in the viewport, robot frozen | XR loop waiting for a START teleop command the client cannot send | `--xr_autostart` |
| Fingers move, arm frozen, nothing logged | Pink IK solver missing or ABI-mismatched; failure returns current joint positions, warnings suppressed | Verify the QP solver; enable `show_ik_warnings` while debugging |
| Isaac Sim stops responding, log goes silent | Multi-GPU render graph deadlock when a second GPU is busy | Restrict the process to one GPU |
| Generated dataset has no images | `camera_enabled=False` in the Mimic config | Enable it and pass `--enable_cameras` |
| Loader raises an `iloc` out-of-range | `task_index` is not a contiguous run starting at 0 | Normalise it to 0 in the conversion script |
| NaN on the first training step | Missing normalisation stats, or a zero-variance dimension | Recompute the dataset statistics |
| Base model download 401s | Not logged into HF, or Cosmos-Reason2 licence not accepted | `hf auth login` plus clicking Agree on the model page |
| Teleop will not start on an ARM box | aarch64 has no `omni.kit.xr.system.openxr` | Move capture to an x86_64 machine |
| Perfect in sim, fails on hardware | Simulation data too clean, no visual or dynamics noise | Add Cosmos augmentation, raise colour jitter, continue on real data |

**One precondition that is easy to miss.** Mimic only expands scenarios that share a task, share a
subtask sequence, and differ solely in object pose. It will not invent new skills, and it will not
produce variants that need a different action sequence — "move the obstacle aside first when the
object is occluded", for instance. Behaviour like that means capturing new demos, or handing motion
planning to cuRobo via SkillGen.

### Containerised teleop stack

`docker/` carries a Compose stack (Isaac Lab + CloudXR runtime + WebXR front end) for the capture
workflow. See `docker/README.a2-demo-collection.md` for the runbook and
`docker/README.x2mimic.md` for image build details.

---

## Repository conventions

Inherited from [`ff-eai/repo-template`](https://github.com/ff-eai/repo-template).

Install the shared git hooks once per clone:

```bash
./.githooks/setup-hooks.sh
```

### Jira credentials (one-time, per developer)

The `commit-msg` hook calls the Jira API, so it needs your own Jira account credentials. Create an
API token at [id.atlassian.com](https://id.atlassian.com/manage-profile/security/api-tokens) and add
both to your shell profile so they persist:

```bash
export JIRA_EMAIL="you@example.com"
export JIRA_TOKEN="your-jira-api-token"
```

**Both are required for every commit.** Without them every commit is rejected.

### Commit and PR conventions

Commit subjects and PR titles must start with a Jira key:

```
EAI-1234: short description of the change
```

Enforced in three places — the org ruleset, the `jira-validate` CI job, and the local `commit-msg`
hook, which also refuses the commit unless the ticket's status is `Resolved`. So the intended flow
is: finish the work → move the ticket to **Resolved** in Jira → commit.

`main` is protected: no direct pushes. Work on a branch, open a PR, and squash merge once
`jira-validate` is green.

| Path | Purpose |
|---|---|
| `.githooks/` | Shared `pre-commit` and `commit-msg` checks, plus the installer |
| `.githooks-config.json` | Hook rules (Jira project, protected branches, file-size limits) |
| `.github/workflows/jira-validate.yml` | Calls the org's reusable PR validation workflow |
| `.github/workflows/jira-tag-sync.yml` | Calls the org's reusable release tag sync workflow |

Those files are kept current automatically by the `template-sync` workflow in
[`ff-eai/.github`](https://github.com/ff-eai/.github); avoid editing them locally.

---

## Sources

- Isaac Lab repository, its teleoperation / imitation-learning tutorial, and the CloudXR
  teleoperation guide
- Synthetic Manipulation Motion Generation blueprint
- Isaac-GR00T: hardware recommendation, data preparation, new-embodiment fine-tuning
- LeRobot dataset format
- In-repo: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/`,
  `source/isaaclab_mimic/isaaclab_mimic/envs/pinocchio_envs/`, `scripts/imitation_learning/`,
  `docker/`

## License

Isaac Lab is released under the [BSD-3 License](LICENSE); `isaaclab_mimic` under
[Apache 2.0](LICENSE-mimic). Upstream dependencies retain their own licenses.
