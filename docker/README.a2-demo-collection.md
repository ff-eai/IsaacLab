# A2 demo collection with the container

Runbook for recording A2 place-can-in-tray demos from the Pico 4 Ultra and turning
them into a synthetic dataset with Isaac Mimic, using the `isaac-lab-x2mimic` image.

The image is named for X2 because that is what it was built for, but it contains the
whole repo — the A2 task, the A2 mimic env, `assets/A2`, and the RoboTwin can/tray
assets are all present, and every A2 gym id is registered:

```
Isaac-PickPlace-A2-Abs-v0        teleop recording
Isaac-PickPlace-A2-Mimic-v0      annotation + Isaac Mimic generation
Isaac-PickPlace-A2OmniHand-Abs-v0
```

> **Status:** the container has been verified for X2 only (`x2-mimic verify`). The A2
> files and assets are confirmed present inside the image, but a full A2 run in the
> container has not been done yet. The host-side A2 pipeline does work.

## 0. Before you start

| Check | Why |
|---|---|
| `nvidia-smi` shows a GPU with ≥20 GB free | Isaac Sim with cameras needs it. On this box GPU 0 usually hosts vLLM, so use GPU 1. |
| TLS cert matches this machine's IP | WebXR needs a secure context; a name mismatch kills the session. |
| Pico hand tracking on, controllers **off/asleep** | Isaac Lab's `handtracking` device reads XR hand joints. |
| `xhost +local:docker` (if the Isaac Sim window won't open) | The container draws on the host `DISPLAY`. |

Reissue the cert whenever the host IP changes, then update `docker/.env.x2mimic`:

```bash
cd webxr_certs && mkcert <new-ip> localhost 127.0.0.1
# set TLS_CERT / TLS_KEY in docker/.env.x2mimic
```

The mkcert root CA is unchanged, so a headset that already trusts it needs no reinstall.

## 1. Bring up the stack

```bash
cd docker
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic up -d
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic ps
```

Three services come up: `x2-mimic` (Isaac Lab), `cloudxr-runtime`, `webxr` (client +
signaling proxy on `:8443`). Confirm the runtime is listening:

```bash
ss -ltn | grep 49100        # CloudXR signaling (auto-webrtc moves it off 48010)
ss -ltn | grep 8443         # WebXR client + proxy
```

## 2. Record

```bash
cd docker
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic \
  exec -e RECORD_TASK=Isaac-PickPlace-A2-Abs-v0 \
       -e MIMIC_TASK=Isaac-PickPlace-A2-Mimic-v0 \
       x2-mimic x2-mimic record /workspace/datasets/a2_pickplace_$(date +%Y%m%d_%H%M).hdf5
```

`/workspace/datasets` is bind-mounted from the repo's `datasets/`, so recordings land
on the host.

Then, in order:

1. In the Isaac Sim window on the host display: **AR panel → Start AR**.
2. On the Pico: open `https://<host-ip>:8443/`.
3. Select the **Pico 4 Ultra** device preset — not `custom`. It sets
   `perEyeWidth 2048 / perEyeHeight 1792`, which is what the server expects.
4. **CONNECT.**

Recording begins immediately (see `--xr_autostart` below). Press **R** to discard the
current attempt and reset.

### Reading the success debug line

A2 uses the four-condition lift-latch success term, and prints its state every 30 steps:

```
[Success_DBG] in_xy=False in_z=True lifted=False released=False -> success=False
   xy_dist=0.304/0.10  z_offset=0.074 (band [0.00,0.10])
   can_z=1.100 (lift>1.05)  min_hand_dist=0.531 (grasp<0.20, release>0.15)
```

| Field | Means |
|---|---|
| `min_hand_dist` | Hand-to-can distance. **This is the liveness signal** — if it does not change as you move, teleop is not reaching the robot. Must drop below `0.20` to latch the grasp. |
| `in_xy` | Can within `0.10` m of the tray in xy. |
| `in_z` | Can resting in the on-surface band above the tray. |
| `lifted` | Grasp latched: hand close, then can held above the table for 0.5 s. |
| `released` | Both hands ≥ `0.15` m from the can after the lift latched. |

All four must hold at once. If a placement looks right but never registers, `xy_dist`
is usually the culprit — the live env wants `0.10`, while the mimic config loosens it
to `0.35` for generation.

## 3. Annotate and generate

```bash
cd docker
C="docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic exec -e MIMIC_TASK=Isaac-PickPlace-A2-Mimic-v0 x2-mimic"

$C x2-mimic inspect  /workspace/datasets/<recording>.hdf5     # episode count + task
$C x2-mimic annotate /workspace/datasets/<recording>.hdf5      # -> *_annotated.hdf5
$C x2-mimic generate /workspace/datasets/<annotated>.hdf5      # -> *_generated.hdf5
```

`generate` honours `NUM_ENVS` (default 4) and `NUM_TRIALS` (default 500), e.g.
`-e NUM_TRIALS=200`. `x2-mimic all <recording>.hdf5` runs annotate + generate together.

Generation is headless and needs no headset, so it can run while you do something else.

## 4. Things that will bite you

**Teleop will not start on its own without `--xr_autostart`.** In XR mode
`record_demos.py` idles until a `START` teleop command arrives, and the bundled
CloudXR.js client has no teleop UI to send one — it has zero references to "teleop".
The retargeter still runs, so hands track in the viewport while `env.step()` is never
called and the robot holds its reset pose. The `x2-mimic record` helper passes
`--xr_autostart` by default; set `XR_AUTOSTART=0` only if you switch to a client that
can send the command (the NVIDIA-hosted one).

**A silent IK freeze looks exactly like broken hand tracking.** Pink IK defaults to the
`daqp` QP solver. If that solver is missing or ABI-mismatched, every solve raises,
`pink_ik.py` catches it broadly and returns the *current* joint positions, and the X2/A2
configs set `show_ik_warnings=False` — so the arm sits perfectly still, **the fingers
keep moving** (they bypass IK), and nothing is logged. `isaaclab` pins `daqp==0.7.2`,
whose `solve()` predates the `primal_start` argument current `qpsolvers` passes; the
image installs **0.9.1** and asserts at build time that a QP actually solves. If you
rebuild from a different base, check:

```bash
python -c "import numpy as np; from qpsolvers import solve_qp; print(solve_qp(np.eye(2), np.array([1.,1.]), solver='daqp'))"
```

**No image in the headset = resolution mismatch.** Check the runtime log for:

```
ERROR [processSystemInfo] Ignoring streaming dimensions 4096x4032, expected 2048x1792
```

The stream connects and is then discarded on a geometry mismatch. Use the **Pico 4
Ultra** client preset rather than typing values into the custom fields.

**Do not set `NV_DEVICE_PROFILE=pico4` or `pico4u`.** Those strings exist only in the
6.0.5 libraries overlaid into the hybrid image; the 5.0.1 daemon underneath does not
implement them and exits 0 at startup. Working profiles: `auto-webrtc` (default),
`quest3`, `apple-vision-pro`.

**XR runs on CPU physics by default.** Isaac Lab forces `device=cpu` for XR sessions
unless `--device` is passed. To override: append `--device cuda:0` to the record command.

## 5. Quick triage

| Symptom | Look at |
|---|---|
| Hands visible, arm frozen | IK solver (`daqp`), or missing `--xr_autostart` |
| Fingers move, arm frozen | IK solver specifically — fingers bypass IK |
| Nothing moves at all, `min_hand_dist` constant | `--xr_autostart`; env is not being stepped |
| Connected but no picture | Client resolution preset |
| Client won't connect | Cert vs host IP; `docker compose logs webxr` |
| Runtime container exits immediately | `NV_DEVICE_PROFILE` set to an unsupported profile |

Logs: `docker compose logs cloudxr-runtime`, `docker compose logs webxr`, and the
CloudXR server log inside the runtime container at `/tmp/com.nvidia.CloudXR_*/cxr_server.*.log`.
