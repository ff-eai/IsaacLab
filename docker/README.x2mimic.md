# X2 demo collection + Isaac Mimic, containerised

Everything needed to record X2 (OmniHand T2) pick-place demos from a VR headset and
turn them into a synthetic dataset with Isaac Mimic.

Three services, one command:

| Service | Image | Role |
|---|---|---|
| `x2-mimic` | `isaac-lab-x2mimic` (built here) | Isaac Lab + X2 env + X2 mimic env + assets |
| `cloudxr-runtime` | `cloudxr-hybrid:5d-6l` | OpenXR runtime the headset streams to |
| `webxr` | `caddy:2-alpine` | Serves the WebXR client + proxies CloudXR signaling over HTTPS |

The CloudXR runtime is its own daemon on its own base image, so it stays a separate
service rather than being folded into the Isaac Lab image.

## Build

```bash
cd docker
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic build
```

Prerequisites on the host:

- `docker login nvcr.io` (the Isaac Sim base image and the CloudXR EA base)
- `cloudxr-hybrid:5d-6l` built locally — see `cloudxr-hybrid/Dockerfile`
- an NVIDIA GPU with the container toolkit

## Run

```bash
cd docker
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic up -d
docker compose -f docker-compose.x2mimic.yaml --env-file .env.x2mimic exec x2-mimic x2-mimic verify
```

`verify` is headless and needs no headset. It builds the mimic env, checks the 46-dim
action layout round-trips, and exercises the `idle_right` subtask signal. Run it before
committing a teleop session to anything.

## The pipeline

Inside the container, `x2-mimic` wraps the three stages:

```bash
x2-mimic record                          # teleop -> /workspace/datasets/x2_place_can_<ts>.hdf5
x2-mimic annotate  <recording>.hdf5      # mark subtask boundaries (--auto)
x2-mimic generate  <annotated>.hdf5      # Isaac Mimic generation (NUM_ENVS, NUM_TRIALS)
x2-mimic all       <recording>.hdf5      # annotate + generate
x2-mimic inspect   <file>.hdf5           # episode count + recorded task
```

`/workspace/datasets` is bind-mounted from `DATASET_HOST_DIR` (default: the repo's
`datasets/`), so recordings survive the container.

Keyboard instead of a headset:

```bash
TELEOP_DEVICE=keyboard x2-mimic record
```

## Recording with the Pico

1. `docker compose ... up -d` (runtime and proxy come up with it)
2. `x2-mimic record` — the Isaac Sim window opens on the host `DISPLAY`
3. In that window: **AR panel → Start AR**
4. On the headset: `https://<host-ip>:8443/` → select the **Pico 4 Ultra** device preset → CONNECT

## Things that will bite you

**The TLS cert must match the address the headset dials.** WebXR needs a secure
context, and a name mismatch kills it. When this machine's IP changes, reissue and
update `.env.x2mimic`:

```bash
cd webxr_certs && mkcert <new-ip> localhost 127.0.0.1
# then set TLS_CERT / TLS_KEY in docker/.env.x2mimic
```

The mkcert root CA stays the same, so a headset that already trusts it needs no
reinstall.

**Do not set `NV_DEVICE_PROFILE=pico4` or `pico4u`.** Those strings exist only in the
6.0.5 libraries overlaid into the hybrid image; the 5.0.1 daemon underneath does not
implement them and exits 0 immediately at startup. Profiles that work: `auto-webrtc`
(default), `quest3`, `apple-vision-pro`. `auto-webrtc` is also what moves signaling
from 48010 to 49100 and enables ICE — `NV_CXR_STREAMSDK_ENABLE_ICE=1` now states that
requirement outright so a profile change cannot silently drop it.

**Client resolution has to match what the server expects.** If the headset connects
but shows no image, check `docker compose logs cloudxr-runtime` (or the server log in
the container) for:

```
ERROR [processSystemInfo] Ignoring streaming dimensions 4096x4032, expected 2048x1792
```

That is the stream being discarded on a geometry mismatch. Both the `pico4ultra` and
`quest3` presets in the bundled client specify `perEyeWidth 2048 / perEyeHeight 1792`;
a `custom` profile with other values gets dropped.

**Hand tracking, not controllers.** Isaac Lab's `handtracking` device reads XR hand
joints. If the Pico controllers are awake they claim the hand slots and no joints are
sent — the robot then sits perfectly still while everything else looks healthy. The
runtime log names the culprit:

```
processControllerEquip] Equipping controller: 'Pico 4 Ultra Left' (0) in slot: 'hand/left'
```

Put the controllers down (or power them off) before connecting.

**numpy must stay below 2.** Isaac Sim's bundled pinocchio is compiled against
NumPy 1.x, so a 2.x numpy makes `import pinocchio` raise and then segfault, taking
every Pink IK task with it. `source/isaaclab/setup.py` asks for `numpy<2`, but
installing torch with `--ignore-installed` (which the image does, to place the CUDA
12 wheels in site-packages) drags in torch's unpinned numpy 2.x. The image therefore
reinstalls `numpy==1.26.0` afterwards and asserts at build time that numpy and
pinocchio import together. Keep that ordering if you edit the install steps.

**The base image's entrypoint hijacks arguments.** `nvcr.io/nvidia/isaac-sim` sets
`ENTRYPOINT ["/isaac-sim/runheadless.sh"]`, which ignores `CMD` and starts the
streaming app. A container run that way looks alive and logs happily while never
executing your command. This image replaces the entrypoint with `x2-mimic`. If you
ever see an Isaac Sim log with no `AppLauncher: Loading experience file` line, that
is what is happening.

**XR runs on CPU physics by default.** Isaac Lab forces `device=cpu` for XR sessions
unless `--device` is passed explicitly (`app_launcher.py`). To override:

```bash
x2-mimic record --device cuda:0
```

## Absolute paths

`pickplace_x2_env_cfg.py` resolves the merged X2+OmniHand URDF, its mesh root and the
RoboTwin can/tray USDs through `/home/wagner/code/IsaacLab/...`. The image symlinks
that path to the Isaac Lab checkout rather than patching the configs, which would
drift from the host tree. `HOST_REPO_PATH` in `.env.x2mimic` controls it; leave it
matching the configs.

## Iterating on env configs

The image contains a copy of the repo. To edit configs on the host and see the change
without rebuilding, uncomment the `../source` and `../scripts` bind mounts in
`docker-compose.x2mimic.yaml`.
