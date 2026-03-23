# Static-Core Refactor Log (SplineGS -> 3DGS Core)

This file tracks every step completed to follow the requested scope:

- Keep only:
  - 3DGS scene representation
  - COLMAP-free pose initialization
  - Differentiable renderer
  - Basic photometric loss
- Strip out Gaussian trajectory spline dynamics from the **active training path**

## What Was Done

### 1) Added a static-core training entrypoint

- Created `train_static_core.py`.
- This script is now the simplified path that avoids spline trajectory modeling.

### 2) Preserved 3DGS scene representation

- Uses `GaussianModel` and `Scene`.
- Trains static gaussians (`stat_gaussians`) as the scene representation.

### 3) Preserved COLMAP-free pose initialization

- Uses `pose_network` through `GaussianModel.create_pose_network(...)`.
- Camera pose is predicted from time/depth and applied with `update_cam(...)`.

### 4) Preserved differentiable renderer

- Uses `gaussian_renderer.render_static(...)`.
- Keeps rasterization-based differentiable rendering in the optimization loop.

### 5) Reduced loss to basic photometric objective

- Uses only L1 + optional DSSIM:
  - `l1_loss(pred, gt)`
  - `lambda_dssim * (1 - ssim(pred, gt))`
- Removed dynamic-specific and geometry-consistency losses from the active path.

### 6) Removed spline dynamics from active training flow

- The static-core loop does **not** call:
  - `render(...)` (dynamic render path)
  - cubic Hermite interpolation functions
  - dynamic control-point updates
  - dynamic trajectory initialization

### 7) Added static-core outputs

- Saves static gaussian checkpoint:
  - `output/<expname>/point_cloud/static_core_final/point_cloud_static.ply`
- Saves pose network:
  - `output/<expname>/point_cloud/static_core_final/posenet.pth`

### 8) Added automated verification script

- Added `verify_static_core.py`.
- The script checks that all requested scope constraints are implemented in code.

### 9) Isolated dynamic spline stack behind an explicit legacy switch

- Added `train_entrypoint.py` to isolate runtime paths:
  - Default mode runs `train_static_core.py`.
  - Legacy dynamic mode runs original `train.py` only when `--legacy-dynamic` is set.
- This keeps spline dynamics available but quarantined from the default workflow.

### 10) GPU preflight (NVIDIA / CUDA)

- Added `preflight_gpu.py` to verify CUDA-enabled PyTorch and (when present) `nvidia-smi`.
- `train_entrypoint.py` runs this check **before** training unless you pass `--skip-gpu-preflight`.

## GPU: do you need NVIDIA?

**Yes, for real training.** This repo uses CUDA throughout (`.cuda()`, `simple-knn` CUDA extension, `gsplat` GPU rasterization). **CPU-only or Apple Silicon (MPS) will not work** without rewriting device placement and replacing CUDA kernels.

### Check your machine first

```bash
python preflight_gpu.py
# Optional machine-readable output:
python preflight_gpu.py --json
```

### If you don’t have a local NVIDIA GPU — how to still run it

1. **Cloud GPU (recommended)** — Rent a Linux instance with an NVIDIA GPU (e.g. Lambda Labs, RunPod, vast.ai, AWS g4dn/g5, Azure NC-series, Google Colab with GPU). Clone the repo there, install the same conda/pip stack as the main README, upload your `data/`, then:
   ```bash
   python train_entrypoint.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run
   ```
2. **Docker on a CUDA host** — Use an image with CUDA + PyTorch; host must have NVIDIA drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
3. **Remote SSH** — Develop on your laptop; sync code/data to a GPU server and run training over SSH.

There is **no supported “make it run on Mac CPU”** path for this codebase as-is.

## How To Run

```bash
# Recommended: preflight runs automatically
python train_entrypoint.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_static_core

# Skip preflight only if you know what you're doing (e.g. exotic setups)
python train_entrypoint.py --skip-gpu-preflight -s data/nvidia_rodynrf/<SCENE>/ --expname my_static_core

# Equivalent direct static call (no automatic preflight — run preflight_gpu.py first)
python preflight_gpu.py && python train_static_core.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_static_core
```

Legacy dynamic (isolated):

```bash
python train_entrypoint.py --legacy-dynamic -s data/nvidia_rodynrf/<SCENE>/ --expname my_legacy_run
```

Then verify:

```bash
python verify_static_core.py
```

## Notes

- Legacy SplineGS files are still present in the repository for reference.
- The new static-core path is intentionally isolated so you can iterate safely without dynamic spline behavior.
- Hard strip is deferred by request; this phase only isolates legacy dynamics.
