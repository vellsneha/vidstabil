# Training & rendering fixes (static core)

This document summarizes issues encountered when training with `train_static_core.py`, rendering stabilized video, and the code/data fixes applied in this workstream.

## Problems we hit

### 1. Empty `images_2` / dataset path

**Symptom:** `IndexError` in `readNvidiaCameras` when loading frames.

**Cause:** `cfg_args` stored `source_path` pointing at a location that did not exist on disk (e.g. `data/...` vs actual `data2/...`).

**Fix:** `render_stabilized_video.py` resolves `source_path` with `--source-path` and fallbacks (e.g. `data2/<scene_name>`).

### 2. Blank or gradient-only stabilized video

**Symptoms:** Entire video looks like a flat gradient, or `visible gaussians: 0` in render logs.

**Causes (multiple):**

- **Degenerate point cloud:** `readNvidiaInfo` in `scene/dataset_readers.py` used a dummy PLY built from a single zero point. Saved `point_cloud_static.ply` had **all XYZ at origin** and useless opacity — nothing to rasterize regardless of camera pose.
- **Trajectory / pose convention:** Early render code mixed `c2w` vs `w2c` and transpose conventions inconsistently with `getWorld2View2_torch` used in training.

**Fixes:**

- **Initialization:** `readNvidiaInfo` now seeds the scene by **back-projecting sparse depth** from training frames (with stride + cap), plus a small random fallback if depth is missing.
- **Rendering:** When using `cam_spline_controls.npz`, the renderer applies poses via **`set_camera_pose_from_spline`** (same path as training) instead of hand-built matrices.

### 3. Training crash: singular matrix (`linalg.inv`)

**Symptom:** `torch.linalg.inv` failed on `world_view_transform` or on export of `train_cam_c2w_spline.npy`.

**Cause:** Ill-conditioned or invalid rotation from the spline; raw matrix inverse is fragile.

**Fixes in `train_static_core.py` (`set_camera_pose_from_spline` and spline export block):**

- Sanitize `R, T` (finite checks; **SVD projection** of `R` onto `SO(3)`).
- **Camera center** computed as `-(R @ T)` instead of inverting `world_view_transform`.
- **Exported `c2w`** built analytically from rigid transform (no `inv(w2c)` on bad matrices).

### 4. “Snapshot” video (almost no motion)

**Symptom:** Whole clip looks like one frame; trajectory barely moves.

**Not only iteration count:** Short runs (e.g. 2500 iters) hurt quality, but **main spline stage starts at iteration 2000** (`STEP1.3`), so only ~500 iterations actually optimize the spline. Combined with strong smoothness / FOV losses, the trajectory can stay near-identity until trained longer.

**Practical guidance:** Prefer **10k–30k** iterations for meaningful stabilization; verify motion with `train_cam_c2w_spline.npy` translation stats if needed.

---

## Files touched (summary)

| Area | File | Change |
|------|------|--------|
| Dataset init | `scene/dataset_readers.py` | `readNvidiaInfo`: depth-based initial point cloud instead of zero dummy |
| Training / export | `train_static_core.py` | Stable `set_camera_pose_from_spline`; safe spline trajectory export |
| Render | `render_stabilized_video.py` | Source path resolution; spline controls → `set_camera_pose_from_spline`; debug visibility |

---

## Artifacts written after a successful train

Under `point_cloud/static_core_final/`:

- `point_cloud_static.ply` / `point_cloud_static.pt` — static Gaussians + rgbdecoder weights
- `posenet.pth` — focal / pose MLP weights
- `train_cam_c2w_gt.npy` — GT camera `c2w` (dataset convention)
- `train_cam_c2w_spline.npy` — per-frame **learned** spline `c2w` (for debugging / external use)
- `cam_spline_controls.npz` — `ctrl_trans`, `ctrl_quats`, `n_frames` for reproducing the spline

---

## Commands

### Train (example: 5000 iterations, custom output dir)

```bash
python train_entrypoint.py \
  --skip-gpu-preflight \
  --source_path /workspace/vidstabil/data2/regular_scene \
  --model_path /workspace/vidstabil/output2/your_run \
  --expname your_run \
  --iterations 5000
```

Or call `train_static_core.py` directly with the same flags (no preflight wrapper).

### Render stabilized video (preferred: spline controls)

```bash
python render_stabilized_video.py \
  --run-dir /workspace/vidstabil/output2/your_run \
  --source-path /workspace/vidstabil/data2/regular_scene \
  --trajectory /workspace/vidstabil/output2/your_run/point_cloud/static_core_final/cam_spline_controls.npz \
  --output /workspace/vidstabil/output2/your_run/stabilized.mp4 \
  --fps 24
```

If `cfg_args` has a wrong `source_path`, always pass `--source-path` explicitly to the folder that contains `images_2/`.

### Sanity check: trajectory motion (optional)

```bash
conda run -n splinegs python - <<'PY'
import numpy as np
p = "/workspace/vidstabil/output2/your_run/point_cloud/static_core_final/train_cam_c2w_spline.npy"
a = np.load(p)
t = a[:, :3, 3]
print("translation std:", t.std(axis=0))
print("min/max:", t.min(axis=0), t.max(axis=0))
PY
```

Very small std/range → video will look like a snapshot even if rendering is correct.

---

## Iteration budget (rough)

| Iterations | Notes |
|------------|--------|
| 2500 | Spline only ~500 iters after unfreeze at 2000 — often blurry / static |
| 5000–10000 | Usable preview |
| 15000–30000 | Better stabilization and scene quality |

Default in `arguments/__init__.py` is **30000** iterations unless overridden.

---

## Render log: `[debug] frame0 visible gaussians`

- **0** with a non-degenerate PLY usually means wrong pose or frustum; after the dataset init fix, checkpoints should have real XYZ spread.
- **Large** count but bad image → look at training quality / decoder, not just pose export.

---

## Legacy note

Older runs trained before these fixes may still have **all-zero** Gaussians in `point_cloud_static.ply`; retrain after pulling the `dataset_readers.py` change to get a valid scene.
