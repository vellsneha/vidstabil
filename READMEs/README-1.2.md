# Step 1.2 — SE(3) Camera Spline

Replaces the per-frame `pose_network` with a cubic Hermite spline in SE(3),
parameterised by K = N // 5 control points.

**Builds on:** Step 1.1 (static-core training path in `train_static_core.py`)

---

## What Was Done

### 1) Created `scene/camera_spline.py`

New file containing the `CameraSpline` class (`nn.Module`).

**Control points**

| Parameter | Shape | Meaning |
|---|---|---|
| `ctrl_trans` | `[K, 3]` | Translation at each knot |
| `ctrl_quats` | `[K, 4]` | Unit quaternion (wxyz) at each knot |

For N = 150 frames: K = 30 control points → **210 parameters** vs the original
pose_network's thousands of MLP weights (equivalent per-frame DOF = 150 × 6 = 900).

**Key methods**

- `get_pose(t)` → `(R [3,3], T [3])`
  - Maps frame index `t ∈ [0, N-1]` to a fractional control-point coordinate.
  - **Translation**: cubic Hermite basis with finite-difference tangents (clamped at
    boundaries).
  - **Rotation**: Squad — `slerp(slerp(q_i, q_{i+1}, u), slerp(q_{i-1}, q_{i+2}, u), 2u(1-u))` — output quaternion is renormalised, then converted to a 3×3
    rotation matrix via the standard quaternion-to-matrix formula.
  - Fully differentiable: no `.detach()` or numpy calls in the forward path.
    Gradients flow through `ctrl_trans` and `ctrl_quats`.

- `get_all_poses(N)` → `list[(R, T)]`
  - Returns poses for all frames 0 … N-1.

- `initialize_from_poses(Rs [N,3,3], Ts [N,3])`
  - Uniformly subsamples N per-frame pose estimates to K control points.
  - Converts sampled rotation matrices to unit quaternions (Shepperd's method).
  - Runs under `torch.no_grad()`; sets `ctrl_trans.data` and `ctrl_quats.data`.

### 2) Modified `train_static_core.py`

**Warm-start (before training loop)**

Runs `pose_network` once over all training cameras under `torch.no_grad()` to
collect initial R, T estimates, then calls `cam_spline.initialize_from_poses(Rs, Ts)`
so the spline starts from a sensible trajectory rather than identity.

```
pose_holder._posenet  (warm-start only — no longer called per iteration)
        ↓
cam_spline.initialize_from_poses(Rs, Ts)
```

**Optimizer**

`cam_spline.parameters()` is added to `pose_optimizer` as a new param group with
the same `lr = opt.pose_lr_init`. Both `ctrl_trans` and `ctrl_quats` receive
gradients from `photo_loss.backward()` every iteration.

**Training loop (per iteration)**

Old path:
```
pose_network(time, depth) → pred_R, pred_T
    → local_viewdirs (numpy)
    → update_cam(R, T, viewdirs, ...)
    → render_static(viewpoint_cam, ...)
```

New path (STEP1.2):
```
cam_spline.get_pose(cam_id) → R [3,3], T [3]   ← differentiable
    → world_view_transform  = getWorld2View2_torch(R, T).T
    → projection_matrix     = getProjectionMatrix(FoVx, FoVy, ...)
    → full_proj_transform   = world_view_transform @ projection_matrix
    → camera_center         = inv(world_view_transform)[3, :3]
    → render_static(viewpoint_cam, ...)
```

`update_cam` is no longer called in the training loop. Camera transforms are
set directly so gradients from the loss flow back into the spline control points.

### 3) Modified `train_entrypoint.py`

No structural change. Added one comment:

```python
# STEP1.2: cam_spline now replaces pose_network in static core
```

### 4) Fixed `verify-1.2.py` import

`from scene.camera_spline import CameraSpline` triggered `scene/__init__.py` →
`GaussianModel` → `simple_knn` (a CUDA extension), crashing on machines without
a compiled GPU environment. The import in the verifier was changed to use
`importlib.util.spec_from_file_location` to load `camera_spline.py` directly,
bypassing the package init. `camera_spline.py` itself has no heavy dependencies —
only `torch` and `torch.nn`.

---

## Verification

```bash
python verify-1.2.py
```

**21/21 checks pass**, including:

- K = N // 5 = 30 for N = 150
- `ctrl_trans` and `ctrl_quats` are `nn.Parameter`
- Parameter count (210) < per-frame equivalent (900)
- `get_pose` returns `R [3,3]`, `T [3]`
- Gradients flow through `get_pose` (both params receive grad)
- R is valid SO(3): `|R^T R − I|_max < 1e-4`, `det = +1.0`
- Unit quaternions after `initialize_from_poses`
- Warm-start accuracy at t = 0 (T err = 0.0, R err = 0.0)
- `get_all_poses(150)` returns 150 pairs with correct shapes
- `optimizer.step()` updates control points
- Trajectory is smooth (max jump < 10× mean jump)
- `train_static_core.py` imports and uses `CameraSpline`
- `update_cam` not called inside training loop
- `# STEP1.2` comments present in both modified files
- No `.numpy()` or `.detach()` calls in `get_pose` forward path

---

## Parameter Budget

| | N = 150 |
|---|---|
| Old: pose_network MLP | ~tens of thousands |
| Old: per-frame DOF equivalent | 150 × 6 = **900** |
| New: `CameraSpline` | 30×3 + 30×4 = **210** |
| Reduction | **77 %** fewer pose parameters |

---

## Files Changed

| File | Change |
|---|---|
| `scene/camera_spline.py` | **New** — `CameraSpline` class |
| `train_static_core.py` | Import, warm-start, optimizer group, loop refactor |
| `train_entrypoint.py` | One comment added |
| `verify-1.2.py` | Import method fixed (`importlib.util`) |

---

## How To Run

```bash
# Verify Step 1.2 (CPU-only, no GPU required for verification)
python verify-1.2.py

# Train with spline poses (requires NVIDIA GPU)
python train_entrypoint.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_spline_run
```

## Notes

- `pose_network` is still instantiated and used for warm-start; it is **not** called
  during the training loop.
- The focal length from `pose_holder._posenet.focal_bias` is still read each
  iteration (detached) to set the camera FoV — focal optimisation is unchanged.
- The checkpoint path (`output/<expname>/point_cloud/static_core_final/`) is
  unchanged from Step 1.1.
- Legacy dynamic path remains available via `--legacy-dynamic` and is unaffected.
