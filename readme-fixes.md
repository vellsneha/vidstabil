# Training iteration speed fixes (static core)

This document summarizes changes made to remove a large **slowdown in the main training stage** (iteration ≥ 2000, after the camera spline is unfrozen) compared to **warm-up** (spline frozen). Observed gaps on the order of **tens of×** were not explained by the intended stability losses alone (those were expected to be only a few×); the main issues were **per-frame Python loops**, **repeated spline work**, and **GPU sync** in scalar code paths.

---

## What was going wrong

1. **`L_smooth` and `L_fov` in a Python loop over frames**  
   Each iteration evaluated smooth acceleration and FoV loss by looping over frames (or the full chunk) and calling scalar APIs such as `get_translation_second_derivative(t)` and `get_pose(t)` many times. That adds huge Python overhead and repeats expensive work (including full **rotation** spline / quaternion path for `get_pose` when only **translation** was needed for FoV).

2. **`get_translation_second_derivative` and GPU sync**  
   The scalar second-derivative path used tensor indexing followed by **`.item()`** to get a segment index. Doing that **once per frame per iteration** forces many **CPU/GPU synchronizations**, which dominates wall time.

3. **`torch.where` and out-of-bounds indexing (bug)**  
   Vectorized Hermite tangents used `torch.where` with expressions like `ctrl_trans[i + 2]`. **Both branches of `torch.where` are evaluated**, so for the last valid segment (`i == K - 2`) index `i + 2 == K` was still read → **CUDA device-side assert** / illegal access. The fix is to use **clamped index tensors** (`im1`, `ip2`) so every gather stays in `[0, K - 1]`.

4. **Reference trajectory `T_ref_fov`**  
   This was already correct: **`T_ref_fov` is built once** before the training loop from `frozen_low_frequency_translation_reference(_Ts_init)` and is **not** recomputed each iteration. Only per-iteration use of current spline translations vs that reference mattered for cost.

---

## What we changed

### `scene/camera_spline.py`

- **Batched Hermite translation** for arbitrary frame index lists:
  - `_translation_second_derivatives_for_t_frame(t_frame)` — `t_frame` shape `[M]`; returns `[M, 3]`.
  - `_translations_for_t_frame(t_frame)` — same; Hermite **translation only** (no Squad / quaternion), suitable for FoV loss.
- **Public helpers:** `get_translation_second_derivatives_at(frame_indices)`, `get_translations_at(frame_indices)`.
- **Optional full-sequence helpers:** `get_all_translation_second_derivatives()`, `get_all_translations()` (internally call the batched path over `t = 0..N-1`).
- **Indexing safety:** `im1 = torch.clamp(i - 1, min=0)`, `ip2 = torch.clamp(i + 2, max=K - 1)` wherever `torch.where` combines center and boundary tangent formulas.

### `train_static_core.py`

- **`STABILITY_LOSS_FRAME_SAMPLE` (default `256`)** with `_stability_loss_frame_indices(...)`: for **`L_smooth`** and **`L_fov`**, frame indices are either **all frames** (if `N ≤ 256`) or a **uniform random subset of size 256** each iteration. The loss is a **Monte Carlo estimate** of the same **mean over frames** (same expectation as a full `(1/N) Σ_t`), but caps spline forward/backward cost per step for long sequences.
- **Chunked training path** uses the same idea per chunk (subsample within `[c_start, c_end)` capped by `STABILITY_LOSS_FRAME_SAMPLE`).
- **Comment** near `T_ref_fov` construction: reference is computed **once** before the loop.

### `profile-2.4.py`

- **Standalone `argparse`** (`import argparse as _argparse`, dedicated `_parser`) so profiler flags (`-s`, `--expname`, `--profile_iters`, etc.) do not clash with the main training script’s argument definitions.

---

## What we did *not* change

- Loss **weights** (`w_smooth`, `w_jitter`, `w_fov`, `w_dilated`).
- **Gates:** `L_jitter` every **10** iterations; `L_dilated` every **5** iterations; `L_smooth` / `L_fov` every main-stage iteration (with subsampling as above).
- **Optimizer structure** (including spline step every 2nd iteration, etc.).

---

## Practical notes

- **Render count:** On a typical main-stage iteration you still pay **one** primary `render_static` for photometric loss. On iterations where **`iteration % 10 == 0`**, jitter adds **two** more renders. Where **`iteration % 5 == 0`**, dilated adds **two** more. Iterations that are multiples of **10** are also multiples of **5**, so those steps can run **both** extra losses → **more** than three total renders on those iterations.
- **Tuning:** To adjust cost vs variance of `L_smooth` / `L_fov`, change **`STABILITY_LOSS_FRAME_SAMPLE`** in `train_static_core.py` (higher = closer to exact full mean, slower).

---

## Files touched (conceptual)

| Area | File(s) |
|------|---------|
| Batched spline + safe `torch.where` | `scene/camera_spline.py` |
| Subsampled stability losses + one-shot `T_ref_fov` note | `train_static_core.py` |
| Profiler CLI | `profile-2.4.py` |
