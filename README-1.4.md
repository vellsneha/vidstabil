# Step 1.4 — Stability losses (four terms)

Adds **stability regularisers** on top of the photometric loss in the **main stage**
(`iteration >= 2000`), after the spline is unfrozen (Step 1.3). The Step 1.3
**zero tensor stub** is replaced by real terms.

**Builds on:** Step 1.3 (warm-up / main split, `loss = photo_loss` then additive
stability block inside `if iteration >= 2000`)

**Related (read-only context):** `README-1.1.md` … `README-1.3.md` — do not edit those for Step 1.4; keep all Step 1.4 documentation here.

---

## Scope

| Term | Name | Status |
|------|------|--------|
| 1 | \(\mathcal{L}_{\text{smooth}}\) — trajectory smoothness (translation acceleration) | **Implemented** |
| 2 | \(\mathcal{L}_{\text{jitter}}\) — rendered jitter (Laplacian of frame difference / flow) | **Implemented** |
| 3 | \(\mathcal{L}_{\text{fov}}\) — FoV preservation (drift from low-frequency path) | **Implemented** |
| 4 | \(\mathcal{L}_{\text{dilated}}\) | Pending |

---

## Term 1 — \(\mathcal{L}_{\text{smooth}}\) (implemented)

### Objective

Penalise **large acceleration** of the **translation** component \(T(t)\) of the
camera trajectory. The translation path is the same **cubic Hermite spline** as in
`CameraSpline.get_pose` (Step 1.2). Because the spline is analytic, \(\ddot{T}(t)\)
is computed in **closed form** (no rendering, no finite differences in the loss).

### Definition

\[
\mathcal{L}_{\text{smooth}} = \frac{1}{N} \sum_{t=0}^{N-1} \|\ddot{T}(t)\|^2
\]

Here \(N = \texttt{total\_frames}\) (number of training cameras / frame indices).
Sampling is over **integer frame indices** \(t = 0, \ldots, N-1\), matching the
discrete video frames.

### Implementation details

1. **`scene/camera_spline.py`**
   - Method: **`get_translation_second_derivative(t_frame)`** (`# STEP1.4`).
   - Same Hermite segment and tangents as `get_pose`; \(\frac{d^2 T}{dt^2} = \alpha^2 \frac{d^2 T}{du^2}\) with \(\alpha = (K-1)/(N-1)\).
   - Differentiable w.r.t. `ctrl_trans`.

2. **`train_static_core.py`**
   - When **`iteration >= 2000`**: accumulate `loss_smooth` over all frames, **`loss += w_smooth * loss_smooth`**, **`w_smooth = 1e-2`**.
   - Every main-stage iteration; **no** extra renders.

### Hyperparameter (Term 1)

| Symbol | Value | Location |
|--------|-------|----------|
| \(w_{\text{smooth}}\) | `1e-2` | `train_static_core.py`, `w_smooth` |

---

## Term 2 — \(\mathcal{L}_{\text{jitter}}\) (implemented)

### Objective

Penalise **high-frequency energy** in the **difference** between consecutive
**stabilised** renders \(I_t^{\text{stab}}, I_{t+1}^{\text{stab}}\) (from the
current spline pose and Gaussian scene). The **pixel difference** is a proxy for
**optical flow** before iteration 7000; after **7000**, **RAFT** estimates dense
flow (see below).

### Definition

\[
\mathcal{L}_{\text{jitter}} = \left\| \nabla^2 \left( I_{t+1}^{\text{stab}} - I_t^{\text{stab}} \right) \right\|_F
\]

(\(\|\cdot\|_F\) is implemented as the Frobenius norm of the **per-channel**
Laplacian stack — same as \(\sqrt{\sum_{c,h,w} (\nabla^2 \mathrm{diff}_c)^2}\).)

### Phases (global iteration counter)

| Iteration range | Mechanism |
|-----------------|-----------|
| `2000`–`6999` (main stage, `iteration % 10 == 0`) | **Pixel difference** \(I_{t+1}-I_t\); **Laplacian** via **3×3** kernel \([0,1,0;1,-4,1;0,1,0]\) applied per RGB channel; **differentiable** through both renders. |
| `≥ 7000` (same `iteration % 10 == 0`) | **RAFT** (`torchvision.models.optical_flow.raft_large`, pretrained weights). Forward runs under **`torch.no_grad()`** so **RAFT weights** and **flow** do **not** receive gradients; **no gradient** from this term to Gaussians/spline (only a fixed regularisation signal). **Laplacian** is applied to the **2-channel flow** field, then Frobenius norm. If RAFT / torchvision is unavailable, **falls back** to the pixel-difference loss (same as pre-7000). |

**Warm-up** (`iteration < 2000`): jitter term **not** evaluated (stability block is
main-stage only).

### Frequency

**Every 10 iterations** when `iteration >= 2000` and `total_frames >= 2` (most
expensive stability term).

### Pair sampling

One random consecutive pair **`t_pair`, `t_pair+1`** with
`uniform(0, N-2)` per jitter evaluation.

### Implementation details

1. **`utils/jitter_loss.py`** (`# STEP1.4`)
   - **`loss_jitter_pixel_diff(I0, I1)`** — Laplacian of `I1 - I0`, Frobenius norm.
   - **`loss_jitter_raft_laplacian(I0, I1, device)`** — optional RAFT + Laplacian on flow; **detached** scalar.

2. **`train_static_core.py`**
   - **`set_camera_pose_from_spline(...)`** — shared helper for spline pose + projection (used by the main training render and by jitter’s two renders).
   - Two **`render_static`** calls for **\(I_t\)** and **\(I_{t+1}\)** with **`w_jitter = 1e-3`**.
   - Tagged **`# STEP1.4`** throughout.

### Hyperparameters (Term 2)

| Symbol | Value | Location |
|--------|-------|----------|
| \(w_{\text{jitter}}\) | `1e-3` | `train_static_core.py`, `w_jitter` |
| Phase switch | `7000` | `if iteration < 7000:` pixel vs RAFT |

---

## Term 3 — \(\mathcal{L}_{\text{fov}}\) (implemented)

### Objective

Penalise the **stabilised translation trajectory** drifting too far from a **fixed
low-frequency reference** \(\bar{T}(t)\), which would require **aggressive cropping**
or introduce **black borders** in a fixed-viewport stabilisation setup.

### Definition

\[
\mathcal{L}_{\text{fov}} = \frac{1}{N} \sum_{t=0}^{N-1} \| T(t) \ominus \bar{T}(t) \|^2
\]

Here \(\ominus\) is the **translation difference** in the same convention as
`CameraSpline.get_pose`: \(T(t) - \bar{T}(t)\) (both world-to-camera translations
for frame \(t\)). \(N = \texttt{total\_frames}\).

### Reference trajectory \(\bar{T}\)

- Built **once** before the training loop from the **initial rough** per-frame
  translations **`_Ts_init`** (same stack as spline warm-start / pose network).
- **Heavily smoothed** along time: **1D Gaussian convolution** over the sequence
  (replicate padding at ends), implemented in **`utils/fov_loss.py`**.
- **Frozen** for all iterations: **`detach()`**, **no gradients**, **not** updated
  during optimisation.

### Implementation details

1. **`utils/fov_loss.py`** (`# STEP1.4` module docstring)
   - **`frozen_low_frequency_translation_reference(T_init)`** → `[N, 3]` detached.

2. **`train_static_core.py`**
   - After spline **`initialize_from_poses`**: compute **`T_ref_fov`** once inside
     **`torch.no_grad()`**.
   - When **`iteration >= 2000`**: every main-stage iteration, accumulate
     **`loss_fov`** over integer frame indices; **`loss += w_fov * loss_fov`**.
   - **No** extra renders (same cost pattern as **`L_smooth`**).

### Hyperparameter (Term 3)

| Symbol | Value | Location |
|--------|-------|----------|
| \(w_{\text{fov}}\) | `1e-3` | `train_static_core.py`, `w_fov` |

---

## Verification

### `verify-1.4.py`

- Confirms **`# STEP1.4`** markers, main-stage gate, and that the old
  **`stability_loss = torch.tensor(0.0, ...)`** stub is gone.
- **Term 1 (smooth):** **`loss_smooth`**, **`get_translation_second_derivative`** in `train_static_core.py` and `scene/camera_spline.py`.
- **Term 2 (jitter):** **`loss_jitter`**, **`iteration % 10 == 0`**, and **`loss_jitter_pixel_diff` / `loss_jitter_raft_laplacian`**.
- **Term 3 (fov):** **`loss_fov`**, **`frozen_low_frequency_translation_reference`**, **`T_ref_fov`** / **`w_fov`**.
- **Term 4:** still **`required: False`** until implemented.

```bash
python verify-1.4.py
python verify-1.4.py --strict
```

### `verify-1.3.py` (compatibility)

Additive loss may include **`w_smooth * loss_smooth`** and **`w_jitter * loss_jitter`**;
see `verify-1.3.py` for stub / additive checks.

---

## Files changed (Steps 1.4 Term 1 & Term 2)

| File | Change |
|------|--------|
| `scene/camera_spline.py` | **`get_translation_second_derivative`** |
| `utils/jitter_loss.py` | **New** — Laplacian, pixel jitter, optional RAFT jitter |
| `utils/fov_loss.py` | **New** — frozen low-frequency translation reference (Term 3) |
| `train_static_core.py` | **`set_camera_pose_from_spline`**, **`loss_smooth`**, **`loss_jitter`**, **`loss_fov`**, **`w_smooth`**, **`w_jitter`**, **`w_fov`**, **`T_ref_fov`** |
| `verify-1.4.py` | Per-term checks; Terms 1–3 **required**; Term 4 optional |
| `verify-1.3.py` | Additive / stub checks aligned with Step 1.4 |

---

## How to run

```bash
python train_entrypoint.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run
# or
python train_static_core.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run

python verify-1.4.py --strict
```

**GPU:** NVIDIA + CUDA required for full training (see `README-1.1.md` / `preflight_gpu.py`).

**RAFT (optional):** requires `torchvision` with `torchvision.models.optical_flow` (e.g. recent torchvision + CUDA). If import fails, the post-7000 path falls back to the pixel-difference loss.

---

## Next (Term 4)

Extend the **`if iteration >= 2000:`** block, add **`# STEP1.4`** markers, set
**`required: True`** for **`dilated`** in **`verify-1.4.py`** (`TERM_SPECS`), and
update the scope table above.

---

## Work log

| Date | Note |
|------|------|
| — | Term 1 (\(\mathcal{L}_{\text{smooth}}\)) documented and implemented. |
| — | Term 2 (\(\mathcal{L}_{\text{jitter}}\)): pixel Laplacian + RAFT phase; `utils/jitter_loss.py`; refactor `set_camera_pose_from_spline`. |
| — | Term 3 (\(\mathcal{L}_{\text{fov}}\)): frozen smoothed reference + per-frame translation MSE; `utils/fov_loss.py`. |

---

## Documentation convention (append-only)

Rules for **this file** (`README-1.4.md`) going forward:

1. **Do not rewrite or delete** existing sections to “refresh” the doc. Add new content **below** what is already here (new headings, **new dated subsections**, or **additional** rows — avoid replacing whole tables when a small addendum will do).
2. **Each step** should leave a **visible trace**: add a dated entry under **Append-only history** describing what was added.
3. **Code edits** made by an assistant (or automation) are listed separately under **Assistant code change log** so they are easy to audit and are not confused with the mathematical spec.

---

## Append-only history

### 2026-03-24 — Documentation policy

- **Added:** sections *Documentation convention (append-only)*, *Append-only history*, and *Assistant code change log (ongoing)*.
- **No** earlier sections were removed or replaced; this entry is additive.

### 2026-03-24 — Term 2 implementation trace (reference)

Summary of what was **added in code** when Term 2 (`L_jitter`) landed (for step-by-step traceability; the full spec remains in *Term 2* above):

- New module `utils/jitter_loss.py`.
- `train_static_core.py`: jitter block; refactor of inline camera pose into `set_camera_pose_from_spline` (same logic as before, shared with jitter’s two renders).
- `verify-1.4.py`: Term 2 checks marked required with patterns for `loss_jitter`, `iteration % 10 == 0`, and jitter helpers.

### 2026-03-24 — Term 3 implementation trace (reference)

- New module `utils/fov_loss.py` (`frozen_low_frequency_translation_reference`).
- `train_static_core.py`: one-time **`T_ref_fov`**; **`loss_fov`** each main-stage iteration.
- `verify-1.4.py`: Term 3 **`required: True`** with patterns for **`loss_fov`** and **`frozen_low_frequency_translation_reference`**.
- `verify-1.3.py`: additive-loss pattern extended for **`w_fov * loss_fov`**.

---

## Assistant code change log (ongoing)

**Append-only table.** Each row documents assistant/tooling edits relative to the prior codebase state. **Do not remove** prior rows.

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-24 | `README-1.4.md` only | Appended documentation convention, append-only history, and assistant code change log (no rewrites of content above). |
| 2026-03-24 (Term 2) | `utils/jitter_loss.py` (new), `train_static_core.py`, `verify-1.4.py` | `L_jitter` (pixel Laplacian + optional RAFT); `set_camera_pose_from_spline` refactor; verifier updates for Term 2. |
| Earlier (Term 1) | `scene/camera_spline.py`, `train_static_core.py`, `verify-1.4.py`, `verify-1.3.py` | `L_smooth` / `get_translation_second_derivative`; stub removal; verifier alignment. |
| 2026-03-24 (Term 3) | `utils/fov_loss.py` (new), `train_static_core.py`, `verify-1.4.py`, `verify-1.3.py`, `README-1.4.md` | `L_fov` (MSE to frozen low-frequency translation ref); scope + Term 3 doc; verifiers. |

### 2026-03-24 — Weight tuning update (reference)

Updated Step 1.4 default weights in `train_static_core.py` to match the Term 4
spec starting point:

- `w_smooth = 0.1`
- `w_jitter = 0.5`
- `w_fov = 0.05`
- `w_dilated = 0.1`

These are now defined once near the top of `train_static_core(...)` and reused
in the stability block.

---

## Term 4 addendum (implemented)

This addendum records the implementation of **Term 4 — \(\mathcal{L}_{\text{dilated}}\)** without rewriting earlier sections.

### Spec alignment

- Uses a dilated frame pair \((t, t+k)\) with default **`k=5`**.
- Runs **every 5 iterations** in the main stage (`iteration >= 2000`).
- Uses rasterizer-provided visibility (`render_static(...)[\"visibility_filter\"]`) for
  \(\mathcal{V}(t)\cap\mathcal{V}(t+k)\).

### Implemented form in `train_static_core.py`

\[
\mathcal{L}_{\text{dilated}} =
\frac{1}{|\mathcal{V}(t)\cap\mathcal{V}(t+k)|}
\sum_{g \in \mathcal{V}(t)\cap\mathcal{V}(t+k)}
\left(
\|\mu_g^t - \mu_g^{t+k}\|^2 + \|\alpha_g^t - \alpha_g^{t+k}\|^2
\right)
\]

- `mu`: camera-space Gaussian center (computed by transforming `stat_gaussians.get_xyz`
  with each frame's view transform via `world_to_camera_points(...)`).
- `alpha`: static model opacity (`stat_gaussians.get_opacity`) for the same Gaussian IDs.
- Added to total loss as `loss = loss + w_dilated * loss_dilated` with **`w_dilated = 0.1`** (see *Weight tuning update* above for the full default set).

### Verification updates

- `verify-1.4.py`: Term 4 set to **required** and checks for
  `loss_dilated`, `iteration % 5 == 0`, and `visibility_filter`.
- `verify-1.3.py`: additive loss check now accepts `w_dilated * loss_dilated`.

### 2026-03-24 — Term 4 implementation trace (reference)

- `train_static_core.py`: added `world_to_camera_points`; implemented
  dilated co-visibility loss block every 5 iterations with `k=5`.
- `verify-1.4.py`: Term 4 marked required.
- `verify-1.3.py`: additive-loss regex extended for Term 4.

### Assistant code change log entry

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-24 (Term 4) | `train_static_core.py`, `verify-1.4.py`, `verify-1.3.py`, `README-1.4.md` | `L_dilated` implemented with co-visible Gaussian matching (`k=5`, every 5 iters) and verifier/doc updates. |
| 2026-03-24 (weights) | `train_static_core.py`, `README-1.4.md` | Default `w_smooth`/`w_jitter`/`w_fov`/`w_dilated` set to spec starting values; README weight-tuning subsection. |
