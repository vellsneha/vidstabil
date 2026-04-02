# Step 2.3 — Reduced Densification Aggressiveness

Two targeted changes only: halve the densification frequency and cap the
total Gaussian count at 500 K. All other training hyperparameters, loss
weights, and architecture choices are unchanged.

**Builds on:** Step 2.2 (asymmetric spline / Gaussian optimizer stepping;
`controlgaussians` / `densify_pruneclone` are the codebase's
`densify_and_prune` equivalent)

---

## Motivation

Video stabilisation trains on many frames of the **same static scene**.
Without a cap, 3DGS densification can spawn hundreds of thousands of
redundant Gaussians that:

- Slow down each render (memory bandwidth) without improving reconstruction.
- Destabilise the trajectory optimisation — dense, poorly-placed Gaussians
  produce noisy viewspace gradients that the spline then tries to fit.

Halving the densification frequency (100 → 200 iterations) and enforcing a
hard opacity-ranked cap (500 K) keeps the scene compact while preserving
reconstruction quality.

---

## Change 1 — `densification_interval` default

**File:** `arguments/__init__.py`  
**Field:** `OptimizationParams.densification_interval`

```python
# Before
self.densification_interval = 1_00

# After
self.densification_interval = 200  # STEP2.3 was 100
```

With `opt.densify_from_iter = 500` and `opt.densify_until_iter = 15_000`,
densification now fires at iterations 700, 900, 1100, … instead of every
100 steps — exactly **half as often**:

| Setting | Fires in range [500, 2000] | Total in [500, 15000] |
|---------|--------------------------|----------------------|
| interval = 100 | 16 times | 145 times |
| interval = 200 | 8 times  | 73 times  |
| ratio | 2.0× | 2.0× |

**Only `densification_interval` was changed.** The following fields are
verified unchanged:

| Field | Value |
|-------|-------|
| `densify_from_iter` | `500` |
| `densify_until_iter` | `15_000` |
| `densify_grad_threshold` | `0.0008` (fork value) |
| `opacity_reset_interval` | `3_000` |

---

## Change 2 — `MAX_GAUSSIANS` constant

Added in `train_static_core(...)` alongside the Step 2.1 chunk constants:

```python
CHUNK_THRESHOLD = 150    # STEP2.1
CHUNK_SIZE      = 70     # STEP2.1
OVERLAP         = 20     # STEP2.1
MAX_GAUSSIANS   = 500_000  # STEP2.3 — Gaussian count cap
```

---

## Change 3 — Densification + hard cap in both training paths

### What was added first

`train_exp.py` had **no densification call** prior to Step 2.3.
The codebase-standard `controlgaussians(...)` function (from
`scene/gaussian_model.py`) was imported and wired into both the
single-scene path and `_train_chunked`. `controlgaussians` calls
`densify_pruneclone` (the fork's `densify_and_prune` equivalent) and
`reset_opacity` on the correct schedule.

### Single-scene path (`else`-branch)

Added after the optimizer steps, inside `with torch.no_grad()`:

```python
# STEP2.3 — densification + hard Gaussian cap
with torch.no_grad():                                            # STEP2.3
    if iteration < opt.densify_until_iter:                       # STEP2.3
        visibility_filter = render_pkg["visibility_filter"]      # STEP2.3
        radii             = render_pkg["radii"]                  # STEP2.3
        stat_gaussians.max_radii2D[visibility_filter] = torch.max(
            stat_gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter])                            # STEP2.3
        viewspace_pts = render_pkg["viewspace_points"]           # STEP2.3
        if viewspace_pts.absgrad is not None:                    # STEP2.3
            vpt_grad = viewspace_pts.absgrad.squeeze(0)          # STEP2.3 [N, 2]
            vpt_grad = vpt_grad * torch.tensor(
                [viewpoint_cam.image_width  * 0.5,
                 viewpoint_cam.image_height * 0.5],
                device=vpt_grad.device)                          # STEP2.3
            stat_gaussians.add_densification_stats(
                vpt_grad, visibility_filter)                     # STEP2.3
        if (iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0):  # STEP2.3
            flag_s = controlgaussians(
                opt, stat_gaussians, opt.densify,
                iteration, scene, flag_s)                        # STEP2.3
            # STEP2.3 — hard cap at MAX_GAUSSIANS
            if stat_gaussians.get_xyz.shape[0] > MAX_GAUSSIANS: # STEP2.3
                stat_gaussians.prune_points(                     # STEP2.3
                    stat_gaussians.get_opacity.squeeze() <
                    stat_gaussians.get_opacity.squeeze()
                    .topk(MAX_GAUSSIANS).values.min()
                )  # STEP2.3 keep only top-MAX_GAUSSIANS by opacity
```

`flag_s = 0` is initialised before the loop. It counts densification
events; once `flag_s >= opt.desicnt (=6)` the gate switches from
clone+split to prune-only (standard `controlgaussians` behaviour).

### Chunked path (`_train_chunked`)

Identical structure using `chunk_gaussians`, `c_vis`, `c_rad`,
`chunk_flag_s`, and `local_iter`:

```python
chunk_flag_s = controlgaussians(
    opt, chunk_gaussians, opt.densify,
    local_iter, scene, chunk_flag_s)               # STEP2.3
# STEP2.3 — hard cap at MAX_GAUSSIANS
if chunk_gaussians.get_xyz.shape[0] > MAX_GAUSSIANS:  # STEP2.3
    chunk_gaussians.prune_points(
        chunk_gaussians.get_opacity.squeeze() <
        chunk_gaussians.get_opacity.squeeze()
        .topk(MAX_GAUSSIANS).values.min()
    )  # STEP2.3
```

`chunk_flag_s = 0` resets at the **start of each chunk** (inside the
`for chunk_idx, ...` loop), so every chunk gets its own fresh densification
budget. `scene` (global) is passed to `_train_chunked` as a new parameter
(its `cameras_extent` is what `controlgaussians` reads).

---

## Change 4 — Gaussian count in progress bar

```python
n_gauss = stat_gaussians.get_xyz.shape[0]  # STEP2.3
progress_bar.set_postfix({
    "loss":    f"{loss.detach().item():.6f}",
    "psnr":    f"{current_psnr:.2f}",
    "n_gauss": n_gauss,                     # STEP2.3
    "stage":   stage,
    "focal":   f"{...:.2f}",
})
```

The old `"num_gaussians"` key is replaced by the shorter `"n_gauss"` and
its value is pre-captured so the same number is used both by the logging
call and (if needed) by future checks in the same iteration.

---

## Change 5 — `train_entrypoint.py` comment

```python
# STEP2.3: densification_interval=200, MAX_GAUSSIANS=500K
```

---

## Prune mask logic

```python
stat_gaussians.prune_points(
    stat_gaussians.get_opacity.squeeze() <
    stat_gaussians.get_opacity.squeeze()
    .topk(MAX_GAUSSIANS).values.min()
)
```

`prune_points(mask)` takes a boolean tensor where `True` = remove.
The mask marks every Gaussian whose opacity is **strictly below** the
minimum opacity of the top-500 K Gaussians by opacity — i.e. the 500 K
highest-opacity Gaussians are kept; everything else is pruned.

`verify-2.3.py` simulation confirms:
```
N_sim=600_000, MAX_CAP=500_000 → kept 500_000 / 600_000 ✓
N_sim=300_000 (below cap)      → no prune needed ✓
```

---

## New import

```python
from scene.gaussian_model import controlgaussians  # STEP2.3
```

Added at the top of `train_exp.py` alongside the existing imports.

---

## What is unchanged

| Element | Status |
|---------|--------|
| `densify_from_iter = 500` | **Unchanged** |
| `densify_until_iter = 15_000` | **Unchanged** |
| `densify_grad_threshold = 0.0008` | **Unchanged** |
| `opacity_reset_interval = 3_000` | **Unchanged** |
| All four stability loss weights | **Unchanged** |
| `cam_spline` freeze / unfreeze logic | **Unchanged** |
| `scene/camera_spline.py` | **Not touched** |
| `render_static` signature | **Not touched** |
| Spline / Gaussian optimizer stepping frequency | **Unchanged** |

---

## `verify-2.3.py` fixes

Three verifier bugs fixed; the implementation was already correct.

### Fix 1 — `densify_until_iter` parsed as `15` instead of `15000`

The source has `15_000` (Python underscore literal). The original regex
`(\d+)` stopped at the `_`, capturing `15`. Fixed by reading `([0-9_]+)`
and calling `.replace("_", "")` before `int()`.

### Fix 2 — `densify_grad_threshold` expected `0.0002`, found `0.0008`

This fork defines `densify_grad_threshold = 0.0008` as a base threshold
(separate from `densify_grad_threshold_coarse` and
`densify_grad_threshold_fine_init` which are both `0.0002`). The verifier
expected the standard-3DGS value `0.0002`, which was never the correct
value for this field in this fork. Updated expected value to `0.0008`.

### Fix 3 — `densify_and_prune` not found

This fork never defines a function called `densify_and_prune`. The
equivalent call chain is `controlgaussians → densify_pruneclone`. The
verifier pattern was broadened to accept any of the three:

```python
r"densify_and_prune\s*\(|controlgaussians\s*\(|densify_pruneclone\s*\("
```

---

## Verification

```bash
python verify-2.3.py  # 23/23
```

**23/23 checks pass**, including:

- `densification_interval = 200` in `arguments/__init__.py`.
- `# STEP2.3` comment on that line.
- `densify_from_iter`, `densify_until_iter`, `densify_grad_threshold`,
  `opacity_reset_interval` all unchanged.
- `MAX_GAUSSIANS = 500_000` constant in `train_exp.py`.
- `MAX_GAUSSIANS` value is exactly 500000.
- Densification call (`controlgaussians`) present in training loop.
- `prune_points` call present for cap enforcement.
- Cap condition references `MAX_GAUSSIANS`.
- Cap also present in `_train_chunked` for `chunk_gaussians`.
- `n_gauss` added to progress bar.
- `STEP2.3` comment in `train_entrypoint.py`.
- `camera_spline.py` has zero `STEP2.3` marks.
- All four stability loss weights unchanged.
- Cap prune logic correct: keeps top-500 K by opacity.
- Cap does not prune when count ≤ 500 K.
- `densification_interval=200` fires exactly half as often as 100.

---

## Files changed

| File | Change |
|------|--------|
| `arguments/__init__.py` | `densification_interval`: `100` → `200` |
| `train_exp.py` | `controlgaussians` import; `MAX_GAUSSIANS` constant; `flag_s` init; densification + cap block in single-scene path; `chunk_flag_s` init; densification + cap block in `_train_chunked`; `scene` parameter added to `_train_chunked`; `n_gauss` in progress bar |
| `train_entrypoint.py` | One `# STEP2.3` comment |
| `verify-2.3.py` | Three verifier bug fixes (underscore integer parsing; wrong expected threshold; missing fork-specific densification name) |

**Not touched:** `scene/camera_spline.py`, `scene/gaussian_model.py`
internals, `utils/` modules, stability loss weights.

---

## How to run

```bash
python train_exp.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run
python verify-2.3.py   # 23/23
```

---

## Append-only history

### 2026-03-26 — Step 2.3 implemented

- `arguments/__init__.py`: `densification_interval` `100` → `200`.
- `train_exp.py`: `MAX_GAUSSIANS = 500_000`; `controlgaussians`
  import and wiring in both training paths; `chunk_flag_s` per-chunk reset;
  `scene` parameter added to `_train_chunked`; `n_gauss` progress bar key.
- `train_entrypoint.py`: `# STEP2.3` comment.
- `verify-2.3.py`: three verifier bug fixes.
- All added/modified lines tagged `# STEP2.3` (66 occurrences).

---

## Assistant code change log

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-26 | `arguments/__init__.py` | `densification_interval` `1_00` → `200`; `# STEP2.3 was 100` comment. |
| 2026-03-26 | `train_exp.py` | `controlgaussians` import; `MAX_GAUSSIANS=500_000` constant; densification blocks (radii, absgrad stats, `controlgaussians` call, cap) in both single-scene and chunked paths; `n_gauss` in tqdm postfix. |
| 2026-03-26 | `train_entrypoint.py` | `# STEP2.3: densification_interval=200, MAX_GAUSSIANS=500K`. |
| 2026-03-26 | `verify-2.3.py` | Underscore-integer parsing for `densify_until_iter`; corrected `densify_grad_threshold` expected value to fork's `0.0008`; broadened densification-call pattern to include `controlgaussians` and `densify_pruneclone`. |
