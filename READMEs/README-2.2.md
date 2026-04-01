# Step 2.2 — Asymmetric Update Frequency (Spline vs Scene)

The **only** new behaviour in this step: `pose_optimizer` (which holds the
`cam_spline` parameters) steps **every second iteration**. The Gaussian
optimizer continues to step **every iteration**. Loss weights, stability
term frequencies, and the warm-up / unfreeze logic are all unchanged.

**Builds on:** Step 2.1 (chunked path + `_train_chunked`; `pose_optimizer`
shared globally; spline freeze / unfreeze at iteration 2000 / `CHUNK_WARMUP`)

---

## Motivation

The camera spline represents a **smooth, low-dimensional trajectory**
(3 × K control points for translation, K quaternions for rotation). The
Gaussian scene, by contrast, has millions of parameters and benefits from
dense gradient signal every iteration. Halving the spline update rate:

- Lets **gradients from two consecutive iterations accumulate** before the
  spline takes a step, giving effectively a 2× larger gradient magnitude per
  step.
- Prevents the spline from chasing high-frequency photometric noise that the
  Gaussians should absorb instead.
- Costs nothing extra at runtime (one `if` check per iteration).

---

## Changes

### CHANGE 1 — Single-scene path (`else`-branch, `train_static_core`)

**Before (Step 2.1):**

```python
loss.backward()
stat_gaussians.optimizer.step()
stat_gaussians.optimizer.zero_grad(set_to_none=True)
pose_optimizer.step()
pose_optimizer.zero_grad(set_to_none=True)
```

**After (Step 2.2):**

```python
loss.backward()
stat_gaussians.optimizer.step()                      # unchanged — every iter
stat_gaussians.optimizer.zero_grad(set_to_none=True)  # unchanged
if iteration % 2 == 0:                               # STEP2.2 — spline steps every 2nd iter
    pose_optimizer.step()                             # STEP2.2
    pose_optimizer.zero_grad(set_to_none=True)       # STEP2.2
```

On **odd** iterations, `loss.backward()` accumulates gradients into the spline
parameters but no step is taken. `zero_grad` is **not** called either, so those
gradients are still present when the **even** iteration runs its backward pass —
the step sees the sum of two iterations' gradients.

### CHANGE 2 — Chunked path (`_train_chunked`)

Identical pattern using `local_iter`:

```python
chunk_optimizer.step()                           # STEP2.1
chunk_optimizer.zero_grad(set_to_none=True)     # STEP2.1
if local_iter % 2 == 0:                         # STEP2.2 — spline steps every 2nd iter
    pose_optimizer.step()                        # STEP2.2
    pose_optimizer.zero_grad(set_to_none=True)  # STEP2.2
```

`chunk_optimizer` (the per-chunk Gaussian optimizer) is unaffected and steps
every `local_iter`.

### CHANGE 3 — Stage-gate print updated

```python
print("[STEP1.3] Main stage: spline unfrozen at iteration 2000 "
      "| spline steps every 2nd iter (STEP2.2)")  # STEP2.2
```

### CHANGE 4 — Frequency confirmation comments

The existing `iteration % 10 == 0` and `iteration % 5 == 0` guards in the
single-scene loop received confirmation comments — they are **correct as-is**
from Step 1.4 and were not changed:

```python
if total_frames >= 2 and iteration % 10 == 0:  # STEP2.2 confirmed — already correct per spec
    ...  # L_jitter block

if total_frames > dilated_k and iteration % 5 == 0:  # STEP2.2 confirmed — already correct per spec
    ...  # L_dilated block
```

---

## Gradient accumulation behaviour

| Iteration | Gaussian opt | Spline opt | Spline gradient state after |
|-----------|-------------|------------|---------------------------|
| odd (1, 3, 5, …) | `step()` + `zero_grad` | — (no step, no zero_grad) | accumulating |
| even (2, 4, 6, …) | `step()` + `zero_grad` | `step()` + `zero_grad` | cleared |

The spline step on even iterations uses the **sum of two backward passes**
(odd + even), giving it 2× the gradient magnitude of a single-iteration step
at the same learning rate. This is equivalent to doubling the effective spline
learning rate every other iteration without changing `opt.pose_lr_init`.

Runtime simulation in `verify-2.2.py` confirms:

```
expected: param = 5.0 - 2.0 = 3.0   (2 accumulated grads of 1.0 each, lr=1.0)
actual:   3.0  ✓
```

---

## What is unchanged

| Element | Status |
|---------|--------|
| `L_jitter` frequency (`iteration % 10 == 0`) | **Unchanged** — set in Step 1.4 |
| `L_dilated` frequency (`iteration % 5 == 0`) | **Unchanged** — set in Step 1.4 |
| Loss weights `w_smooth / w_jitter / w_fov / w_dilated` | **Unchanged** |
| `lambda_dssim = 0.2` | **Unchanged** |
| Spline warm-up freeze / unfreeze logic (Step 1.3) | **Unchanged** |
| `scene/camera_spline.py` | **Not touched** |
| `render_static` signature | **Not touched** |
| Gaussian `optimizer.step()` — every iteration | **Unchanged** |

---

## `verify-2.2.py` fixes

The verifier shipped with two bugs that were fixed as part of this step:

### Fix 1 — Variable-width lookbehind removed

`re.compile(r"(?<!if\s.{0,40}...)")` used a variable-width lookbehind, which
Python 3.9's `re` module rejects with:

```
re.error: look-behind requires fixed-width pattern
```

The compiled `gauss_step_pattern` was never referenced — the actual check used
a while-loop below it. The dead `re.compile(...)` block was removed.

### Fix 2 — Regex `{0,500}` windows widened

The jitter and dilated frequency patterns searched for `% 10 == 0` →
`loss_jitter` within 500 characters. The two `render_static` calls between the
guard and the assignment produce a gap of ~1 543 chars. Similarly, the dilated
block's four render calls produce a gap of ~2 442 chars.

| Pattern | Old limit | New limit | Actual gap |
|---------|-----------|-----------|-----------|
| `% 10 → loss_jitter` | `{0,500}` | `{0,2000}` | ~1 543 chars |
| `% 5 → loss_dilated` | `{0,500}` | `{0,3000}` | ~2 442 chars |

---

## Verification

```bash
python verify-2.2.py
```

**22/22 checks pass**, including:

- `# STEP2.2` comments present (9 occurrences).
- `pose_optimizer.step()` conditional on `iteration % 2 == 0`.
- Gaussian `optimizer.step()` not gated by `% 2`.
- `pose_optimizer.zero_grad` co-located inside the `% 2` block.
- Same conditional (`local_iter % 2 == 0`) inside `_train_chunked`.
- `L_jitter` frequency `iteration % 10 == 0` confirmed unchanged.
- `L_dilated` frequency `iteration % 5 == 0` confirmed unchanged.
- Stage-gate print contains `STEP2.2` note.
- Confirmation comments on jitter / dilated frequency lines.
- `camera_spline.py` has zero `STEP2.2` marks.
- All four loss weights unchanged.
- `render_static` present and unchanged.
- `lambda_dssim = 0.2` unchanged.
- Runtime simulation: spline unchanged on odd, updated on even,
  accumulated gradients = 2× single-step gradient.

---

## Files changed

| File | Change |
|------|--------|
| `train_static_core.py` | `if iteration % 2 == 0` guard on `pose_optimizer.step()` (single-scene path + `_train_chunked`); stage-gate print; frequency confirmation comments |
| `verify-2.2.py` | Removed dead variable-width lookbehind; widened `{0,500}` → `{0,2000}` / `{0,3000}` for jitter / dilated frequency patterns |

**Not touched:** `scene/camera_spline.py`, all `utils/` modules, `train_entrypoint.py`.

---

## How to run

```bash
python train_static_core.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run
python verify-2.2.py   # 22/22
```

Phase 2 (Steps 2.1 + 2.2) is complete. Ready for Phase 3 (dynamic masking).

---

## Append-only history

### 2026-03-26 — Step 2.2 implemented

- Added `if iteration % 2 == 0` guard on `pose_optimizer.step()` in the
  single-scene path and `if local_iter % 2 == 0` in `_train_chunked`.
- Updated stage-gate print with `STEP2.2` note.
- Added confirmation comments on jitter (`% 10`) and dilated (`% 5`) guards.
- Fixed `verify-2.2.py`: removed dead lookbehind compile; widened regex windows.
- All added/modified lines tagged `# STEP2.2`.

---

## Assistant code change log

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-26 | `train_static_core.py` | Asymmetric spline stepping: `if iteration % 2 == 0` / `if local_iter % 2 == 0` guards; stage-gate print update; frequency confirmation comments. |
| 2026-03-26 | `verify-2.2.py` | Removed variable-width lookbehind (`re.error` on Py 3.9); widened `{0,500}` → `{0,2000}` / `{0,3000}` for jitter / dilated patterns. |
