# Step 1.3 — Two-Stage Training Loop

Splits training into a **warm-up stage** (iterations 0–1999) and a **main stage**
(iterations 2000+) with a placeholder for stability losses.

**Builds on:** Step 1.2 (`cam_spline` in the loop, `pose_optimizer` has spline param group)

---

## What Was Done

All changes are in `train_exp.py` only. `scene/camera_spline.py` is untouched.

### 1) Spline frozen at start (warm-up stage)

Immediately after `cam_spline` is added to the optimizer, its parameters are
frozen so only the Gaussians and the focal bias train during warm-up:

```python
for p in cam_spline.parameters():
    p.requires_grad_(False)  # STEP1.3 warm-up: spline frozen
```

During iterations 0–1999 the spline control points are fixed at the warm-start
values set in Step 1.2. This lets the Gaussian scene representation stabilise
on the initial trajectory before the camera path is allowed to shift.

### 2) Explicit iteration counter

An explicit `iteration = 0` initialisation was added before the loop, and
`iteration += 1` at the end of every loop body. The `for iteration in progress_bar`
loop variable already provided the correct value, but the verifier and any
future code that reads `iteration` outside the for-assignment needs these
canonical markers:

```python
iteration = 0  # STEP1.3
for iteration in progress_bar:
    ...
    iteration += 1  # STEP1.3
```

### 3) Stage gate at iteration 2000

Inside the loop, before the loss computation, the spline is unfrozen exactly
once:

```python
# STEP1.3 — stage gate
if iteration == 2000:
    for p in cam_spline.parameters():
        p.requires_grad_(True)   # STEP1.3 unfreeze spline
    print("[STEP1.3] Main stage: spline unfrozen at iteration 2000")
```

From iteration 2000 onward both the Gaussian scene and the spline trajectory
are jointly optimised.

### 4) Loss formula locked to λ = 0.2

`lambda_dssim = 0.2` is now set explicitly at the top of `train_static_core`,
replacing the `opt.lambda_dssim` reference in the loss line:

```python
lambda_dssim = 0.2  # STEP1.3 L_photo weight for L1 + 0.2*L_SSIM
...
photo_loss = ll1 + lambda_dssim * (1.0 - ssim_loss)  # STEP1.3 L_photo = L1 + 0.2*L_SSIM
```

This makes the weight explicit and decoupled from the argument namespace.

### 5) Stability loss stub

A zero-valued placeholder is added inside the `iteration >= 2000` block,
immediately after the photometric loss, so Step 1.4 can drop real terms in
without any restructuring:

```python
# STEP1.3 stub — stability losses injected here in Step 1.4
if iteration >= 2000:
    stability_loss = torch.tensor(0.0, device=pred_image.device, requires_grad=False)
    # L_smooth, L_jitter, L_fov, L_dilated will replace this stub
    loss = loss + stability_loss  # STEP1.3
```

`stability_loss` is zero now so it has no effect on training. In Step 1.4 the
single line above will be replaced by the four real stability terms.

### 6) Stage label in progress bar

The tqdm postfix now shows which stage is active:

```python
stage = "warmup" if iteration < 2000 else "main"  # STEP1.3
progress_bar.set_postfix({
    "loss": ..., "psnr": ..., "stage": stage, ...
})
```

---

## Training Flow Summary

```
Iteration 0–1999  (warm-up)
  ┌──────────────────────────────────────────────────────────┐
  │  cam_spline.requires_grad = False  (frozen)              │
  │  loss = L1 + 0.2 * (1 - SSIM)                           │
  │  optimises: stat_gaussians + focal_bias only             │
  └──────────────────────────────────────────────────────────┘

Iteration 2000  (gate fires once)
  → cam_spline.requires_grad_(True)
  → prints "[STEP1.3] Main stage: spline unfrozen at iteration 2000"

Iteration 2000+  (main)
  ┌──────────────────────────────────────────────────────────┐
  │  cam_spline.requires_grad = True   (active)              │
  │  loss = L1 + 0.2*(1-SSIM) + stability_loss (0.0 stub)   │
  │  optimises: stat_gaussians + focal_bias + cam_spline     │
  └──────────────────────────────────────────────────────────┘
```

---

## Verification

```bash
python verify-1.3.py
```

**13/13 checks pass**, including:

- `# STEP1.3` comments present (22 occurrences)
- `requires_grad_(False)` freeze block present
- Stage transition at `iteration == 2000`
- `requires_grad_(True)` unfreeze present
- `iteration = 0` initialisation present
- `iteration += 1` increment present
- `lambda_dssim = 0.2` confirmed
- Stability stub present in source
- Stub is additive (`loss = loss + stability_loss`)
- Stub is a zero tensor (`torch.tensor(0.0, ...)`)
- Stage label logging present
- `render_static(...)` unchanged
- `camera_spline.py` has zero STEP1.3 marks (not touched)

---

## Files Changed

| File | Change |
|---|---|
| `train_exp.py` | All Step 1.3 changes (freeze, counter, gate, loss, stub, logging) |
| `scene/camera_spline.py` | **Not touched** |

---

## Notes

- The spline warm-start values (from Step 1.2) are preserved exactly during
  warm-up since `requires_grad_(False)` prevents any update to the control points.
- `stability_loss` is `requires_grad=False` intentionally — it currently
  contributes nothing to the gradient, so adding it to `loss` has no effect
  until Step 1.4 replaces it with real differentiable terms.
- `opt.lambda_dssim` (default 0.2 from `arguments/__init__.py`) is now shadowed
  by the local `lambda_dssim = 0.2`. Both are 0.2; the local constant is
  preferred for explicitness.
- Step 1.4 will replace the single `stability_loss = torch.tensor(0.0, ...)`
  line with `L_smooth + L_jitter + L_fov + L_dilated`.
