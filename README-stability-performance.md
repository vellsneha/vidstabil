# Stability losses: performance work and correctness notes

This document explains **why** main-stage training slowed down dramatically after iteration 2000, **what** we changed to fix it, and **pitfalls** encountered (CUDA indexing, loss aggregation). It is meant for anyone maintaining `train_static_core.py` or `scene/camera_spline.py`.

---

## 1. Symptom: warm-up fast, main stage orders of magnitude slower

Observed pattern (typical):

- **Iterations 0–1999 (warm-up):** spline parameters frozen (`requires_grad=False`). Throughput on the order of **tens of iterations per second**.
- **Iteration 2000+ (main stage):** stability losses active, spline trainable. Throughput can drop to **~2 it/s** or worse for long sequences.

A naive reading is “stability losses are 30–40× more expensive than photometric loss.” That is **misleading**. The photometric path is similar in both stages; the dominant change is **how much work is added per step** and **whether gradients flow into the spline** (see §3).

---

## 2. Root cause: cost scaled with sequence length \(N\), not “loss complexity”

### 2.1 What the spec actually asks for

For **\(L_{\text{smooth}}\)** and **\(L_{\text{fov}}\)** the intended quantities are **per-frame** objectives, then averaged over **all frames** \(t = 0,\ldots,N-1\):

- \(L_{\text{smooth}} \propto \frac{1}{N} \sum_t \| d^2 T / dt^2 (t) \|^2\) (translation Hermite spline).
- \(L_{\text{fov}} \propto \frac{1}{N} \sum_t \| T(t) - \bar{T}(t) \|^2\) with \(\bar{T}\) the frozen low-frequency reference from `frozen_low_frequency_translation_reference`.

So the **mathematical** object is an average over **\(N\)** frames.

### 2.2 What made training slow

An implementation that, **every optimizer step**, does a **full pass over all \(N\)** frame indices (e.g. a Python `for t in range(total_frames)` or building a full `[N, 3]` tensor and backpropagating through it) makes the cost of those two terms **\(O(N)\) per iteration**.

Warm-up **does not run** the stability block at all (`iteration < 2000`). So you go from **zero** \(N\)-dependent stability work to **full \(N\)** work in one step. For \(N\) in the hundreds or thousands, that is not a modest “3–5×” slowdown from “extra loss terms”; it is a **large constant × \(N\)** effect, easily **tens of times** in wall time when \(N\) is large.

Other costs in main stage (extra renders for jitter / dilated losses on a schedule, backward through the spline for photometric loss) matter, but **the \(O(N)\) smooth/FoV terms were the main structural problem** for long sequences.

### 2.3 What we did *not* rely on for the fix

- **Halving spline optimizer steps (STEP2.2)** reduces how often `pose_optimizer.step()` runs; it does **not** remove an \(O(N)\) forward pass over all frames each iteration.
- **Vectorizing** a loop over \(N\) into one batched kernel helps constant factors, but **still** scales linearly in \(N\) for forward and backward unless we **reduce how many frame indices participate per step**.

---

## 3. Solution: subsampled Monte Carlo estimate of the same objective

### 3.1 Idea

Each iteration, we draw a set of frame indices `idx` of size at most **`STABILITY_LOSS_FRAME_SAMPLE`** (default **256**). We compute spline quantities **only** at those indices via:

- `CameraSpline.get_translation_second_derivatives_at(idx)`
- `CameraSpline.get_translations_at(idx)`

and form **empirical means** over the sample:

- `loss_smooth = (acc ** 2).sum(dim=-1).mean()`
- `loss_fov = ((Ts_now - Tbar_s) ** 2).sum(dim=-1).mean()`

with `Tbar_s = T_ref_fov[idx]`.

If `idx` is **uniform** over `0..N-1` (implemented with `torch.randperm(N)[:M]` when `M < N`), then these means are **unbiased estimators** of the full-batch averages \(\frac{1}{N}\sum_t (\cdot)\) for the smoothness energy and FoV energy **up to the usual Monte Carlo variance**. That is the standard trick: same **expected** objective, **\(O(M)\)** work per step with \(M = \min(256, N)\).

When `N ≤ 256`, we use **all** frames (`torch.arange(N)`), recovering the full deterministic mean.

### 3.2 Where it lives

- **Constant:** `STABILITY_LOSS_FRAME_SAMPLE` at the top of `train_static_core.py`.
- **Helper:** `_stability_loss_frame_indices(n_frames, max_samples, device)` for the single-scene path.
- **Chunked path:** subsamples **within** the chunk `[c_start, c_end)` by drawing `rel` in `0..chunk_n_frames-1` and mapping `idx = rel + c_start`, so indices stay aligned with global `T_ref_fov` and the spline’s frame indexing.

### 3.3 Tuning

- **Increase** `STABILITY_LOSS_FRAME_SAMPLE` (e.g. 512) if you see noisy training or need tighter convergence; cost grows roughly linearly in that cap.
- **Decrease** (e.g. 128) for speed; variance of the gradient estimate increases.

---

## 4. Correct aggregation: per-frame energy, then mean over frames

The spec uses **squared Euclidean norm per frame**, then average over time.

Correct pattern:

```text
per_frame = (vec ** 2).sum(dim=-1)   # shape [M]
loss = per_frame.mean()
```

A **bug** to avoid is averaging over **all scalar components** as `(vec ** 2).mean()` over `[M, 3]`, which is **one third** of the intended per-frame mean of \(\|v\|^2\) when each row is a 3-vector. The implementation uses **`.sum(dim=-1).mean()`** so each frame contributes one nonnegative scalar (the squared norm), then we average over frames.

---

## 5. Spline implementation: batched-by-indices API

`CameraSpline` exposes:

- **`_translation_second_derivatives_for_t_frame(t_frame)`** — `t_frame` shape `[M]`, float frame indices in `[0, N-1]`.
- **`_translations_for_t_frame(t_frame)`** — same.
- **`get_translation_second_derivatives_at(frame_indices)`** / **`get_translations_at(frame_indices)`** — integer indices `[M]`.
- **`get_all_translation_second_derivatives()`** / **`get_all_translations()`** — full `0..N-1` via `torch.arange(N)` and the same helpers.

This keeps **one** implementation of Hermite translation math and allows either **full** or **partial** frame sets without a Python loop over `t`.

---

## 6. Bug fix: `torch.where` and out-of-bounds indexing on CUDA

### 6.1 What went wrong

Hermite tangents use neighbor control points, e.g. expressions involving `ctrl_trans[i + 2]` for segment index `i`. For the **last** valid segment, `i = K - 2`, the expression `i + 2` equals **`K`**, which is **invalid** for a tensor of shape `[K, 3]` (valid indices `0..K-1`).

In **scalar** code paths (`get_pose`, `get_translation_second_derivative`), Python `if` branches **never evaluate** the invalid index on the boundary.

In **vectorized** code, **`torch.where` evaluates both branches** before masking. So even when the mask selects the boundary fallback (`p1 - p0`), the **other** branch still computes `ctrl_trans[i + 2]`, triggering a **device-side assert** in CUDA index kernels.

That showed up **exactly at iteration 2000** when stability first called the batched helpers.

### 6.2 Fix

Use **clamped** indices for tensors used inside the branches, e.g.:

- `im1 = torch.clamp(i - 1, min=0)`
- `ip2 = torch.clamp(i + 2, max=K - 1)`

The value computed for the “wrong” branch at the boundary may be meaningless, but it is **masked out**; the **selected** value remains the correct fallback. See comments in `scene/camera_spline.py` next to `_translation_second_derivatives_for_t_frame` and `_translations_for_t_frame`.

### 6.3 Misleading stack traces

CUDA asserts are **asynchronous**. The Python line that raises `RuntimeError: device-side assert triggered` is often **not** the line that launched the bad kernel (e.g. it may appear on a later `tensor.cpu()` or unrelated op). Use **`CUDA_LAUNCH_BLOCKING=1`** once to get a traceback aligned with the failing kernel when debugging.

---

## 7. Scope: what this README does *not* change

- **\(L_{\text{jitter}}\)** and **\(L_{\text{dilated}}\)** still use their original **frequency** schedules (e.g. every 10 / 5 iterations) and extra `render_static` calls; those are separate cost knobs.
- **STEP2.2** asymmetric spline stepping is unchanged.
- **Chunk warm-up** (`CHUNK_WARMUP`) vs global iteration 2000 are **different gates**; chunked training uses the subsampled stability block after `local_iter >= CHUNK_WARMUP`.

---

## 8. Related files

| File | Role |
|------|------|
| `train_static_core.py` | `STABILITY_LOSS_FRAME_SAMPLE`, `_stability_loss_frame_indices`, single-scene + chunked stability blocks |
| `scene/camera_spline.py` | Batched Hermite helpers, `get_*_at`, `torch.where` safe indexing |
| `utils/fov_loss.py` | `frozen_low_frequency_translation_reference` for \(\bar{T}\) |
| `profile-2.4.py` | Profiling harness for per-component time (if you need to re-measure after changes) |

---

## 9. Summary

| Topic | Takeaway |
|-------|----------|
| **Why 38× vs “3–5×”** | Not “loss is heavy”; it was **\(O(N)\) work per step** for smooth/FoV vs **none** in warm-up, for large \(N\). |
| **Fix** | **Subsample frames** per iteration; same **expected** objective, **\(O(M)\)** with \(M \le 256\). |
| **Aggregation** | Use **`.sum(dim=-1).mean()`** for vector losses per frame. |
| **CUDA crash at 2000** | **`torch.where` + `ctrl_trans[i+2]`** OOB; fixed with **clamped indices**. |

This combination is what makes long-sequence static-core training **practical** under a wall-clock budget while preserving the **intent** of the stability terms.
