# Step 2.1 — Chunked Windowed Optimisation for Long Videos

Introduces a **chunked training path** that fires automatically when
`total_frames > CHUNK_THRESHOLD`. For short videos the original single-scene
loop (Steps 1.1–1.4) runs **unchanged** in the `else`-branch.

**Builds on:** Step 1.4 (four stability losses active in main stage; cam_spline
warm-up / unfreeze; `pose_optimizer` carries spline param group)

---

## Motivation

A single `GaussianModel` trained on a very long sequence (~150+ frames) cannot
reasonably represent every frame's scene content, and the L-smooth / L-fov
stability sums over all N frames become expensive. Splitting into overlapping
windows gives each chunk its own independent Gaussian scene while keeping the
**global spline** shared — so pose continuity is preserved across chunk
boundaries.

---

## Constants

Added at the top of `train_static_core(...)`, after the stability-loss weights
and before model init:

| Constant | Value | Meaning |
|----------|-------|---------|
| `CHUNK_THRESHOLD` | `150` | Use chunked path when `total_frames > 150` |
| `CHUNK_SIZE` | `70` | Frames per chunk |
| `OVERLAP` | `20` | Frames of overlap between adjacent chunks |

All three tagged `# STEP2.1`.

---

## `build_chunk_indices` (new top-level function)

```python
def build_chunk_indices(total_frames, chunk_size, overlap):  # STEP2.1
```

Returns a list of `(start, end)` tuples (end exclusive) covering
`[0, total_frames)` with the requested step size `chunk_size − overlap`.
The last chunk always clips to `total_frames` and may be smaller than
`chunk_size`.

### Example

```
build_chunk_indices(150, 70, 20)
→ [(0, 70), (50, 120), (100, 150)]
```

Step = 70 − 20 = 50 frames per advance; last chunk clips at 150.

### Output for a 300-frame video

```
[(0, 70), (50, 120), (100, 170), (150, 220), (200, 270), (250, 300)]
→ 6 chunks → 6 independent GaussianModel instances
```

---

## Branching in `train_static_core`

Immediately after the cam_spline freeze (end of the Step 1.3 warm-up setup):

```python
use_chunked = (total_frames > CHUNK_THRESHOLD)  # STEP2.1

if use_chunked:          # STEP2.1 — long video path
    chunk_list = build_chunk_indices(total_frames, CHUNK_SIZE, OVERLAP)
    print(f"[STEP2.1] Chunked mode: {len(chunk_list)} chunks, ...")
    _train_chunked(chunk_list, cam_spline, pose_optimizer, train_cams,
                   pose_holder, dataset, opt, background, T_ref_fov,
                   args.model_path)
else:                    # STEP2.1 — short video, original single-scene path unchanged
    <existing training loop — zero changes>
```

The `else`-branch is a byte-for-byte copy of the Step 1.4 loop; no
modifications were made inside it.

---

## `_train_chunked` (new top-level function)

### Signature

```python
def _train_chunked(
    chunk_list,         # list of (c_start, c_end) tuples
    cam_spline,         # global CameraSpline — shared, updated by all chunks
    pose_optimizer,     # global Adam (pose_network + cam_spline params)
    all_train_cameras,  # full Camera list for all frames
    pose_holder,        # GaussianModel carrying _posenet / focal_bias
    dataset,            # ModelParams
    opt,                # OptimizationParams
    background,         # background colour tensor
    T_ref_fov,          # frozen low-frequency translation reference [N, 3]
    model_path,         # output directory for per-chunk checkpoints
)
```

### Per-chunk setup (4a–4d)

For each `(c_start, c_end)` in `chunk_list`:

| Step | Action |
|------|--------|
| **4a** | `chunk_gaussians = GaussianModel(dataset)` — independent scene per chunk |
| **4a** | `Scene(dataset, chunk_gaussians, chunk_gaussians, load_coarse=None)` — initialises point cloud (Scene does not accept `frame_range`; cameras are filtered manually) |
| **4a** | `chunk_cameras = [cam for cam in all_train_cameras if c_start <= cam.uid < c_end]` |
| **4a** | `chunk_gaussians.training_setup(opt, stage="fine_static")` |
| **4a** | `chunk_optimizer = chunk_gaussians.optimizer` |
| **4b** | Warm-start from prior scene skipped (no clone/copy method); noted as TODO |
| **4c** | Freeze spline: `for p in cam_spline.parameters(): p.requires_grad_(False)` |

### Per-chunk warm-up constant

```python
CHUNK_WARMUP = 500  # STEP2.1 — shorter warm-up per chunk
```

The spline is re-frozen at the start of each chunk and unfrozen at
`local_iter == CHUNK_WARMUP`, mirroring the Step 1.3 global gate but scoped
to each chunk's local iteration counter.

### Per-chunk training loop (4e)

```
iters_per_chunk = opt.iterations // len(chunk_list)
```

Each iteration:

1. **Stage gate** — at `local_iter == CHUNK_WARMUP`: unfreeze spline, print log.
2. **Sample** — `viewpoint_cam = random.choice(chunk_cameras)`.
3. **Pose** — `set_camera_pose_from_spline(viewpoint_cam, cam_spline, focal, cam_id)` where `cam_id = viewpoint_cam.uid` (absolute frame index).
4. **Render** — `render_static(..., stat_pc=chunk_gaussians, dyn_pc=chunk_gaussians, ...)`.
5. **Photometric loss** — identical `L1 + 0.2*(1 − SSIM)` formula.
6. **Stability losses** — same four terms, same weights, same frequencies, substituting `chunk_gaussians` for `stat_gaussians` and `chunk_cameras` for the global camera list. Active after `local_iter >= CHUNK_WARMUP`.
7. **Backward + steps** — `chunk_optimizer.step()`, then `pose_optimizer.step()` (global spline).

### Stability loss adaptations inside chunks

| Term | Frequency | Adaptation |
|------|-----------|------------|
| L_smooth | every iter (main stage) | Loop over `range(c_start, c_end)` (absolute indices); divide by `chunk_n_frames` |
| L_fov | every iter (main stage) | Slice `T_ref_fov[c_start:c_end]`; loop over absolute indices |
| L_jitter | every 10 iters | `t_pair = randint(0, chunk_n_frames-2)`; use `chunk_cameras[t_pair].uid` for spline frame_idx |
| L_dilated | every 5 iters | `t0 = randint(0, chunk_n_frames-dilated_k-1)`; use `chunk_cameras[t0].uid` / `.uid` |

### End of chunk

```python
print(f"[STEP2.1] Chunk {chunk_idx} done (frames {c_start}–{c_end-1})")
```

Optional checkpoint saved to `output/<expname>/point_cloud/chunk_NN_final/point_cloud_static.ply`.

---

## `train_entrypoint.py`

One comment added at the dispatch site:

```python
# STEP2.1: chunked windowed path active for total_frames > CHUNK_THRESHOLD
```

---

## Global spline continuity

`pose_optimizer` is constructed once in `train_static_core` (carries
`pose_network` params + `cam_spline.parameters()`), then **passed by reference**
into `_train_chunked`. Every chunk's inner loop calls:

```python
pose_optimizer.step()      # STEP2.1 — global spline optimizer
pose_optimizer.zero_grad(set_to_none=True)
```

This means all chunks jointly update the **same** spline, giving coherent pose
continuity across chunk boundaries at the cost of some gradient interference
(acceptable trade-off at this stage; see Step 2.2 for asymmetric stepping).

---

## Short-video path unchanged

When `total_frames <= 150` the `else`-branch executes. It is structurally
identical to Step 1.4 — same iteration counter, same tqdm progress bar, same
optimizer calls. Running `verify-1.4.py` on a short-video run exercises only
this branch and passes without modification.

---

## Verification (`verify-2.1.py`)

Static checks confirm:

- `# STEP2.1` markers present.
- `build_chunk_indices` defined as a top-level function.
- `_train_chunked` defined as a top-level function.
- `CHUNK_THRESHOLD / CHUNK_SIZE / OVERLAP` constants present.
- `use_chunked = (total_frames > CHUNK_THRESHOLD)` branch present.
- `chunk_cameras` filtered by `cam.uid`.
- `pose_optimizer.step()` called inside `_train_chunked`.
- `else`-branch (short-video) structurally unchanged from Step 1.4.

```bash
python verify-2.1.py
```

---

## Files changed

| File | Change |
|------|--------|
| `train_exp.py` | `build_chunk_indices`, `_train_chunked`, constants, `use_chunked` branch, `else`-wrap of existing loop |
| `train_entrypoint.py` | One `# STEP2.1` comment |

**Not touched:** `scene/camera_spline.py`, `scene/gaussian_model.py`,
`gaussian_renderer/__init__.py`, all stability loss utils, all loss weights.

---

## How to run

```bash
# Short video (≤150 frames) — original single-scene path
python train_exp.py -s data/nvidia_rodynrf/<SCENE>/ --expname my_run

# Long video (>150 frames) — chunked path fires automatically
python train_exp.py -s data/long_video/ --expname long_run
```

---

## Append-only history

### 2026-03-26 — Step 2.1 implemented

- New top-level functions `build_chunk_indices` and `_train_chunked` in `train_exp.py`.
- `CHUNK_THRESHOLD=150`, `CHUNK_SIZE=70`, `OVERLAP=20` constants.
- `use_chunked` branch; existing loop moved into `else`.
- One comment in `train_entrypoint.py`.
- All added/modified lines tagged `# STEP2.1`.

---

## Assistant code change log

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-26 | `train_exp.py`, `train_entrypoint.py` | Step 2.1: `build_chunk_indices`, `_train_chunked`, constants, `use_chunked` branch, `else`-wrap. |
