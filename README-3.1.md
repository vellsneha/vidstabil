cd vidstabil
python preprocess_dynamic_masks.py -s /path/to/scene --backend gsam2 --text-prompt "person . car ."
python train_entrypoint.py -s /path/to/scene --expname run --use_dynamic_mask

# Step 3.1 — Masked Photometric Loss (Dynamic Objects)

Exclude **moving-object pixels** from the photometric term so transient content (people, cars, etc.) does not create “ghost Gaussians” or dominate \( \mathcal{L}_{photo} \).

**Builds on:** Phase 2 training (`train_static_core.py`, chunked and single-scene paths). Masks are **precomputed** once per video and **cached** on disk; training only reads them.

---

## Motivation

Static-scene stabilisation should fit the **background**. Regions that move independently frame-to-frame violate the static-world assumption. [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) (Grounding DINO + SAM 2) supplies \(M_t\): value **1** on dynamic objects, **0** on background. The loss uses only background pixels:

\[
\mathcal{L}_{photo} = \left\| (I_{\mathrm{rendered}} - I_{\mathrm{input}}) \odot (1 - M_t) \right\|_1
\]

plus the existing DSSIM term, weighted the same way as before, with both \(I_{\mathrm{rendered}}\) and \(I_{\mathrm{input}}\) multiplied by \((1 - M_t)\) per channel so masked regions do not drive SSIM.

Implementation detail: the L1 part matches the codebase convention in `l1_loss` — sum of absolute differences times weights, divided by `sum(weights) + ε` (masked mean), not an unnormalised sum over all pixels.

---

## Change 1 — `ModelParams` flags

**File:** `arguments/__init__.py`

| Field | Default | Meaning |
|-------|---------|---------|
| `use_dynamic_mask` | `False` | When `True`, load per-frame masks from the scene directory. |
| `dynamic_mask_subdir` | `"dynamic_masks"` | Subfolder under `<source_path>` containing `{t:03d}.png` masks. |

---

## Change 2 — Cached mask layout (\(M_t\))

For each training frame `t` (same indexing as `images_2/{t:03d}.png`):

```
<source_path>/<dynamic_mask_subdir>/{t:03d}.png
```

- Single-channel PNG.
- **255** (or any value \(>\) 0.5 after normalisation) = **moving** (\(M_t = 1\)).
- **0** = background (\(M_t = 0\)).

Masks are resized to the frame resolution inside `readNvidiaCameras` if needed (same `PILtoTorch` path as RGB).

---

## Change 3 — `photometric_loss_masked_dynamic` in `utils/loss_utils.py`

**Function:** `photometric_loss_masked_dynamic(pred, gt, M_t, lambda_dssim, ssim_fn)`

- `M_t`: tensor `[1, H, W]`, values in \([0, 1]\), **1 on dynamic** pixels.
- Builds `w = 1 - M_t` and calls `l1_loss(pred, gt, mask=w)`.
- If `lambda_dssim > 0`, computes SSIM on `pred * w` and `gt * w` (expanded to 3 channels).

---

## Change 4 — `Camera` and data loading

**Files:** `scene/dataset_readers.py`, `scene/cameras.py`, `scene/dataset.py`

- `CameraInfo` gains optional `dynamic_mask_t` (numpy \(H \times W\) float).
- `Camera` stores `dynamic_mask_t` as a float tensor `[1, H, W]` on `data_device`.
- `FourDGSdataset` passes `dynamic_mask_t` from `caminfo` into `Camera`.

When `use_dynamic_mask` is `True` but mask files are missing, a **single** warning is printed and those frames fall back to **unmasked** photometric loss until files exist.

---

## Change 5 — Training loop (`train_static_core.py`)

- After `use_masked_photo = bool(dataset.use_dynamic_mask)`, both the **single-scene** and **`_train_chunked`** branches call `photometric_loss_masked_dynamic` when `viewpoint_cam.dynamic_mask_t` is not `None`.
- If masks are disabled or missing, behaviour matches **Phase 2** (plain L1 + DSSIM).

---

## Preprocessing: integrated Grounded SAM 2

Masks are generated **once** before training by `preprocess_dynamic_masks.py`, which calls `gsam2/integrated.py`. That code follows the upstream **HuggingFace Grounding DINO + SAM 2 image predictor** path (same idea as `grounded_sam2_hf_model_demo.py` in the official repo).

### Step A — Clone upstream (once per machine)

```bash
cd /path/to/vidstabil/project   # parent of inner vidstabil/ package
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
```

Override location with `VIDSTABIL_GSAM2_ROOT` if the clone lives elsewhere.

### Step B — Install SAM 2 from that clone

Follow [Grounded-SAM-2 INSTALL.md](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/INSTALL.md): typically `cd third_party/Grounded-SAM-2 && pip install -e .` (CUDA/PyTorch as documented there).

### Step C — Download checkpoints

```bash
cd third_party/Grounded-SAM-2/checkpoints && bash download_ckpts.sh
cd ../gdino_checkpoints && bash download_ckpts.sh   # if using local Grounding DINO; HF tiny model downloads at runtime
```

### Step D — Python deps for VidStabil’s wrapper

```bash
pip install -r requirements-gsam2.txt
```

### Step E — Run preprocessing (cache masks)

Default backend is **`gsam2`** (integrated). Typical runtime is on the order of **~30 seconds for a ~10 s clip** on a mid-range GPU (varies with resolution, frame count, and model choice).

```bash
cd vidstabil
python preprocess_dynamic_masks.py -s /path/to/scene --backend gsam2 --text-prompt "person . car ."
```

Options include `--grounding-model`, `--sam2-checkpoint`, `--sam2-model-config`, `--box-threshold`, `--text-threshold`, `--force-cpu`.

### Backend — `synthetic` (tests only)

No ML; draws a moving ellipse so layout and `verify-3.1.py` can run without GPU weights:

```bash
python preprocess_dynamic_masks.py -s /path/to/scene --backend synthetic
```

---

## Training with masks

```bash
python train_entrypoint.py -s /path/to/scene --expname masked_run --use_dynamic_mask
```

Ensure `dynamic_masks/` exists and aligns with `images_2/` frame count when `use_dynamic_mask` is enabled.

---

## Verification

```bash
cd vidstabil
python verify-3.1.py
```

Checks include: `ModelParams` fields, loss implementation, dataset and training wiring, numeric sanity, `gsam2/integrated.py`, and a **synthetic** preprocess run (no Grounded SAM 2 weights required).

---

## Files touched (Step 3.1)

| File | Role |
|------|------|
| `arguments/__init__.py` | `use_dynamic_mask`, `dynamic_mask_subdir` |
| `utils/loss_utils.py` | `photometric_loss_masked_dynamic` |
| `scene/dataset_readers.py` | Load `dynamic_masks/{t:03d}.png`; `CameraInfo.dynamic_mask_t` |
| `scene/cameras.py` | `Camera.dynamic_mask_t` |
| `scene/dataset.py` | Pass-through to `Camera` |
| `train_static_core.py` | Masked L_photo in both training branches |
| `train_entrypoint.py` | Comment pointer to Step 3.1 |
| `gsam2/integrated.py` | In-process Grounded SAM 2 mask generation |
| `preprocess_dynamic_masks.py` | CLI: `gsam2` (default) or `synthetic` |
| `requirements-gsam2.txt` | Extra pip deps for preprocessing |
| `verify-3.1.py` | Automated verification |

---

## What is unchanged

| Element | Status |
|---------|--------|
| Densification / `MAX_GAUSSIANS` | **Unchanged** |
| Stability losses (`w_smooth`, `w_jitter`, `w_fov`, `w_dilated`) | **Unchanged** |
| Spline warm-up / unfreeze schedule | **Unchanged** |
| Default `use_dynamic_mask` | **`False`** (opt-in) |

---

## Append-only history

### 2026-03-30 — Step 3.1 implemented

- Masked photometric loss with cached \(M_t\); `verify-3.1.py`.

### 2026-03-30 — Integrated Grounded SAM 2 preprocessing

- `gsam2/integrated.py` + default `preprocess_dynamic_masks.py --backend gsam2`; `requirements-gsam2.txt`; upstream clone under `third_party/Grounded-SAM-2`.

---

## Assistant code change log

| Date | Files | Summary |
|------|-------|---------|
| 2026-03-30 | `arguments/__init__.py` | `use_dynamic_mask`, `dynamic_mask_subdir` |
| 2026-03-30 | `utils/loss_utils.py` | `photometric_loss_masked_dynamic` |
| 2026-03-30 | `scene/*`, `train_static_core.py` | Load and apply `dynamic_mask_t` |
| 2026-03-30 | `gsam2/*`, `preprocess_dynamic_masks.py`, `requirements-gsam2.txt`, `README-3.1.md`, `verify-3.1.py` | Integrated GSAM2 + docs + verification |
