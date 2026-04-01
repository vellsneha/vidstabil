# README 3.1 Post-Run Fixes

This README consolidates the issues, fixes, and practical commands discovered after Phase 3, when the pipeline was tested on real data and then pushed through an attempted end-to-end run.

The main theme of this phase was that the codebase could run structurally with placeholder data, but real-data execution exposed several correctness, environment, and performance problems across:

- dataset preparation
- dynamic-mask preprocessing
- depth generation
- track generation
- static-core training
- legacy dynamic training
- stabilized rendering

The sections below keep the important takeaways and the commands that were actually useful.

---

## 1. High-level outcome

After the post-Phase-3 debugging work, the pipeline was brought much closer to a real-data workflow:

- real scenes can be prepared without relying on dummy placeholders
- depth generation works with practical defaults
- dynamic masks can be generated and reused across training paths
- bootstrapped track generation is more robust and less wasteful
- static-core training handles real-data edge cases better
- legacy dynamic training handles real masks and incomplete tracks more safely
- rendering is more robust to bad poses, missing source-path assumptions, and checkpoint-layout differences

The major remaining external blocker noted during this work was environment-specific rendering support for `gsplat` CUDA builds and GPU memory pressure during CoTracker track generation.

---

## 2. What broke during the full real-data run

### 2.1 Wrong dataset root during rendering

One of the first hard failures came from rendering with the wrong scene path. The renderer expected:

```text
<source_path>/images_2/*.png
```

but the command used a dataset root that did not contain that layout, which caused:

```text
IndexError: list index out of range
```

inside `readNvidiaCameras()`.

Important takeaway:

- `render_stabilized.py` does not render from checkpoint files alone
- it reconstructs the scene from the original dataset root passed via `-s/--source_path`
- if `cfg_args` contains an old or wrong path, pass `--source_path` explicitly

### 2.2 Invalid camera rotations during render verification

Another failure was:

```text
R not SO(3) at t=0
```

This came from invalid or numerically unstable spline rotations during render-time pose reconstruction.

Fix applied:

- render-time pose sanitization was added
- finite poses are projected to the nearest valid rotation matrix using SVD
- invalid spline outputs fall back to identity rotation and zero translation

### 2.3 `gsplat` environment/build problems

Even after the dataset-path issue was fixed, rendering was still blocked in one environment because `gsplat` was not actually available with a usable CUDA extension.

Observed problems:

- local source build complained about missing GLM headers
- vendor/submodule contents under the local `gsplat` tree appeared incomplete
- `pip install gsplat==1.4.0` did not provide the CUDA wrapper expected by this project
- `gsplat.cuda._wrapper` existed, but the compiled `_C` binding was missing

Important takeaway:

- a plain pip install was not a reliable drop-in replacement for the local project build
- if rendering still fails after code fixes, check the local `gsplat` build first

### 2.4 Dummy data was good enough to run, but not good enough to train/render well

Real testing confirmed that parts of the preprocessing flow had been producing placeholders for:

- `instance_mask`
- `uni_depth`
- `bootscotracker_dynamic`
- `bootscotracker_static`

These placeholders made the directory structure look valid, but they were not real supervision and could not support a faithful real-data run.

### 2.5 Degenerate scene initialization caused blank or useless renders

One important issue in earlier static-core runs was that the initial point cloud could collapse into a dummy zero-centered cloud. That produced symptoms such as:

- blank output
- flat gradient output
- `visible gaussians: 0`

Root cause:

- the initial PLY could be seeded from a useless dummy point instead of a real geometry estimate

Fix applied:

- `scene/dataset_readers.py` was updated to initialize the scene using sparse depth back-projection
- if depth is unavailable, it now falls back more safely instead of always producing a useless all-zero cloud

### 2.6 Static-core training became very slow after the spline stage started

Once the main stability stage started, training speed dropped sharply.

Root causes:

- `L_smooth` and `L_fov` were computed through expensive per-frame Python loops
- translation spline operations were repeated unnecessarily
- scalar `.item()` usage caused repeated CPU/GPU synchronization
- vectorization introduced an out-of-bounds CUDA bug due to `torch.where` evaluating both branches

Fixes applied:

- batched spline helpers were added in `scene/camera_spline.py`
- stability losses now sample frames instead of processing the full sequence each iteration
- `torch.where` indexing was made safe using clamped neighbor indices
- per-frame vector losses are aggregated correctly as squared norm per frame, then averaged

Important practical point:

- warm-up speed and main-stage speed are not directly comparable
- the main-stage slowdown was mostly an implementation cost issue, not just an inherent property of the loss terms

### 2.7 Track generation was too slow and memory-hungry

Real bootstrapped track generation exposed two practical problems:

- the original implementation did too much repeated work
- CoTracker could hit CUDA OOM on realistic settings

Fixes applied:

- generation changed from repeated pairwise query-target calls to one full-video call per query frame
- optional downscaling via `--max_hw` was added
- predicted coordinates are scaled back to original resolution before saving
- segmentation-mask shape bugs were fixed
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now used for track jobs

### 2.8 Legacy dynamic training did not match real mask behavior well

Real-data testing showed that the legacy dynamic path had multiple assumptions that were too brittle:

- it preferred older mask flows instead of real `dynamic_masks/`
- it had a hardcoded `12`-frame assumption in one path
- it assumed complete track availability if a track folder existed
- temporal filtering was too strict and could eliminate all dynamic points
- empty dynamic initialization could lead to zero-point setups and CUDA failures

Fixes applied:

- legacy training now prefers `dynamic_mask_t` when `--use_dynamic_mask` is enabled
- hardcoded frame-count assumptions were removed
- incomplete `bootscotracker_*` folders are treated as absent unless the full expected set exists
- temporal filtering now uses a configurable ratio threshold
- empty filtered sets fall back more safely instead of crashing

---

## 3. Important code and behavior fixes

### 3.1 Rendering fixes

Files involved:

- `render_stabilized.py`
- `render_stabilized_video.py`

Important changes:

- clearer and safer `--source_path` handling
- device handling no longer assumes CUDA blindly in several paths
- pose sanitization before verification
- spline-based pose application uses the same conventions as training
- clearer runtime error wrapping around low-level `gsplat` failures
- renderer now supports both:
  - static-core checkpoints
  - legacy dynamic checkpoints
- iteration directory discovery is more robust

### 3.2 Dataset and scene loading fixes

Files involved:

- `prepare_dataset.py`
- `scene/dataset_readers.py`

Important changes:

- scene prep now supports real CLI-driven dataset creation
- static-core no longer requires `instance_mask/` to exist
- missing mask folders can fall back safely
- missing or partial bootstracker directories are treated as optional instead of assumed-valid
- initial scene geometry is seeded from depth instead of a useless dummy point cloud

### 3.3 Depth generation fixes

File involved:

- `gen_depth.py`

Important change:

- default depth model was moved toward `v2`

Why:

- `v2old` hit missing dependency issues involving `NystromAttention`
- `v2` was the practical model that actually succeeded during the real-data run

### 3.4 Track generation fixes

File involved:

- `gen_tracks.py`

Important changes:

- faster full-video tracking per query frame
- lower-memory option through `--max_hw`
- correct coordinate rescaling
- corrected segmentation-mask shape handling

### 3.5 Static-core training fixes

Files involved:

- `train_static_core.py`
- `scene/camera_spline.py`
- `profile-2.4.py`

Important changes:

- stable spline pose handling
- safer `c2w` export without fragile raw inverse usage
- batched translation and derivative evaluation
- stability-loss frame subsampling via `STABILITY_LOSS_FRAME_SAMPLE`
- one-shot reference trajectory construction for `T_ref_fov`
- CLI cleanup for profiler usage

### 3.6 Legacy dynamic training fixes

Files involved:

- `train.py`
- `arguments/__init__.py`

Important changes:

- support for `dynamic_mask_consistency_ratio`
- safer dynamic seed selection
- ratio-based temporal mask consistency instead of requiring agreement in all frames
- safer fallback behavior when dynamic selection becomes empty

---

## 4. Practical commands

The commands below are the ones that mattered most during real-data setup, recovery, and reruns.

### 4.1 Prepare a real scene

```bash
cd /workspace/vidstabil

python prepare_dataset.py \
  --src-frames "/workspace/vidstabil/data/test_clip/images" \
  --scene "/workspace/vidstabil/data/crowd9_scene" \
  --gen-depth \
  --depth-model v2
```

### 4.2 Generate dynamic masks

```bash
python preprocess_dynamic_masks.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --backend gsam2 \
  --text-prompt "person ."
```

Recommended note:

- if the process seems quiet at startup, it may still be initializing GSAM2/SAM2 before the progress bar appears

Possible improved prompt for crowded retail scenes:

```text
person . shopping cart . hand . arm .
```

### 4.3 Generate real tracks

Set the allocator tweak first:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Dynamic tracks:

```bash
python gen_tracks.py \
  --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" \
  --mask_dir "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" \
  --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_dynamic" \
  --grid_size 64 \
  --max_hw 512
```

Static tracks:

```bash
python gen_tracks.py \
  --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" \
  --mask_dir "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" \
  --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_static" \
  --is_static \
  --grid_size 24 \
  --max_hw 512
```

### 4.4 Train the static-core path

```bash
python train_entrypoint.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run \
  --use_dynamic_mask \
  --iterations 10000
```

Alternative example with explicit model path and GPU preflight bypass:

```bash
python train_entrypoint.py \
  --skip-gpu-preflight \
  --source_path /workspace/vidstabil/data2/regular_scene \
  --model_path /workspace/vidstabil/output2/your_run \
  --expname your_run \
  --iterations 5000
```

Important training note:

- if static-core training runs in chunked mode, not seeing the usual per-iteration progress bar is expected
- repeated `Reading Nvidia Info` messages are also expected in chunked mode because each chunk rebuilds a fresh `Scene(...)`

### 4.5 Train the legacy dynamic path

```bash
python train_entrypoint.py \
  --legacy-dynamic \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --use_dynamic_mask \
  --iterations 10000
```

Useful temporal-mask tuning:

```bash
--dynamic_mask_consistency_ratio 0.3
```

Stricter:

```bash
--dynamic_mask_consistency_ratio 0.5
```

More permissive:

```bash
--dynamic_mask_consistency_ratio 0.2
```

### 4.6 Render a stabilized video

Static-core:

```bash
python render_stabilized.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run \
  --output_video "/workspace/vidstabil/output/my_run/stabilized.mp4"
```

Legacy dynamic:

```bash
python render_stabilized.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --output_video "/workspace/vidstabil/output/my_run_tracks/stabilized.mp4"
```

Preferred render path when using explicit spline controls:

```bash
python render_stabilized_video.py \
  --run-dir /workspace/vidstabil/output2/your_run \
  --source-path /workspace/vidstabil/data2/regular_scene \
  --trajectory /workspace/vidstabil/output2/your_run/point_cloud/static_core_final/cam_spline_controls.npz \
  --output /workspace/vidstabil/output2/your_run/stabilized.mp4 \
  --fps 24
```

Critical reminder:

- always point `--source_path` or `--source-path` to the dataset root that actually contains `images_2/`

### 4.7 Check whether a learned trajectory is basically static

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

Interpretation:

- very small translation range means the output may look like a frozen snapshot even if rendering is technically working

### 4.8 Recover a broken `gsplat` environment

```bash
cd /workspace/vidstabil
git submodule update --init --recursive
ls gsplat/gsplat/cuda/csrc/third_party/glm/glm/gtc/type_ptr.hpp
pip uninstall -y gsplat
pip install -v ./gsplat
python -c "import gsplat.cuda._wrapper as w; print(hasattr(w, '_C'))"
```

Expected final check:

```text
True
```

---

## 5. Recommended defaults after the post-run fixes

### 5.1 Real-data preprocessing defaults

- depth model: `v2`
- dynamic track generation: `--grid_size 64 --max_hw 512`
- static track generation: `--grid_size 24 --max_hw 512`
- allocator tweak for track jobs:
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 5.2 Training defaults and guidance

- for static-core, short runs such as `2500` iterations are often not enough
- the main spline stage starts around iteration `2000`
- this means a `2500`-iteration run gives only about `500` iterations of actual spline optimization
- a more realistic range is:
  - `5000-10000` for early usable previews
  - `15000-30000` for better stabilization and scene quality

### 5.3 Dynamic-mask guidance

- `person .` works as a minimal prompt, but crowded scenes may need broader prompts
- imperfect masks can contribute to blurry or unstable results

### 5.4 Legacy dynamic temporal filtering

- recommended default: `0.3`
- increase for stricter temporal support
- decrease for more permissive support when masks flicker

---

## 6. Known practical symptoms and how to interpret them

### 6.1 `Reading Nvidia Info` repeating during training

Usually expected in chunked static-core training, because each chunk recreates the scene.

### 6.2 No progress bar in static-core training

Expected in chunked mode. The non-chunked path uses `tqdm`; the chunked path may only show chunk-level messages.

### 6.3 Render output is blank or shows no useful scene

Check:

- wrong `--source_path`
- degenerate point cloud from an older run trained before the dataset-reader fix
- invalid camera trajectory or pose convention mismatch
- `gsplat` CUDA environment problems

### 6.4 Render output looks blurry

Likely contributors:

- overly strong stabilization losses
- poor dynamic masks
- undertraining

Useful mitigation ideas that came out of this work:

- reduce `w_jitter`
- reduce `w_fov`
- improve prompt coverage for mask generation
- compare masked vs unmasked runs
- train longer

### 6.5 Output video looks like a frozen snapshot

Often means the learned trajectory barely moved, not necessarily that rendering failed.

Check the translation statistics of `train_cam_c2w_spline.npy`.

### 6.6 CUDA assert appears at a misleading line

Remember:

- CUDA errors are asynchronous
- the Python line that throws may not be the line that launched the bad kernel
- for debugging, use:

```bash
CUDA_LAUNCH_BLOCKING=1
```

---

## 7. Files most affected by this post-run debugging pass

- `render_stabilized.py`
- `render_stabilized_video.py`
- `prepare_dataset.py`
- `gen_depth.py`
- `gen_tracks.py`
- `scene/dataset_readers.py`
- `scene/camera_spline.py`
- `train_static_core.py`
- `train.py`
- `arguments/__init__.py`
- `profile-2.4.py`

---

## 8. Suggested rerun order

If reproducing the real-data workflow from scratch after these fixes, the clean order is:

1. prepare the scene
2. generate dynamic masks
3. generate depth if not already present
4. generate tracks if using the legacy dynamic path
5. train
6. render

For static-core specifically, the practical recommendation after this debugging pass is:

- verify the scene has real depth and masks
- retrain from a fresh output directory if an older run was trained against dummy or degenerate scene assets
- render with an explicit correct `--source_path`

---

## 9. Bottom line

The full pipeline run on real data was valuable because it exposed the difference between a pipeline that is merely structurally runnable and a pipeline that is actually correct, robust, and practical on real scenes.

The most important lessons were:

- real-data verification matters
- wrong source paths can invalidate otherwise good checkpoints
- placeholder scene assets must be replaced before trusting training results
- track generation and stability losses needed significant practical optimization
- rendering correctness depends on both pose validity and a working `gsplat` CUDA environment

This README should be the main reference for the post-Phase-3 fixes and rerun procedure.
