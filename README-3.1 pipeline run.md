# Real-Data Workflow Notes

This document records the changes and command sequence used to move this VidStabil setup from a dummy-data workflow to a real-data workflow, including the legacy dynamic track path and rendering support.

## Goal

Replace the old dummy dataset preparation flow under `data/` with a real pipeline that:

- prepares a real scene directory from extracted frames
- generates real depth
- generates real dynamic masks
- optionally generates bootstrapped tracks
- supports training on both:
  - static-core path
  - legacy dynamic path
- supports rendering for both checkpoint types

## Final scene layout

For a real scene like `data/crowd9_scene`, the expected structure is:

```text
data/crowd9_scene/
  images_2/
  gt/
  uni_depth/
  dynamic_masks/
  bootscotracker_dynamic/
  bootscotracker_static/
  normal/
```

Notes:

- `images_2/` contains `000.png`, `001.png`, ...
- `gt/` contains `v000_t000.png`, `v000_t001.png`, ...
- `uni_depth/` contains `000.npy`, `001.npy`, ...
- `dynamic_masks/` contains `000.png`, `001.png`, ...
- `bootscotracker_dynamic/` and `bootscotracker_static/` contain `000_000.npy`, `000_001.npy`, ...

`instance_mask/` is now optional for the static-core path.

## What was changed

### 1. `prepare_dataset.py`

Old behavior:

- hardcoded `/workspace/vidstabil/data/...`
- created dummy `instance_mask/`
- created dummy `uni_depth/`
- created dummy `bootscotracker_*`

New behavior:

- takes real CLI args:
  - `--src-frames`
  - `--scene`
  - `--gen-depth`
  - `--gen-tracks`
  - `--motion-masks`
- creates `images_2/` and `gt/` from real extracted frames
- optionally runs real `gen_depth.py`
- optionally runs real `gen_tracks.py`

### 2. `gen_depth.py`

Problem:

- default `v2old` depth path failed because it required `xformers/NystromAttention`

Fix:

- default depth model changed to `v2`
- error message improved for unsupported `v2old`

### 3. `scene/dataset_readers.py`

Problem:

- loader assumed `instance_mask/` always existed
- loader assumed bootstracker folders were complete if the directory existed
- static-core path should not have been blocked by missing tracks

Fixes:

- missing `instance_mask/` now falls back to a zero mask
- missing `bootscotracker_*` folders are treated as optional
- partially generated `bootscotracker_*` folders are ignored unless the full expected `n x n` file set exists
- null handling added so absent tracks do not trigger `.cpu()` errors

### 4. `gen_tracks.py`

Problems:

- original version was very slow: roughly one CoTracker call per `(query, target)` pair
- high memory usage
- downscaled-mask path initially had a mask-shape bug

Fixes:

- changed from repeated pairwise calls to one full-video call per query frame
- added `--max_hw` for downscaled tracking
- scaled predicted coordinates back to original resolution before saving
- fixed segmentation-mask shape passed to CoTracker

### 5. `train.py` legacy dynamic path

Problems:

- used `viewpoint.mask` from legacy instance-mask flow instead of real `dynamic_masks/`
- dynamic point init could fail when no points satisfied the strict mask criteria
- dynamic track gather had a hardcoded `12` frame assumption
- temporal dynamic filtering required points to stay valid in all frames, which was too strict for real masks
- empty dynamic selection could lead to zero-point dynamic initialization and CUDA failures

Fixes:

- added `_motion_mask_for_view(...)` helper
- legacy dynamic path now prefers `dynamic_mask_t` when `--use_dynamic_mask` is enabled
- dynamic seed selection now has guarded fallback behavior instead of crashing on empty sets
- replaced hardcoded `12` with actual track length from `target_tracks`
- replaced all-frames temporal mask intersection with a configurable consistency threshold:
  - `dynamic_mask_consistency_ratio`
  - default `0.3`
- if thresholded filtering still removes everything, it falls back to the pre-filtered dynamic set instead of creating zero dynamic gaussians

### 6. `arguments/__init__.py`

Added:

- `dynamic_mask_consistency_ratio = 0.3`

This is used by the legacy dynamic path for temporal mask support filtering.

### 7. `render_stabilized.py`

Old behavior:

- only supported static-core checkpoints
- expected:
  - `point_cloud/static_core_final/point_cloud_static.ply`
  - `cam_spline.pth`
  - `posenet.pth`

New behavior:

- auto-detects checkpoint type
- supports:
  - static-core rendering
  - legacy dynamic rendering
- legacy dynamic renderer:
  - loads latest `point_cloud/iteration_<n>/`
  - loads dynamic/static gaussians
  - loads legacy `posenet.pth`
  - renders via `gaussian_renderer.render(...)`
- fixed iteration directory discovery so folders like `fine_best` do not break checkpoint selection

## Real commands used

### Prepare the scene

Example used here:

```bash
cd /workspace/vidstabil

python prepare_dataset.py \
  --src-frames "/workspace/vidstabil/data/test_clip/images" \
  --scene "/workspace/vidstabil/data/crowd9_scene" \
  --gen-depth \
  --depth-model v2
```

### Generate dynamic masks

```bash
python preprocess_dynamic_masks.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --backend gsam2 \
  --text-prompt "person ."
```

### Generate tracks

Recommended faster commands:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python gen_tracks.py \
  --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" \
  --mask_dir "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" \
  --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_dynamic" \
  --grid_size 64 \
  --max_hw 512
```

```bash
python gen_tracks.py \
  --image_dir "/workspace/vidstabil/data/crowd9_scene/images_2" \
  --mask_dir "/workspace/vidstabil/data/crowd9_scene/dynamic_masks" \
  --out_dir "/workspace/vidstabil/data/crowd9_scene/bootscotracker_static" \
  --is_static \
  --grid_size 24 \
  --max_hw 512
```

### Train: static-core path

This path uses dynamic masks but does not require bootstracker tracks.

```bash
python train_entrypoint.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run \
  --use_dynamic_mask \
  --iterations 10000
```

### Train: legacy dynamic path

This path is the one that uses the generated tracks.

```bash
python train_entrypoint.py \
  --legacy-dynamic \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --use_dynamic_mask \
  --iterations 10000
```

Important:

- the legacy dynamic path has internal stages
- seeing a progress bar ending at `1000` does not mean `--iterations 10000` was ignored
- stage order is:
  - warm: `coarse_iterations`
  - fine_static: `coarse_iterations`
  - fine: `iterations`

### Render: static-core checkpoint

```bash
python render_stabilized.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run \
  --output_video "/workspace/vidstabil/output/my_run/stabilized.mp4"
```

### Render: legacy dynamic checkpoint

```bash
python render_stabilized.py \
  -s "/workspace/vidstabil/data/crowd9_scene" \
  --expname my_run_tracks \
  --output_video "/workspace/vidstabil/output/my_run_tracks/stabilized.mp4"
```

## Why bootstracker generation was slow

Original issue:

- one expensive tracking call per `(query frame, target frame)` pair
- effectively quadratic in number of frames

Improvement:

- one full-video call per query frame
- optional downscaling via `--max_hw`

This significantly reduced runtime and memory pressure.

## Why dynamic initialization failed on real data

The root cause was not “no motion in the scene”.

The real issues were:

- legacy dynamic code read the wrong mask source
- temporal filtering required a point to survive in the dynamic mask for all frames
- real masks flicker and are not perfectly consistent

The correct fix that was applied:

- use `dynamic_masks/` through `dynamic_mask_t`
- soften temporal consistency to a ratio-based threshold

## Current practical defaults

Recommended for real data:

- depth model: `v2`
- track generation:
  - dynamic: `--grid_size 64 --max_hw 512`
  - static: `--grid_size 24 --max_hw 512`
- legacy dynamic temporal support:
  - `--dynamic_mask_consistency_ratio 0.3`

If you want stricter temporal filtering:

```bash
--dynamic_mask_consistency_ratio 0.5
```

If you want more permissive temporal filtering:

```bash
--dynamic_mask_consistency_ratio 0.2
```

## Files changed during this migration

- `prepare_dataset.py`
- `gen_depth.py`
- `gen_tracks.py`
- `scene/dataset_readers.py`
- `arguments/__init__.py`
- `train.py`
- `render_stabilized.py`
- `README-3.1.md`

## Summary

This repo is now set up to:

- prepare real scenes without dummy depth or dummy tracks
- train static-core directly on real dynamic masks
- optionally train the legacy dynamic path using real bootstracker tracks
- render both static-core and legacy-dynamic experiment outputs

If reproducing this from scratch, follow:

1. prepare scene
2. preprocess dynamic masks
3. optionally generate tracks
4. train
5. render
