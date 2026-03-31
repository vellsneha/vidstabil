# Chat 2

This document summarizes what was investigated and resolved during this chat for the `vidstabil` pipeline.

## Goal

Understand why a run produced by masked training (`output/masked_run`) was not rendering with `render_stabilized.py`, and identify the environment issues that appeared afterward.

## What was checked

We inspected:

- `render_stabilized.py`
- `train_entrypoint.py`
- `preprocess_dynamic_masks.py`
- `scene/dataset_readers.py`
- `output/masked_run/cfg_args`
- `output/masked_run/point_cloud/static_core_final/`

## Findings

### 1. `output/masked_run` itself was structurally valid

The run output contained the expected static-core checkpoint artifacts:

- `cfg_args`
- `point_cloud/static_core_final/point_cloud_static.ply`
- `point_cloud/static_core_final/cam_spline.pth`
- `point_cloud/static_core_final/posenet.pth`

So the render failure was not caused by a missing trained checkpoint inside `output/masked_run`.

### 2. `render_stabilized.py` does not render from checkpoint only

`render_stabilized.py` rebuilds the scene from the original dataset path passed with `-s/--source_path`, then loads the checkpoint from:

```text
output/<expname>
```

So rendering needs both:

- a valid experiment name, such as `masked_run`
- a valid dataset root with the expected frame layout

### 3. The real render failure came from the dataset path

The command used was:

```bash
python render_stabilized.py --expname masked_run -s /workspace/vidstabil/data3/crowd9_scene --output_video /workspace/vidstabil/output/masked_run/stabilized.mp4
```

The traceback showed:

```text
IndexError: list index out of range
```

inside `readNvidiaCameras()` in `scene/dataset_readers.py`.

That function explicitly does:

```python
image_list = sorted(glob.glob(os.path.join(path, "images_2/*.png")))
img_0 = cv2.imread(image_list[0])
```

This means the loader expected:

```text
<source_path>/images_2/*.png
```

but the provided path did not contain that dataset layout, so `image_list` was empty.

### 4. Correct dataset root was under `data/`, not `data3/`

Later context showed the real scene content existed under:

```text
/workspace/vidstabil/data/crowd9_scene/
```

not:

```text
/workspace/vidstabil/data3/crowd9_scene/
```

So the render command needed to use the actual dataset root.

## Recommended render command

```bash
cd /workspace/vidstabil
python render_stabilized.py \
  --expname masked_run \
  -s /workspace/vidstabil/data/crowd9_scene \
  --output_video /workspace/vidstabil/output/masked_run/stabilized.mp4
```

## Additional environment issue found later: `gsplat`

After the dataset-path issue, a separate dependency/build problem appeared while working with `gsplat`.

### Source build failure

The local `gsplat` build failed with missing GLM headers such as:

```text
fatal error: glm/gtc/type_ptr.hpp: No such file or directory
fatal error: glm/gtx/matrix_operation.hpp: No such file or directory
```

Inspection showed:

```text
/workspace/vidstabil/gsplat/gsplat/cuda/csrc/third_party/glm
```

existed but was effectively empty, which strongly suggested missing submodule/vendor contents.

### Pip wheel was not a usable CUDA replacement

Installing:

```bash
pip install gsplat==1.4.0
```

succeeded, but the installed package did not expose the compiled CUDA extension:

```python
import gsplat.cuda._wrapper as w
hasattr(w, "_C")  # False
```

So this pip-installed package was not a drop-in replacement for the local compiled extension required by the project.

## Recommended `gsplat` recovery steps

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

## Summary

Two separate issues were identified in this chat:

1. Rendering failed because the wrong dataset root was passed to `render_stabilized.py`. The loader needed a valid `images_2/*.png` directory under the source scene path.
2. A later `gsplat` environment failure was caused by missing GLM vendor/submodule content, and the fallback pip install did not provide the required compiled CUDA extension.
