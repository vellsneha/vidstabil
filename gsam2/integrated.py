"""
STEP 3.1 — In-process Grounded SAM 2 mask generation.

Uses the same stack as IDEA-Research/Grounded-SAM-2 `grounded_sam2_hf_model_demo.py`:
HuggingFace Grounding DINO + SAM 2 image predictor (local SAM2 checkpoint).

Upstream repo (clone into third_party/):
  https://github.com/IDEA-Research/Grounded-SAM-2

Setup:
  1. git clone the repo to ``<project>/third_party/Grounded-SAM-2`` (or set VIDSTABIL_GSAM2_ROOT).
  2. Download SAM 2 checkpoints: ``cd third_party/Grounded-SAM-2/checkpoints && bash download_ckpts.sh``
  3. pip install -r requirements-gsam2.txt (project root or vidstabil/).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    """Parent of the inner ``vidstabil`` Python package (contains ``third_party/``)."""
    return Path(__file__).resolve().parent.parent.parent


def default_gsam2_root() -> Path:
    env = os.environ.get("VIDSTABIL_GSAM2_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return _project_root() / "third_party" / "Grounded-SAM-2"


def ensure_gsam2_on_path(gsam2_root: Path) -> None:
    if not gsam2_root.is_dir():
        raise FileNotFoundError(
            f"Grounded-SAM-2 not found at {gsam2_root}.\n"
            "Clone: git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "
            f'"{gsam2_root}"'
        )
    r = str(gsam2_root)
    if r not in sys.path:
        sys.path.insert(0, r)


def normalize_text_prompt(text: str) -> str:
    """HF Grounding DINO expects lowercased phrases ending with a dot (per upstream demo)."""
    t = text.strip().lower()
    if not t.endswith("."):
        t = t + "."
    return t


def _list_frames(scene: str) -> list[str]:
    d = os.path.join(scene, "images_2")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Expected images_2 under scene: {d}")
    names = sorted(
        f for f in os.listdir(d) if f.lower().endswith(".png") and f[:3].isdigit()
    )
    if not names:
        raise FileNotFoundError(f"No 000.png-style frames in {d}")
    return [os.path.join(d, f) for f in names]


def run_integrated_masks(
    scene: str,
    out_subdir: str = "dynamic_masks",
    text_prompt: str = "person . car .",
    grounding_model: str = "IDEA-Research/grounding-dino-tiny",
    sam2_checkpoint_rel: str = "checkpoints/sam2.1_hiera_large.pt",
    sam2_config_rel: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    force_cpu: bool = False,
    gsam2_root: Path | None = None,
) -> int:
    """
    For each frame in ``images_2/``, run Grounding DINO -> SAM2 and write a binary union mask
    (255 = dynamic / foreground objects matching the text prompt).

    Returns number of frames processed.
    """
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    root = gsam2_root if gsam2_root is not None else default_gsam2_root()
    ensure_gsam2_on_path(root)

    ckpt = (root / sam2_checkpoint_rel).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"SAM 2 checkpoint missing: {ckpt}\n"
            f"Run: cd {root / 'checkpoints'} && bash download_ckpts.sh"
        )

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(
        sam2_config_rel,
        str(ckpt),
        device=device,
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(grounding_model)
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(device)

    text = normalize_text_prompt(text_prompt)
    paths = _list_frames(scene)
    out_root = os.path.join(scene, out_subdir)
    os.makedirs(out_root, exist_ok=True)

    def _one_frame(img_path: str, t_idx: int) -> None:
        image = Image.open(img_path).convert("RGB")
        arr = np.array(image)
        sam2_predictor.set_image(arr)

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding(**inputs)

        det = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        input_boxes = det[0]["boxes"].cpu().numpy()
        h, w = arr.shape[0], arr.shape[1]

        if input_boxes.shape[0] == 0:
            union = np.zeros((h, w), dtype=np.uint8)
        else:
            masks, _scores, _logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            union = (masks.any(axis=0)).astype(np.uint8) * 255

        out_png = os.path.join(out_root, f"{t_idx:03d}.png")
        Image.fromarray(union).convert("L").save(out_png)

    t0 = time.perf_counter()
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_idx, img_path in enumerate(tqdm(paths, desc="GSAM2 masks")):
                _one_frame(img_path, t_idx)
    else:
        for t_idx, img_path in enumerate(tqdm(paths, desc="GSAM2 masks")):
            _one_frame(img_path, t_idx)

    elapsed = time.perf_counter() - t0
    n = len(paths)
    print(f"[STEP3.1] GSAM2 integrated: {n} frames in {elapsed:.1f}s ({elapsed / max(n, 1) * 1000:.0f} ms/frame)")
    return n
