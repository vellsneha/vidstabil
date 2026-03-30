#!/usr/bin/env python3
"""
STEP 3.1 — Precompute per-frame binary masks M_t (1 = moving / dynamic object, 0 = background).

Masks are written to <scene>/<out_subdir>/{t:03d}.png (single-channel PNG, 0 or 255).
Training loads them when --use_dynamic_mask is set (see README-3.1.md).

Backends:
  gsam2     — Integrated Grounded SAM 2 (HuggingFace Grounding DINO + SAM 2) from
              IDEA-Research/Grounded-SAM-2 in third_party/ (default).
  synthetic — No ML deps; moving ellipse for layout / CI tests only.

Run from the inner ``vidstabil/`` package directory:
    python preprocess_dynamic_masks.py -s /path/to/scene --backend gsam2
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow ``import gsam2`` when launching as ``python preprocess_dynamic_masks.py`` from this dir.
_pkg = os.path.dirname(os.path.abspath(__file__))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)


def _frames_dir(scene: str) -> str:
    p = os.path.join(scene, "images_2")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Expected images_2 under scene: {p}")
    return p


def _list_frames(scene: str) -> list[str]:
    d = _frames_dir(scene)
    names = sorted(
        f for f in os.listdir(d) if f.lower().endswith(".png") and f[:3].isdigit()
    )
    if not names:
        raise FileNotFoundError(f"No 000.png-style frames in {d}")
    return [os.path.join(d, f) for f in names]


def write_synthetic_masks(scene: str, out_subdir: str) -> int:
    """Create a simple moving elliptical foreground mask per frame (M_t = 1 inside blob)."""
    import numpy as np
    from PIL import Image

    paths = _list_frames(scene)
    out_root = os.path.join(scene, out_subdir)
    os.makedirs(out_root, exist_ok=True)

    im0 = Image.open(paths[0]).convert("RGB")
    w, h = im0.size
    cx0, cy0 = w * 0.35, h * 0.45
    rx, ry = w * 0.12, h * 0.18

    for t, src in enumerate(paths):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = cx0 + (t % max(len(paths), 1)) * (w * 0.008)
        cy = cy0 + 0.5 * np.sin(t * 0.35) * (h * 0.05)
        inside = ((xx - cx) ** 2 / (rx**2) + (yy - cy) ** 2 / (ry**2)) <= 1.0
        mask_u8 = (inside.astype(np.uint8) * 255)
        out_path = os.path.join(out_root, f"{t:03d}.png")
        Image.fromarray(mask_u8).convert("L").save(out_path)

    return len(paths)


def main():
    p = argparse.ArgumentParser(description="STEP 3.1 — cache dynamic masks M_t for masked L_photo")
    p.add_argument("-s", "--scene", required=True, help="Dataset root (contains images_2/)")
    p.add_argument(
        "--out-subdir",
        default="dynamic_masks",
        help="Subfolder under scene for cached masks (default: dynamic_masks)",
    )
    p.add_argument(
        "--backend",
        choices=("gsam2", "synthetic"),
        default="gsam2",
        help="gsam2: integrated Grounded SAM 2 (requires third_party clone + checkpoints); "
        "synthetic: test masks without ML",
    )
    p.add_argument(
        "--text-prompt",
        default="person . car .",
        help="Grounding DINO text prompt (classes separated by ' . ', ending with a dot)",
    )
    p.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace id for Grounding DINO (default: tiny for speed)",
    )
    p.add_argument(
        "--sam2-checkpoint",
        default="checkpoints/sam2.1_hiera_large.pt",
        help="Path relative to Grounded-SAM-2 root for SAM 2 weights",
    )
    p.add_argument(
        "--sam2-model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Hydra config name relative to SAM 2 package (see upstream demo)",
    )
    p.add_argument("--box-threshold", type=float, default=0.4)
    p.add_argument("--text-threshold", type=float, default=0.3)
    p.add_argument("--force-cpu", action="store_true", help="Run on CPU (slow)")
    args = p.parse_args()

    if args.backend == "synthetic":
        n = write_synthetic_masks(args.scene, args.out_subdir)
    else:
        from gsam2.integrated import run_integrated_masks

        n = run_integrated_masks(
            args.scene,
            out_subdir=args.out_subdir,
            text_prompt=args.text_prompt,
            grounding_model=args.grounding_model,
            sam2_checkpoint_rel=args.sam2_checkpoint,
            sam2_config_rel=args.sam2_model_config,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            force_cpu=args.force_cpu,
        )

    print(f"[STEP3.1] Wrote {n} masks to {os.path.join(args.scene, args.out_subdir)}")


if __name__ == "__main__":
    main()
