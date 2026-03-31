#!/usr/bin/env python3
"""
Prepare a real VidStabil "scene" directory from extracted frames.

This script used to generate *dummy* depth/masks/tracks under ``/workspace/vidstabil/data``.
It now focuses on building the expected folder layout and (optionally) running the
real preprocessing steps:

- images_2/:   000.png, 001.png, ...
- gt/:         v000_t000.png, v000_t001.png, ...
- uni_depth/:  per-frame .npy depth (optional: generate via gen_depth.py)
- bootscotracker_dynamic/ + bootscotracker_static/: (optional: generate via gen_tracks.py)
- instance_mask/: optional; if absent, training will proceed with empty instance masks
  (see hardened loader in ``scene/dataset_readers.py``).

Typical pipeline:
  1) python prepare_dataset.py --src-frames /path/to/frames --scene /path/to/scene --gen-depth --gen-tracks --motion-masks /path/to/motion_masks
  2) python preprocess_dynamic_masks.py -s /path/to/scene --backend gsam2 --text-prompt "person ."
  3) python train_entrypoint.py -s /path/to/scene --expname my_run --use_dynamic_mask
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _list_pngs(src_frames: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(src_frames, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in: {src_frames}")
    return paths


def _link_or_copy(src: str, dst: str, *, mode: str) -> None:
    if os.path.exists(dst):
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _prepare_frames(src_frames: str, scene: str, *, mode: str) -> int:
    frame_paths = _list_pngs(src_frames)
    _ensure_dir(os.path.join(scene, "images_2"))
    _ensure_dir(os.path.join(scene, "gt"))

    for i, src in enumerate(frame_paths):
        _link_or_copy(src, os.path.join(scene, "images_2", f"{i:03d}.png"), mode=mode)
    for i in range(len(frame_paths)):
        _link_or_copy(
            os.path.join(scene, "images_2", f"{i:03d}.png"),
            os.path.join(scene, "gt", f"v000_t{i:03d}.png"),
            mode=mode,
        )
    return len(frame_paths)


def _copy_motion_masks(motion_masks: str, scene: str, *, mode: str) -> str:
    src = motion_masks
    if not os.path.isdir(src):
        raise FileNotFoundError(f"--motion-masks must be a directory: {src}")
    dst = os.path.join(scene, "motion_masks")
    _ensure_dir(dst)
    masks = sorted(glob.glob(os.path.join(src, "*.png")))
    if not masks:
        raise FileNotFoundError(f"No PNG masks found in: {src}")
    for m in masks:
        _link_or_copy(m, os.path.join(dst, os.path.basename(m)), mode=mode)
    return dst


def _run(cmd: list[str]) -> None:
    print("[prepare_dataset] Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare real dataset scene from frames.")
    p.add_argument("--src-frames", required=True, help="Directory of extracted PNG frames.")
    p.add_argument("--scene", required=True, help="Output scene directory to create/use.")
    p.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to place frames/masks into the scene (default: symlink).",
    )
    p.add_argument("--gen-depth", action="store_true", help="Generate uni_depth/ using gen_depth.py")
    p.add_argument(
        "--depth-model",
        default="v2",
        help="UniDepth model variant passed to gen_depth.py (default: v2).",
    )
    p.add_argument(
        "--depth-type",
        choices=("depth", "disp"),
        default="depth",
        help="Depth output type for gen_depth.py (default: depth).",
    )
    p.add_argument("--gen-tracks", action="store_true", help="Generate bootscotracker_* using gen_tracks.py")
    p.add_argument(
        "--motion-masks",
        default=None,
        help="Directory containing per-frame motion masks PNGs (used for gen_tracks.py).",
    )
    p.add_argument("--grid-size-dynamic", type=int, default=256)
    p.add_argument("--grid-size-static", type=int, default=50)
    args = p.parse_args()

    scene = os.path.abspath(args.scene)
    src_frames = os.path.abspath(args.src_frames)
    _ensure_dir(scene)

    n = _prepare_frames(src_frames, scene, mode=args.mode)
    print(f"[prepare_dataset] Frames prepared: {n} -> {os.path.join(scene, 'images_2')}")

    if args.motion_masks is not None:
        mm_dst = _copy_motion_masks(os.path.abspath(args.motion_masks), scene, mode=args.mode)
        print(f"[prepare_dataset] Motion masks staged at: {mm_dst}")

    if args.gen_depth:
        _ensure_dir(os.path.join(scene, "uni_depth"))
        root = Path(__file__).resolve().parent
        _run(
            [
                sys.executable,
                str(root / "gen_depth.py"),
                "--image_dir",
                os.path.join(scene, "images_2"),
                "--out_dir",
                os.path.join(scene, "uni_depth"),
                "--depth_type",
                args.depth_type,
                "--depth_model",
                args.depth_model,
            ]
        )

    if args.gen_tracks:
        if args.motion_masks is None:
            raise SystemExit("--gen-tracks requires --motion-masks (per-frame PNG masks).")
        root = Path(__file__).resolve().parent
        dyn_out = os.path.join(scene, "bootscotracker_dynamic")
        sta_out = os.path.join(scene, "bootscotracker_static")
        _ensure_dir(dyn_out)
        _ensure_dir(sta_out)
        mask_dir = os.path.join(scene, "motion_masks")
        _run(
            [
                sys.executable,
                str(root / "gen_tracks.py"),
                "--image_dir",
                os.path.join(scene, "images_2"),
                "--mask_dir",
                mask_dir,
                "--out_dir",
                dyn_out,
                "--grid_size",
                str(args.grid_size_dynamic),
            ]
        )
        _run(
            [
                sys.executable,
                str(root / "gen_tracks.py"),
                "--image_dir",
                os.path.join(scene, "images_2"),
                "--mask_dir",
                mask_dir,
                "--out_dir",
                sta_out,
                "--grid_size",
                str(args.grid_size_static),
                "--is_static",
            ]
        )

    print("\n[prepare_dataset] Done.")
    print(f"[prepare_dataset] Scene: {scene}")
    print("\nNext:")
    print(f"  python preprocess_dynamic_masks.py -s {scene} --backend gsam2 --text-prompt \"person .\"")
    print(f"  python train_entrypoint.py -s {scene} --expname my_run")


if __name__ == "__main__":
    main()