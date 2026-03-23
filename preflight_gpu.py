#!/usr/bin/env python3
"""
GPU preflight for SplineGS / static-core training.

This project expects an NVIDIA GPU with CUDA-enabled PyTorch and CUDA extensions
(simple-knn, gsplat). CPU-only or Apple Silicon (MPS) are not supported without
major code changes.

Usage:
  python preflight_gpu.py              # exit 0 if OK, 1 if not
  python preflight_gpu.py --json       # machine-readable summary on stdout
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def _nvidia_smi_summary() -> tuple[bool, str]:
    """Best-effort: check driver/GPU via nvidia-smi (often missing on Mac / no GPU)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found on PATH (normal on Mac / systems without NVIDIA driver)."
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return False, f"nvidia-smi failed: {r.stderr.strip() or r.stdout.strip()}"
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return False, "nvidia-smi returned no GPU lines."
        return True, line[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return False, f"Could not run nvidia-smi: {e}"


def check_gpu_requirements() -> tuple[bool, list[str]]:
    """
    Returns (ok, lines) where lines are human-readable status messages.
    """
    lines: list[str] = []

    try:
        import torch
    except ImportError as e:
        lines.append(f"[FAIL] PyTorch not importable: {e}")
        lines.append("  Install PyTorch with CUDA from https://pytorch.org/get-started/locally/")
        return False, lines

    lines.append(f"[INFO] PyTorch version: {torch.__version__}")
    cuda_built = getattr(torch.version, "cuda", None)
    lines.append(f"[INFO] PyTorch built with CUDA: {cuda_built or 'None'}")

    if not torch.cuda.is_available():
        lines.append("[FAIL] torch.cuda.is_available() is False.")
        lines.append("  You need an NVIDIA GPU + CUDA drivers + CUDA-enabled PyTorch.")
        ok_nv, nv_msg = _nvidia_smi_summary()
        lines.append(f"[INFO] nvidia-smi: {'OK — ' + nv_msg if ok_nv else nv_msg}")
        lines.extend(_what_to_do_without_gpu())
        return False, lines

    try:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        lines.append(f"[OK] CUDA available: {n} device(s); device 0: {name}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"[FAIL] CUDA reported available but device query failed: {e}")
        lines.extend(_what_to_do_without_gpu())
        return False, lines

    ok_nv, nv_msg = _nvidia_smi_summary()
    if ok_nv:
        lines.append(f"[OK] nvidia-smi: {nv_msg}")
    else:
        lines.append(f"[WARN] {nv_msg}")

    # Optional: gsplat (rasterizer) — training will fail later if missing
    try:
        import gsplat  # noqa: F401

        lines.append("[OK] gsplat import succeeded.")
    except ImportError as e:
        lines.append(f"[WARN] gsplat not importable yet: {e}")
        lines.append("  Install/build gsplat per project README before training.")

    return True, lines


def _what_to_do_without_gpu() -> list[str]:
    return [
        "",
        "What to do if you don't have a local NVIDIA GPU:",
        "  1) Cloud GPU — rent an NVIDIA instance (e.g. Lambda Labs, RunPod, vast.ai,",
        "     Google Colab Pro, AWS g4dn/g5, Azure NC-series). Clone this repo there,",
        "     install deps, run: python train_entrypoint.py ...",
        "  2) Remote dev — SSH into a Linux box with CUDA; use rsync/scp for data.",
        "  3) Docker — use an NVIDIA CUDA base image + nvidia-container-toolkit on the host.",
        "",
        "This codebase is not runnable on CPU or Apple MPS without replacing CUDA kernels",
        "and rewriting device placement (.cuda() → device-agnostic).",
        "",
    ]


def run_preflight_or_exit(*, json_output: bool = False) -> None:
    ok, lines = check_gpu_requirements()
    if json_output:
        payload = {
            "ok": ok,
            "messages": lines,
        }
        print(json.dumps(payload, indent=2))
    else:
        for line in lines:
            print(line)
    raise SystemExit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NVIDIA/CUDA readiness for training.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout.")
    args = parser.parse_args()
    run_preflight_or_exit(json_output=args.json)


if __name__ == "__main__":
    main()
