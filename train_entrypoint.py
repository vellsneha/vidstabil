#!/usr/bin/env python3
"""
Isolation entrypoint:
- Default path: static-core (`train_static_core.py`)
- Legacy dynamic path: original SplineGS (`train.py`) only when explicitly requested
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Training entrypoint with legacy isolation.")
    parser.add_argument(
        "--legacy-dynamic",
        action="store_true",
        help="Run legacy dynamic spline training path (original train.py).",
    )
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip NVIDIA/CUDA checks (not recommended; training will likely fail without GPU).",
    )
    args, passthrough = parser.parse_known_args()

    if not args.skip_gpu_preflight:
        from preflight_gpu import check_gpu_requirements

        ok, lines = check_gpu_requirements()
        for line in lines:
            print(line)
        if not ok:
            print("\n[preflight] Aborting. Fix GPU setup or pass --skip-gpu-preflight (at your own risk).")
            raise SystemExit(1)
        print("[preflight] GPU checks passed.\n")

    root = Path(__file__).resolve().parent
    target = root / ("train.py" if args.legacy_dynamic else "train_static_core.py")

    mode = "legacy dynamic" if args.legacy_dynamic else "static core"
    print(f"[entrypoint] Mode: {mode}")
    print(f"[entrypoint] Dispatching to: {target.name}")

    cmd = [sys.executable, str(target), *passthrough]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
