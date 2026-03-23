#!/usr/bin/env python3
"""
Verification script for the static-core SplineGS refactor.

Checks:
1) 3DGS scene representation usage exists.
2) COLMAP-free pose initialization is used.
3) Differentiable renderer path uses render_static.
4) Basic photometric loss (L1 +/- DSSIM) exists.
5) Active static-core training path avoids spline dynamics calls.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train_static_core.py"
README_FILE = ROOT / "README_STATIC_CORE.md"
ENTRYPOINT_FILE = ROOT / "train_entrypoint.py"
PREFLIGHT_FILE = ROOT / "preflight_gpu.py"


def require(condition, ok_msg, fail_msg):
    if condition:
        print(f"[OK] {ok_msg}")
        return True
    print(f"[FAIL] {fail_msg}")
    return False


def main():
    passed = True

    passed &= require(TRAIN_FILE.exists(), "Found train_static_core.py", "Missing train_static_core.py")
    passed &= require(README_FILE.exists(), "Found README_STATIC_CORE.md", "Missing README_STATIC_CORE.md")
    passed &= require(ENTRYPOINT_FILE.exists(), "Found train_entrypoint.py", "Missing train_entrypoint.py")
    passed &= require(PREFLIGHT_FILE.exists(), "Found preflight_gpu.py", "Missing preflight_gpu.py")
    if not passed:
        sys.exit(1)

    code = TRAIN_FILE.read_text(encoding="utf-8")

    # 1) 3DGS representation
    passed &= require(
        "GaussianModel" in code and "Scene(" in code,
        "Uses GaussianModel + Scene for 3DGS representation",
        "3DGS representation (GaussianModel/Scene) not found",
    )

    # 2) COLMAP-free pose initialization
    passed &= require(
        "create_pose_network" in code and "pose_holder._posenet" in code,
        "Uses pose network initialization in static-core training",
        "Pose initialization path not found",
    )

    # 3) Differentiable renderer
    passed &= require(
        re.search(r"\brender_static\s*\(", code) is not None,
        "Uses render_static differentiable renderer",
        "render_static call missing from active training path",
    )

    # 4) Basic photometric loss
    has_l1 = re.search(r"\bl1_loss\s*\(", code) is not None
    has_ssim_term = "lambda_dssim" in code and "ssim(" in code
    passed &= require(
        has_l1 and has_ssim_term,
        "Uses basic photometric loss (L1 + optional DSSIM)",
        "Photometric loss pattern not found",
    )

    # 5) No spline-dynamics calls in active path
    forbidden = [
        "interpolate_cubic_hermite(",
        "create_from_pcd_dynamic(",
        "onedown_control_pts(",
        "render(",
    ]
    # "render(" check should ignore render_static
    has_forbidden_render = re.search(r"(?<!_)render\s*\(", code) is not None
    has_forbidden = any(tok in code for tok in forbidden[:-1]) or has_forbidden_render
    passed &= require(
        not has_forbidden,
        "No spline trajectory dynamics calls in static-core training path",
        "Found spline/dynamic calls in train_static_core.py",
    )

    # 6) Isolation gate: default must dispatch to static-core, legacy only behind explicit flag.
    entry_code = ENTRYPOINT_FILE.read_text(encoding="utf-8")
    has_gate = "--legacy-dynamic" in entry_code and '("train.py" if args.legacy_dynamic else "train_static_core.py")' in entry_code
    passed &= require(
        has_gate,
        "Legacy dynamics are isolated behind --legacy-dynamic gate",
        "Isolation gate missing or invalid in train_entrypoint.py",
    )

    preflight_code = PREFLIGHT_FILE.read_text(encoding="utf-8")
    passed &= require(
        "check_gpu_requirements" in preflight_code and "torch.cuda.is_available" in preflight_code,
        "preflight_gpu.py implements CUDA availability check",
        "preflight_gpu.py missing expected checks",
    )
    passed &= require(
        "--skip-gpu-preflight" in entry_code and "check_gpu_requirements" in entry_code,
        "train_entrypoint.py runs GPU preflight unless --skip-gpu-preflight",
        "train_entrypoint.py missing preflight integration",
    )

    if passed:
        print("\nAll static-core checks passed.")
        sys.exit(0)
    print("\nOne or more checks failed.")
    sys.exit(2)


if __name__ == "__main__":
    main()
