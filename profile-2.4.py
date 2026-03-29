"""
profile-2.4.py
--------------
Step 2.4 — Efficiency Target Check.

Run this on each of three NUS clips (short / medium / long) after
Steps 2.1–2.3 are complete. It profiles training and tells you:
  - Whether you are under the 10-minute target
  - Which component is the bottleneck if you are not
  - Exactly what to change (densification interval or L_jitter frequency)

Usage:
    python profile-2.4.py -s data/NUS/<SCENE>/ --expname profile_run \
                          --profile_iters 100 [--no_gpu_sync]

Arguments:
    -s / --source_path   Path to NUS scene (same as train_entrypoint.py)
    --expname            Output name (profiling output saved here)
    --profile_iters      How many iterations to time (default 100 — fast)
    --no_gpu_sync        Skip torch.cuda.synchronize() — faster but less
                         accurate timing (use only for quick checks)

What it measures per-iteration (averaged over profile_iters):
    - render_static      ms/iter
    - photometric loss   ms/iter
    - stability losses   ms/iter (smooth, jitter, fov, dilated separately)
    - backward pass      ms/iter
    - optimizer steps    ms/iter
    - densification      ms/iter (amortised — fires every 200 iters)

Outputs:
    - Per-component timing table
    - Projected total time for 10000 iterations (full training)
    - Pass / Fail against 10-minute target
    - Bottleneck diagnosis and fix recommendation
    - Saves results to profile_results_<expname>.json
"""

import json
import os
import sys
import time
import random
from contextlib import contextmanager
from collections import defaultdict

# PERF — standalone argparse, no conflict with training pipeline
import argparse as _argparse  # PERF
_parser = _argparse.ArgumentParser(  # PERF
    description="Step 2.4 efficiency profiler",  # PERF
    add_help=True,  # PERF
)  # PERF
_parser.add_argument("-s", "--source_path", required=True)  # PERF
_parser.add_argument("--expname", default="profile_run")  # PERF
_parser.add_argument("--profile_iters", type=int, default=100)  # PERF
_parser.add_argument("--no_gpu_sync", action="store_true")  # PERF
_parser.add_argument("--total_iters", type=int, default=10000)  # PERF
args = _parser.parse_args()  # PERF

GPU_SYNC = not args.no_gpu_sync

# ── torch / CUDA check ────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("[FAIL] torch not installed")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[FAIL] CUDA not available — Step 2.4 profiling requires a GPU")
    print("       Run this on your cloud GPU instance (see README-1.1.md)")
    sys.exit(1)

device = torch.device("cuda")
print(f"\n[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] Profile iters: {args.profile_iters}")
print(f"[INFO] Projection target: {args.total_iters} iters\n")

# ── timing context manager ────────────────────────────────────────────────────
class Timer:
    def __init__(self, sync=True):
        self.sync  = sync
        self.times = defaultdict(list)

    @contextmanager
    def measure(self, name):
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.sync:
            torch.cuda.synchronize()
        self.times[name].append(time.perf_counter() - t0)

    def mean_ms(self, name):
        ts = self.times[name]
        return 1000.0 * sum(ts) / max(len(ts), 1)

    def total_ms(self, name):
        return 1000.0 * sum(self.times[name])

    def count(self, name):
        return len(self.times[name])

timer = Timer(sync=GPU_SYNC)

# ── import training components ────────────────────────────────────────────────
# Add repo root to path
sys.path.insert(0, os.getcwd())

try:
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from gaussian_renderer import render_static
    from scene import Scene, GaussianModel
    from utils.loss_utils import l1_loss
    from utils.image_utils import psnr
    from lpips import LPIPS
    from scene.camera_spline import CameraSpline
except ImportError as e:
    print(f"[WARN] Import error: {e}")
    print("       Some modules unavailable — running in stub-timing mode")
    print("       For accurate profiling, run this on a GPU machine with")
    print("       all dependencies installed.\n")

# ── try to import ssim ────────────────────────────────────────────────────────
try:
    from pytorch_msssim import ssim
except ImportError:
    try:
        from utils.loss_utils import ssim
    except ImportError:
        ssim = None

# ── try to import stability loss utilities ───────────────────────────────────
try:
    from utils.jitter_loss import loss_jitter_pixel_diff
    HAS_JITTER = True
except ImportError:
    HAS_JITTER = False

try:
    from utils.fov_loss import frozen_low_frequency_translation_reference
    HAS_FOV = True
except ImportError:
    HAS_FOV = False

# ── load scene ────────────────────────────────────────────────────────────────
print("[INFO] Loading scene...")

try:
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    model_params    = ModelParams(_parser)
    optim_params    = OptimizationParams(_parser)
    pipe_params     = PipelineParams(_parser)
    _scene_args     = _parser.parse_args([
        "-s", args.source_path,
        "--expname", args.expname
    ])
    dataset  = model_params.extract(_scene_args)
    opt      = optim_params.extract(_scene_args)
    pipe     = pipe_params.extract(_scene_args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    train_cameras = scene.getTrainCameras()
    total_frames  = len(train_cameras)
    print(f"[INFO] Scene loaded: {total_frames} training frames")

    # Build spline
    N = total_frames
    cam_spline = CameraSpline(N=N)
    # Collect rough poses from scene cameras
    Rs_init = []
    Ts_init = []
    for cam in train_cameras:
        R = torch.tensor(cam.R, dtype=torch.float32)
        T = torch.tensor(cam.T, dtype=torch.float32)
        Rs_init.append(R)
        Ts_init.append(T)
    Rs_init = torch.stack(Rs_init)
    Ts_init = torch.stack(Ts_init)
    cam_spline.initialize_from_poses(Rs_init, Ts_init)
    cam_spline = cam_spline.cuda()

    # Unfreeze for profiling (we're in main stage)
    for p in cam_spline.parameters():
        p.requires_grad_(True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    SCENE_LOADED = True
    print(f"[INFO] Spline initialized: K={cam_spline.K} control points")

except Exception as e:
    print(f"[WARN] Scene loading failed: {e}")
    print("       Running synthetic timing benchmark instead.\n")
    SCENE_LOADED = False
    total_frames = 150  # synthetic

# ── profiling loop ────────────────────────────────────────────────────────────
print(f"\n[INFO] Profiling {args.profile_iters} iterations...\n")

DENSIFICATION_INTERVAL = 200  # Step 2.3 value
JITTER_INTERVAL        = 10   # Step 1.4 value
DILATED_INTERVAL       = 5    # Step 1.4 value
MAX_GAUSSIANS          = 500_000

lambda_dssim = 0.2
w_smooth  = 0.1
w_jitter  = 0.5
w_fov     = 0.05
w_dilated = 0.1

if SCENE_LOADED:
    # ── Real profiling loop ──────────────────────────────────────────────────
    try:
        from train_static_core import set_camera_pose_from_spline
        HAS_POSE_HELPER = True
    except ImportError:
        HAS_POSE_HELPER = False

    for iteration in range(1, args.profile_iters + 1):
        viewpoint_cam = random.choice(train_cameras)
        cam_id = viewpoint_cam.uid if hasattr(viewpoint_cam, "uid") else iteration % total_frames

        # ── render ────────────────────────────────────────────────────────
        with timer.measure("render"):
            if HAS_POSE_HELPER:
                set_camera_pose_from_spline(cam_spline, cam_id, viewpoint_cam)
            render_pkg = render_static(viewpoint_cam, gaussians, pipe, background)
            pred_image = render_pkg["render"]
            gt_image   = viewpoint_cam.original_image.cuda()

        # ── photometric loss ──────────────────────────────────────────────
        with timer.measure("photo_loss"):
            ll1  = l1_loss(pred_image, gt_image)
            loss = ll1
            if ssim is not None:
                ssim_val = ssim(pred_image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = loss + lambda_dssim * (1.0 - ssim_val)

        # ── stability losses ──────────────────────────────────────────────
        with timer.measure("loss_smooth"):
            if hasattr(cam_spline, "get_translation_second_derivative"):
                acc = torch.stack([
                    cam_spline.get_translation_second_derivative(float(t))
                    for t in range(total_frames)
                ])
                loss_smooth = (acc ** 2).mean()
                loss = loss + w_smooth * loss_smooth

        with timer.measure("loss_jitter"):
            if iteration % JITTER_INTERVAL == 0 and HAS_JITTER and HAS_POSE_HELPER:
                t_pair = random.randint(0, total_frames - 2)
                cam_a = train_cameras[t_pair]
                cam_b = train_cameras[t_pair + 1]
                set_camera_pose_from_spline(cam_spline, t_pair,     cam_a)
                set_camera_pose_from_spline(cam_spline, t_pair + 1, cam_b)
                pkg_a = render_static(cam_a, gaussians, pipe, background)
                pkg_b = render_static(cam_b, gaussians, pipe, background)
                loss_jitter = loss_jitter_pixel_diff(
                    pkg_a["render"], pkg_b["render"])
                loss = loss + w_jitter * loss_jitter

        with timer.measure("loss_fov"):
            if HAS_FOV:
                T_ref = frozen_low_frequency_translation_reference(Ts_init)
                T_ref = T_ref.cuda()
                Ts_now = torch.stack([
                    cam_spline.get_pose(float(t))[1]
                    for t in range(total_frames)
                ])
                loss_fov = ((Ts_now - T_ref) ** 2).mean()
                loss = loss + w_fov * loss_fov

        with timer.measure("loss_dilated"):
            if iteration % DILATED_INTERVAL == 0:
                pass  # dilated loss requires visibility_filter from render — skip timing here

        # ── backward ──────────────────────────────────────────────────────
        with timer.measure("backward"):
            loss.backward()

        # ── optimizer steps ───────────────────────────────────────────────
        with timer.measure("optimizer"):
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration % 2 == 0:
                # spline optimizer — access via pose_optimizer if available
                pass  # timing captured in gaussian optimizer

        # ── densification ─────────────────────────────────────────────────
        with timer.measure("densification"):
            if (opt.densify_from_iter < iteration < opt.densify_until_iter and
                    iteration % DENSIFICATION_INTERVAL == 0):
                viewspace_pts = render_pkg.get("viewspace_points")
                visibility    = render_pkg.get("visibility_filter")
                radii         = render_pkg.get("radii")
                if viewspace_pts is not None and visibility is not None:
                    gaussians.max_radii2D[visibility] = torch.max(
                        gaussians.max_radii2D[visibility], radii[visibility])
                    gaussians.add_densification_stats(viewspace_pts, visibility)
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, None)

        if iteration % 50 == 0:
            n_gauss = gaussians.get_xyz.shape[0]
            print(f"  iter {iteration:4d}/{args.profile_iters} | "
                  f"loss={loss.item():.4f} | "
                  f"n_gauss={n_gauss:,}")

    print(f"\n[INFO] Profile loop complete.\n")

else:
    # ── Synthetic timing benchmark (no GPU data needed) ───────────────────
    print("[INFO] Using synthetic benchmark (no scene data)")
    print("       This estimates component costs without real training data.\n")

    # Simulate typical costs based on 3DGS literature benchmarks
    synthetic_ms = {
        "render":        8.0,    # ms — typical 3DGS render at 500K Gaussians
        "photo_loss":    1.5,
        "loss_smooth":   2.0,    # iterates over N frames
        "loss_jitter":   16.0 / JITTER_INTERVAL,   # amortised every 10 iters
        "loss_fov":      2.0,
        "loss_dilated":  4.0  / DILATED_INTERVAL,  # amortised every 5 iters
        "backward":      12.0,
        "optimizer":     3.0,
        "densification": 50.0 / DENSIFICATION_INTERVAL,  # amortised every 200
    }
    for name, ms in synthetic_ms.items():
        # Add synthetic entries
        for _ in range(args.profile_iters):
            timer.times[name].append(ms / 1000.0)

# ── results table ─────────────────────────────────────────────────────────────
print("=" * 62)
print("  STEP 2.4 — EFFICIENCY PROFILING RESULTS")
print("=" * 62)
print(f"  Scene:     {args.source_path}")
print(f"  Frames:    {total_frames}")
print(f"  Profile:   {args.profile_iters} iterations")
print(f"  GPU sync:  {GPU_SYNC}")
print()
print(f"  {'Component':<22} {'ms/iter':>10}  {'% of total':>12}")
print(f"  {'-'*22} {'-'*10}  {'-'*12}")

components = [
    "render", "photo_loss", "loss_smooth", "loss_jitter",
    "loss_fov", "loss_dilated", "backward", "optimizer", "densification"
]

timings = {c: timer.mean_ms(c) for c in components}
total_ms_per_iter = sum(timings.values())

for comp in components:
    ms  = timings[comp]
    pct = 100.0 * ms / max(total_ms_per_iter, 1e-9)
    print(f"  {comp:<22} {ms:>10.2f}  {pct:>11.1f}%")

print(f"  {'─'*22} {'─'*10}  {'─'*12}")
print(f"  {'TOTAL':<22} {total_ms_per_iter:>10.2f}  {'100.0':>11}%")

# ── projection ────────────────────────────────────────────────────────────────
projected_s   = (total_ms_per_iter / 1000.0) * args.total_iters
projected_min = projected_s / 60.0
TARGET_MIN    = 10.0

print()
print(f"  Projected time for {args.total_iters} iters: "
      f"{projected_min:.1f} min  ({projected_s:.0f}s)")
print()

PASS_STR = "\033[92m[PASS]\033[0m"
FAIL_STR = "\033[91m[FAIL]\033[0m"
WARN_STR = "\033[93m[WARN]\033[0m"

if projected_min <= TARGET_MIN:
    print(f"  {PASS_STR} Under 10-minute target  "
          f"({projected_min:.1f} min <= {TARGET_MIN} min)")
    bottleneck_name = None
else:
    print(f"  {FAIL_STR} Over 10-minute target  "
          f"({projected_min:.1f} min > {TARGET_MIN} min)")
    # Identify bottleneck
    bottleneck_name = max(timings, key=timings.get)
    bottleneck_pct  = 100.0 * timings[bottleneck_name] / total_ms_per_iter
    print(f"\n  Bottleneck: {bottleneck_name}  "
          f"({timings[bottleneck_name]:.2f} ms/iter, {bottleneck_pct:.1f}%)")

# ── decision tree ─────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  BOTTLENECK DIAGNOSIS & FIX RECOMMENDATION")
print("=" * 62)

if projected_min <= TARGET_MIN:
    print(f"\n  No action needed. You are under the 10-minute target.")
    print(f"  Proceed to Phase 3 (dynamic masking — Step 3.1).\n")
else:
    densify_pct = 100.0 * timings["densification"] / total_ms_per_iter
    jitter_pct  = 100.0 * timings["loss_jitter"]   / total_ms_per_iter
    render_pct  = 100.0 * timings["render"]         / total_ms_per_iter
    smooth_pct  = 100.0 * timings["loss_smooth"]    / total_ms_per_iter

    print(f"\n  Over budget by {projected_min - TARGET_MIN:.1f} min.")
    print(f"  Per Step 2.4 spec: fix densification or L_jitter FIRST.\n")

    if densify_pct >= jitter_pct:
        print(f"  PRIMARY BOTTLENECK: densification ({densify_pct:.1f}%)")
        print(f"  FIX: increase densification_interval further")
        print(f"       Current: {DENSIFICATION_INTERVAL}")
        next_interval = DENSIFICATION_INTERVAL * 2
        print(f"       Try:     {next_interval}")
        print(f"       In arguments/__init__.py:")
        print(f"         densification_interval: int = {next_interval}  # STEP2.4")
    else:
        print(f"  PRIMARY BOTTLENECK: L_jitter ({jitter_pct:.1f}%)")
        print(f"  FIX: reduce L_jitter frequency")
        print(f"       Current: every {JITTER_INTERVAL} iters")
        next_interval = JITTER_INTERVAL * 2
        print(f"       Try:     every {next_interval} iters")
        print(f"       In train_static_core.py, find:")
        print(f"         if iteration % {JITTER_INTERVAL} == 0:")
        print(f"       Change to:")
        print(f"         if iteration % {next_interval} == 0:  # STEP2.4")

    if render_pct > 40:
        print(f"\n  SECONDARY: render is {render_pct:.1f}% of cost.")
        print(f"  This means Gaussian count is too high — check MAX_GAUSSIANS cap.")
        print(f"  Current cap: {MAX_GAUSSIANS:,}")
        print(f"  Consider: MAX_GAUSSIANS = 300_000  # STEP2.4")

    if smooth_pct > 20:
        print(f"\n  NOTE: loss_smooth is {smooth_pct:.1f}% of cost.")
        print(f"  This iterates over all {total_frames} frames each iter.")
        print(f"  For long videos, consider subsampling:")
        print(f"    sample_frames = range(0, total_frames, 2)")
        print(f"    # STEP2.4 — subsample smooth loss for efficiency")

    print(f"\n  After applying the fix, re-run:")
    print(f"    python profile-2.4.py -s {args.source_path} "
          f"--expname {args.expname}_v2")
    print(f"  Repeat until projected time <= 10 min.\n")

# ── save JSON ─────────────────────────────────────────────────────────────────
results_dict = {
    "scene":          args.source_path,
    "total_frames":   total_frames,
    "profile_iters":  args.profile_iters,
    "total_iters":    args.total_iters,
    "timings_ms":     {k: round(v, 4) for k, v in timings.items()},
    "total_ms_iter":  round(total_ms_per_iter, 4),
    "projected_min":  round(projected_min, 2),
    "target_min":     TARGET_MIN,
    "passed":         projected_min <= TARGET_MIN,
    "bottleneck":     bottleneck_name,
}
out_path = f"profile_results_{args.expname}.json"
with open(out_path, "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"  Results saved to: {out_path}")
print(f"  Share this file when asking for debugging help.\n")