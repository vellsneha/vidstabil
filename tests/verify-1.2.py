"""
verify_step1_2.py
-----------------
Run from the root of your SplineGS fork:
    python verify_step1_2.py

Checks every requirement from Step 1.2, accounting for the actual
Step 1.1 architecture:
  - Entry point:  train_exp.py  (not train.py)
  - Pose source:  pose_network via create_pose_network / update_cam
  - Renderer:     gaussian_renderer.render_static
  - Scene:        GaussianModel + stat_gaussians

Checks:
  1.  scene/camera_spline.py exists and is importable
  2.  CameraSpline(N) creates K = N//5 control points
  3.  ctrl_trans [K,3] and ctrl_quats [K,4] are nn.Parameter
  4.  Parameter count is less than pose_network equivalent (N*6)
  5.  get_pose(t) returns R[3,3] and T[3]
  6.  get_pose is fully differentiable (gradients flow, no detach/numpy)
  7.  R output is valid SO(3): R^T R = I, det = +1
  8.  Quaternion output is unit norm
  9.  initialize_from_poses warm-starts from rotation matrices
  10. get_all_poses(N) returns exactly N (R,T) pairs
  11. Optimizer receives and updates spline parameters
  12. Trajectory is smooth — no discontinuous jumps
  13. train_exp.py imports and uses CameraSpline
  14. train_exp.py no longer calls update_cam in training loop
  15. STEP1.2 comments present in both modified files
  16. get_pose contains no numpy calls or .detach() in forward path
"""

import sys, os, ast
import importlib

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"

results = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"{tag} {name}")
    if detail:
        print(f"       {detail}")
    results.append((name, condition))
    return condition

def warn(name, detail=""):
    print(f"{WARN} {name}")
    if detail:
        print(f"       {detail}")

def info(msg):
    print(f"{INFO} {msg}")

# ── 0. torch ──────────────────────────────────────────────────────────────────
try:
    import torch, torch.nn as nn
except ImportError:
    print(f"{FAIL} torch not installed — cannot run any checks")
    sys.exit(1)

sys.path.insert(0, os.getcwd())

print("\n── Checking scene/camera_spline.py ──────────────────────────────────\n")

# ── 1. Importable ─────────────────────────────────────────────────────────────
# Load camera_spline.py directly to avoid triggering scene/__init__.py, which
# pulls in GaussianModel → simple_knn (a CUDA extension unavailable without GPU).
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "camera_spline",
        os.path.join(os.getcwd(), "scene", "camera_spline.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CameraSpline = _mod.CameraSpline
    check("CameraSpline importable from scene/camera_spline.py", True)
except Exception as e:
    check("CameraSpline importable from scene/camera_spline.py", False, str(e))
    print("\nCannot continue — fix the import first.")
    sys.exit(1)

# ── helpers ───────────────────────────────────────────────────────────────────
def random_rotmat():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()

def make_poses(N):
    Rs = torch.stack([random_rotmat() for _ in range(N)])
    Ts = torch.randn(N, 3).float()
    return Rs, Ts

N = 150
K_expected = N // 5  # 30
Rs, Ts = make_poses(N)

# ── 2. K = N//5 ───────────────────────────────────────────────────────────────
spline = CameraSpline(N=N)
K_actual = getattr(spline, "K",
           getattr(spline, "num_control_points",
           getattr(spline, "n_control_points", None)))
if K_actual is None and hasattr(spline, "ctrl_trans"):
    K_actual = spline.ctrl_trans.shape[0]
check("K = N//5 = 30 control points for N=150",
      K_actual == K_expected,
      f"got K={K_actual}, expected {K_expected}")

# ── 3. ctrl_trans [K,3] and ctrl_quats [K,4] are nn.Parameter ────────────────
has_trans = hasattr(spline, "ctrl_trans") and isinstance(spline.ctrl_trans, nn.Parameter)
has_quats = hasattr(spline, "ctrl_quats") and isinstance(spline.ctrl_quats, nn.Parameter)
check("ctrl_trans is nn.Parameter",
      has_trans,
      f"shape: {tuple(spline.ctrl_trans.shape)}" if has_trans else "attribute missing")
check("ctrl_quats is nn.Parameter",
      has_quats,
      f"shape: {tuple(spline.ctrl_quats.shape)}" if has_quats else "attribute missing")

if has_trans and has_quats:
    check("ctrl_trans shape is [K,3]",
          spline.ctrl_trans.shape == torch.Size([K_expected, 3]),
          f"got {tuple(spline.ctrl_trans.shape)}")
    check("ctrl_quats shape is [K,4]",
          spline.ctrl_quats.shape == torch.Size([K_expected, 4]),
          f"got {tuple(spline.ctrl_quats.shape)}")

# ── 4. Parameter count reduction vs pose_network (N*6) ───────────────────────
spline_params = sum(p.numel() for p in spline.parameters())
old_equiv     = N * 6
check("Parameter count < per-frame equivalent (N x 6 = 900)",
      spline_params < old_equiv,
      f"spline={spline_params}, per-frame-equiv={old_equiv} "
      f"({100*spline_params/old_equiv:.0f}% of original)")

# ── 5. get_pose output shapes ─────────────────────────────────────────────────
print("\n── Checking get_pose ────────────────────────────────────────────────\n")
spline.initialize_from_poses(Rs, Ts)
try:
    R_out, T_out = spline.get_pose(0.0)
    shape_ok = (tuple(R_out.shape) == (3,3) and tuple(T_out.shape) == (3,))
    check("get_pose returns R[3,3] and T[3]", shape_ok,
          f"R:{tuple(R_out.shape)}, T:{tuple(T_out.shape)}")
except Exception as e:
    check("get_pose returns R[3,3] and T[3]", False, str(e))

# ── 6. Differentiability ──────────────────────────────────────────────────────
try:
    sp2 = CameraSpline(N=N)
    sp2.initialize_from_poses(Rs, Ts)
    R_t, T_t = sp2.get_pose(float(N // 2))
    (T_t.sum() + R_t.sum()).backward()
    grads = [p for p in sp2.parameters() if p.grad is not None]
    check("Gradients flow through get_pose (differentiable)",
          len(grads) > 0,
          f"{len(grads)}/{len(list(sp2.parameters()))} params received gradients")
except Exception as e:
    check("Gradients flow through get_pose", False, str(e))

# ── 7. Valid SO(3) ────────────────────────────────────────────────────────────
try:
    R_mid, _ = spline.get_pose(float(N // 2))
    Rd = R_mid.detach()
    ident_err = (Rd.T @ Rd - torch.eye(3)).abs().max().item()
    det_val   = torch.det(Rd).item()
    check("R is valid SO(3): R^T R = I and det = +1",
          ident_err < 1e-4 and abs(det_val - 1.0) < 1e-4,
          f"|R^T R - I|_max={ident_err:.2e}, det={det_val:.6f}")
except Exception as e:
    check("R is valid SO(3)", False, str(e))

# ── 8. Quaternion unit norm ───────────────────────────────────────────────────
if has_quats:
    norms = spline.ctrl_quats.data.norm(dim=1)
    unit_ok = (norms - 1.0).abs().max().item() < 1e-4
    check("ctrl_quats are unit quaternions after initialize_from_poses",
          unit_ok,
          f"max |norm - 1| = {(norms-1).abs().max().item():.2e}")

# ── 9. initialize_from_poses warm-start accuracy ──────────────────────────────
try:
    sp3 = CameraSpline(N=N)
    sp3.initialize_from_poses(Rs, Ts)
    R0, T0 = sp3.get_pose(0.0)
    t_err = (T0.detach() - Ts[0]).norm().item()
    r_err = (R0.detach() - Rs[0]).abs().max().item()
    check("initialize_from_poses warm-starts near input poses (t=0)",
          t_err < 0.5 and r_err < 0.5,
          f"T err={t_err:.4f}, R err={r_err:.4f} (warm start, not exact)")
except Exception as e:
    check("initialize_from_poses warm-start", False, str(e))

# ── 10. get_all_poses ─────────────────────────────────────────────────────────
try:
    all_poses = spline.get_all_poses(N)
    count_ok  = len(all_poses) == N
    shape_ok  = all(tuple(r.shape)==(3,3) and tuple(t.shape)==(3,)
                    for r, t in all_poses)
    check("get_all_poses(N) returns N (R,T) pairs with correct shapes",
          count_ok and shape_ok,
          f"count={len(all_poses)}, shapes_ok={shape_ok}")
except Exception as e:
    check("get_all_poses(N)", False, str(e))

# ── 11. Optimizer updates spline ──────────────────────────────────────────────
try:
    sp4 = CameraSpline(N=N)
    sp4.initialize_from_poses(Rs, Ts)
    opt = torch.optim.Adam(sp4.parameters(), lr=1e-3)
    R_t, T_t = sp4.get_pose(float(N // 2))
    T_t.sum().backward()
    before = [p.data.clone() for p in sp4.parameters()]
    opt.step()
    after  = [p.data.clone() for p in sp4.parameters()]
    changed = any((b - a).abs().max() > 0 for b, a in zip(before, after))
    check("optimizer.step() updates spline control points", changed)
except Exception as e:
    check("optimizer.step() updates spline control points", False, str(e))

# ── 12. Smoothness — no hard jumps ────────────────────────────────────────────
try:
    all_poses = spline.get_all_poses(N)
    trans = torch.stack([t.detach() for _, t in all_poses])
    diffs = (trans[1:] - trans[:-1]).norm(dim=1)
    max_j, mean_j = diffs.max().item(), diffs.mean().item()
    check("Trajectory smooth — no discontinuous jumps",
          max_j < 10 * mean_j,
          f"max jump={max_j:.4f}, mean={mean_j:.4f}, "
          f"ratio={max_j/max(mean_j,1e-9):.1f}x")
except Exception as e:
    check("Trajectory smoothness", False, str(e))

# ── 13 & 14. train_exp.py ────────────────────────────────────────────
print("\n── Checking train_exp.py ────────────────────────────────────\n")
TRAIN_FILE = "train_exp.py"
train_src = None
if os.path.exists(TRAIN_FILE):
    with open(TRAIN_FILE) as f:
        train_src = f.read()

    uses_spline = ("CameraSpline" in train_src or "camera_spline" in train_src)
    check(f"{TRAIN_FILE} imports/uses CameraSpline", uses_spline)

    # Check update_cam not called inside training loop
    lines = train_src.splitlines()
    loop_started = False
    update_cam_in_loop = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Heuristic: training loop starts at a for/while with iter/step/epoch
        if not loop_started:
            low = stripped.lower()
            if (stripped.startswith("for ") or stripped.startswith("while ")) and \
               any(kw in low for kw in ["iter", "step", "epoch", "range"]):
                loop_started = True
        if loop_started and "update_cam" in stripped and not stripped.startswith("#"):
            update_cam_in_loop.append((i, line.rstrip()))

    if update_cam_in_loop:
        check("update_cam not called inside training loop", False,
              f"Found {len(update_cam_in_loop)} active call(s):\n" +
              "\n".join(f"  line {l}: {c}" for l, c in update_cam_in_loop))
    else:
        check("update_cam not called inside training loop (replaced by spline)", True,
              "no active update_cam calls found inside training loop")

    # Bonus: check pose_network still present for warm-start
    has_pose_net = "pose_network" in train_src or "create_pose_network" in train_src
    if has_pose_net:
        info("pose_network still present in file (used for warm-start init — correct)")
    else:
        warn("pose_network reference not found — confirm warm-start initialization path")

else:
    warn(f"{TRAIN_FILE} not found in current directory — "
         "are you running from the repo root?")

# ── 15. STEP1.2 comments ──────────────────────────────────────────────────────
print("\n── Checking STEP1.2 comments ────────────────────────────────────────\n")
for fpath in ["scene/camera_spline.py", TRAIN_FILE]:
    if os.path.exists(fpath):
        with open(fpath) as f:
            src = f.read()
        count = src.count("# STEP1.2")
        check(f"# STEP1.2 comments present in {fpath}",
              count > 0, f"found {count} occurrence(s)")
    else:
        warn(f"# STEP1.2 check skipped — {fpath} not found")

# ── 16. No numpy / .detach() in get_pose forward path ────────────────────────
print("\n── Static analysis of get_pose ──────────────────────────────────────\n")
SPLINE_FILE = "scene/camera_spline.py"
if os.path.exists(SPLINE_FILE):
    with open(SPLINE_FILE) as f:
        spline_src = f.read()
    try:
        tree = ast.parse(spline_src)
        get_pose_src = ""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "get_pose":
                    get_pose_src = ast.get_source_segment(spline_src, node) or ""
                    break
        if get_pose_src:
            has_numpy  = ".numpy()" in get_pose_src or "import numpy" in get_pose_src
            has_detach = ".detach()" in get_pose_src
            check("get_pose has no .numpy() calls (would break autograd)",
                  not has_numpy,
                  "found .numpy() — remove it" if has_numpy else "clean")
            check("get_pose has no .detach() calls (would break gradient flow)",
                  not has_detach,
                  "found .detach() — remove it" if has_detach else "clean")
        else:
            warn("Could not isolate get_pose source for static analysis")
    except SyntaxError as e:
        warn(f"AST parse failed on {SPLINE_FILE}: {e}")
else:
    warn(f"{SPLINE_FILE} not found — skipping static analysis")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  All checks passed.")
    print("  Step 1.2 verified. Ready for Step 1.3:")
    print("  (two-stage training loop + stability losses)")
else:
    failed = [name for name, ok in results if not ok]
    print("  Fix these before proceeding:")
    for name in failed:
        print(f"    - {name}")
print("─" * 62 + "\n")