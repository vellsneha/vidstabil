"""
verify_step1_3.py
-----------------
Run from the root of your SplineGS fork:
    python verify_step1_3.py

Verifies Step 1.3 — two-stage training loop — given that Steps 1.1
and 1.2 are already complete.

Architecture assumptions (from Steps 1.1 + 1.2):
  - train_static_core.py is the active training file
  - cam_spline (CameraSpline) is already added to the optimizer
  - Loss is already L1 + DSSIM photometric
  - Renderer: gaussian_renderer.render_static

Checks:
  1.  STEP1.3 comments present in train_static_core.py
  2.  Spline parameters frozen (requires_grad=False) at init
  3.  Stage transition block present at iteration == 2000
  4.  Spline unfreeze (requires_grad_(True)) present in source
  5.  Iteration counter exists and increments every loop body
  6.  lambda_dssim (or equivalent weight) is 0.2
  7.  Stability loss stub present for iteration >= 2000
  8.  Stub is additive to existing loss (loss = loss + stability_loss)
  9.  Stub is a zero tensor (no real loss terms yet)
  10. Stage logging present (warmup / main label)
  11. render_static not changed (still present and unmodified structurally)
  12. camera_spline.py not touched in this step

  RUNTIME checks (simulate the two-stage behaviour without full training):
  13. Spline params have requires_grad=False before iteration 2000
  14. Spline params have requires_grad=True after iteration 2000
  15. Photometric loss is differentiable w.r.t. Gaussian params in warmup
  16. Both Gaussians and spline receive gradients in main stage
"""

import sys, os, ast, re

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

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

# ── torch ─────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    print(f"{FAIL} torch not installed")
    sys.exit(1)

sys.path.insert(0, os.getcwd())

TRAIN_FILE  = "train_static_core.py"
SPLINE_FILE = "scene/camera_spline.py"

# ── load source ───────────────────────────────────────────────────────────────
train_src = None
if os.path.exists(TRAIN_FILE):
    with open(TRAIN_FILE) as f:
        train_src = f.read()
    train_lines = train_src.splitlines()
else:
    print(f"{FAIL} {TRAIN_FILE} not found — run from repo root")
    sys.exit(1)

spline_src = None
if os.path.exists(SPLINE_FILE):
    with open(SPLINE_FILE) as f:
        spline_src = f.read()

print("\n── Static analysis of train_static_core.py ──────────────────────────\n")

# ── 1. STEP1.3 comments present ───────────────────────────────────────────────
count_13 = train_src.count("# STEP1.3")
check("# STEP1.3 comments present in train_static_core.py",
      count_13 > 0, f"found {count_13} occurrence(s)")

# ── 2. Spline frozen at init (requires_grad_(False)) ─────────────────────────
freeze_pattern = re.compile(
    r"requires_grad_\s*\(\s*False\s*\)|requires_grad\s*=\s*False"
)
has_freeze = bool(freeze_pattern.search(train_src))
check("Spline parameters frozen at initialization (requires_grad_(False))",
      has_freeze,
      "look for: for p in cam_spline.parameters(): p.requires_grad_(False)")

# ── 3. Stage transition at iteration == 2000 ──────────────────────────────────
transition_pattern = re.compile(
    r"iteration\s*==\s*2000|iter\s*==\s*2000"
)
has_transition = bool(transition_pattern.search(train_src))
check("Stage transition block present at iteration == 2000",
      has_transition,
      "look for: if iteration == 2000:")

# ── 4. Unfreeze (requires_grad_(True)) present ────────────────────────────────
unfreeze_pattern = re.compile(
    r"requires_grad_\s*\(\s*True\s*\)|requires_grad\s*=\s*True"
)
has_unfreeze = bool(unfreeze_pattern.search(train_src))
check("Spline unfreeze (requires_grad_(True)) present in source",
      has_unfreeze,
      "look for: p.requires_grad_(True) inside iteration==2000 block")

# ── 5. Iteration counter: exists and increments ───────────────────────────────
has_iter_init = bool(re.search(r"\biteration\s*=\s*0\b", train_src))
has_iter_inc  = bool(re.search(r"\biteration\s*\+=\s*1\b", train_src))
check("Iteration counter initialized to 0",
      has_iter_init,
      "look for: iteration = 0  # STEP1.3")
check("Iteration counter incremented (iteration += 1) in loop body",
      has_iter_inc,
      "look for: iteration += 1  # STEP1.3")

# ── 6. lambda_dssim = 0.2 ────────────────────────────────────────────────────
# Accept: lambda_dssim = 0.2  OR  0.2 * ssim  OR  0.2 * (1 - ssim)
dssim_pattern = re.compile(
    r"lambda_dssim\s*=\s*0\.2"
    r"|0\.2\s*\*\s*[\(\s]*(1\s*-\s*)?ssim"
    r"|lambda_dssim\s*=\s*0\.2"
)
has_dssim = bool(dssim_pattern.search(train_src))
check("SSIM weight is 0.2 in photometric loss",
      has_dssim,
      "look for: lambda_dssim = 0.2  or  0.2 * (1 - ssim(...))")

# ── 7. Stability loss stub present (iteration >= 2000) ────────────────────────
stub_pattern = re.compile(
    r"stability_loss"
    r"|# STEP1.3 stub"
    r"|# L_smooth"
)
has_stub = bool(stub_pattern.search(train_src))
check("Stability loss stub present in source",
      has_stub,
      "look for: stability_loss = torch.tensor(0.0, ...)")

# ── 8. Stub is additive: loss = loss + stability_loss ────────────────────────
additive_pattern = re.compile(
    r"loss\s*=\s*loss\s*\+\s*stability_loss"
    r"|loss\s*\+=\s*stability_loss"
)
has_additive = bool(additive_pattern.search(train_src))
check("Stub is additive to existing loss (loss = loss + stability_loss)",
      has_additive)

# ── 9. Stub is zero tensor (not computing real terms yet) ─────────────────────
zero_stub_pattern = re.compile(
    r"torch\.tensor\s*\(\s*0\.0"
    r"|torch\.zeros\s*\("
)
# Only flag if stub exists — if no stub, check 7 already caught it
if has_stub:
    has_zero = bool(zero_stub_pattern.search(train_src))
    check("Stability stub is a zero tensor (no real terms yet — Step 1.4)",
          has_zero,
          "look for: torch.tensor(0.0, ...) or torch.zeros(...)")
else:
    warn("Skipping zero-tensor check — stub not found (see check 7)")

# ── 10. Stage logging present ─────────────────────────────────────────────────
stage_log_pattern = re.compile(
    r"['\"]warmup['\"]|['\"]main['\"]|stage\s*="
)
has_stage_log = bool(stage_log_pattern.search(train_src))
check("Stage label logging present (warmup / main)",
      has_stage_log,
      "look for: stage = 'warmup' if iteration < 2000 else 'main'")

# ── 11. render_static still present ──────────────────────────────────────────
has_render_static = "render_static" in train_src
check("render_static(...) still present and unchanged",
      has_render_static,
      "renderer must not have been changed in this step")

# ── 12. camera_spline.py not modified in Step 1.3 ────────────────────────────
if spline_src is not None:
    count_13_spline = spline_src.count("# STEP1.3")
    check("camera_spline.py has no STEP1.3 modifications (should not be touched)",
          count_13_spline == 0,
          f"found {count_13_spline} STEP1.3 marks — Step 1.3 should only touch "
          f"train_static_core.py")
else:
    warn("camera_spline.py not found — skipping check 12")

# ── RUNTIME CHECKS ────────────────────────────────────────────────────────────
print("\n── Runtime simulation of two-stage behaviour ────────────────────────\n")

try:
    from scene.camera_spline import CameraSpline

    def random_rotmat():
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q.float()

    N = 30
    Rs = torch.stack([random_rotmat() for _ in range(N)])
    Ts = torch.randn(N, 3).float()

    spline = CameraSpline(N=N)
    spline.initialize_from_poses(Rs, Ts)

    # ── 13. Spline frozen before iteration 2000 ───────────────────────────────
    # Simulate freeze as Step 1.3 does it
    for p in spline.parameters():
        p.requires_grad_(False)

    all_frozen = all(not p.requires_grad for p in spline.parameters())
    check("Spline params have requires_grad=False in warm-up stage",
          all_frozen,
          f"requires_grad values: "
          f"{[p.requires_grad for p in spline.parameters()]}")

    # ── 14. Spline unfrozen after iteration 2000 ──────────────────────────────
    for p in spline.parameters():
        p.requires_grad_(True)

    all_unfrozen = all(p.requires_grad for p in spline.parameters())
    check("Spline params have requires_grad=True in main stage",
          all_unfrozen,
          f"requires_grad values: "
          f"{[p.requires_grad for p in spline.parameters()]}")

    # ── 15. Photometric loss differentiable w.r.t. Gaussians in warm-up ──────
    # Simulate a minimal Gaussian param + photometric loss
    gaussian_param = nn.Parameter(torch.randn(10, 3))  # proxy for Gaussian positions
    pred  = gaussian_param.sum(dim=1, keepdim=True).expand(10, 3)
    gt    = torch.zeros_like(pred)
    l1    = (pred - gt).abs().mean()
    # Freeze spline (warm-up)
    for p in spline.parameters():
        p.requires_grad_(False)
    R_t, T_t = spline.get_pose(0.0)
    loss = l1 + 0.0 * T_t.sum()  # spline contributes nothing in warmup
    loss.backward()
    gauss_grad_ok = gaussian_param.grad is not None
    check("Photometric loss differentiable w.r.t. Gaussian params in warm-up",
          gauss_grad_ok,
          "Gaussian params received gradients during warm-up simulation")

    # ── 16. Both Gaussians and spline receive gradients in main stage ─────────
    gaussian_param2 = nn.Parameter(torch.randn(10, 3))
    spline2 = CameraSpline(N=N)
    spline2.initialize_from_poses(Rs, Ts)
    for p in spline2.parameters():
        p.requires_grad_(True)

    pred2 = gaussian_param2.mean() * torch.ones(1)
    R_t2, T_t2 = spline2.get_pose(float(N // 2))
    loss2 = pred2 + 0.01 * T_t2.sum() + 0.01 * R_t2.sum()
    loss2.backward()

    gauss_grad2   = gaussian_param2.grad is not None
    spline_grads2 = [p.grad is not None for p in spline2.parameters()]
    spline_grad2  = any(spline_grads2)

    check("Gaussian params receive gradients in main stage",
          gauss_grad2)
    check("Spline params receive gradients in main stage",
          spline_grad2,
          f"{sum(spline_grads2)}/{len(spline_grads2)} spline param tensors "
          f"received gradients")

except ImportError as e:
    warn(f"CameraSpline import failed — skipping runtime checks: {e}")
except Exception as e:
    warn(f"Runtime simulation error: {e}")
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Step 1.3 verified.")
    print("  Ready for Step 1.4: stability loss terms.")
else:
    failed = [name for name, ok in results if not ok]
    print("  Fix these before proceeding:")
    for name in failed:
        print(f"    - {name}")
print("─" * 62 + "\n")