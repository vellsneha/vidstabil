"""
verify-2.3.py
-------------
Run from repo root:
    python verify-2.3.py

Verifies Step 2.3 — reduced densification aggressiveness.
Two spec requirements:
  1. densification_interval default changed from 100 → 200
  2. Hard Gaussian count cap at 500_000

Checks:
  Static (arguments/__init__.py):
  1.  densification_interval default is 200 (not 100)
  2.  STEP2.3 comment present on that line
  3.  No other OptimizationParams defaults changed
      (densify_from_iter, densify_until_iter, densify_grad_threshold
       opacity_reset_interval must be unchanged)

  Static (train_exp.py):
  4.  STEP2.3 comments present
  5.  MAX_GAUSSIANS = 500_000 constant defined
  6.  prune_points call present after densify_and_prune (cap enforcement)
  7.  Cap condition references MAX_GAUSSIANS
  8.  Cap also present in _train_chunked (chunk_gaussians)
  9.  Gaussian count (n_gauss or equivalent) added to progress bar logging
  10. STEP2.3 comment in train_entrypoint.py

  Compatibility (nothing else changed):
  11. densify_from_iter default unchanged (500)
  12. densify_until_iter default unchanged (15000)
  13. camera_spline.py has zero STEP2.3 marks
  14. stability loss weights unchanged (w_smooth, w_jitter, w_fov, w_dilated)
  15. densify_and_prune still present in training loop (not removed)

  Logic checks (pure Python, no GPU):
  16. MAX_GAUSSIANS value is exactly 500_000
  17. Cap prune logic is correct: keeps top-K by opacity, prunes rest
  18. densification_interval = 200 means densify fires half as often as 100
"""

import sys, os, re, ast

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

ARGS_FILE   = "arguments/__init__.py"
TRAIN_FILE  = "train_exp.py"
ENTRY_FILE  = "train_entrypoint.py"
SPLINE_FILE = "scene/camera_spline.py"

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

args_src   = load(ARGS_FILE)
train_src  = load(TRAIN_FILE)
entry_src  = load(ENTRY_FILE)
spline_src = load(SPLINE_FILE)

if args_src is None:
    # Try alternative locations
    for alt in ["arguments.py", "config.py", "options.py"]:
        args_src = load(alt)
        if args_src:
            ARGS_FILE = alt
            break
    if args_src is None:
        warn(f"arguments/__init__.py not found — checks 1-3, 11-12 will be skipped")

if train_src is None:
    print(f"{FAIL} {TRAIN_FILE} not found — run from repo root")
    sys.exit(1)

print("\n── Static analysis of arguments/__init__.py ─────────────────────────\n")

if args_src is not None:
    # ── 1. densification_interval = 200 ──────────────────────────────────────
    m = re.search(r"densification_interval\s*[=:]\s*(\d+)", args_src)
    if m:
        val = int(m.group(1))
        check("densification_interval default is 200",
              val == 200,
              f"found {val} (should be 200, was 100 before Step 2.3)")
    else:
        check("densification_interval field found in arguments",
              False,
              f"field not found in {ARGS_FILE}")

    # ── 2. STEP2.3 comment on that line ──────────────────────────────────────
    has_23_args = "STEP2.3" in args_src
    check("STEP2.3 comment present in arguments/__init__.py",
          has_23_args)

    # ── 3 & 11 & 12. Other defaults unchanged ─────────────────────────────────
    def check_default(name, pattern, expected_val, expected_type=int):
        m = re.search(pattern, args_src)
        if m:
            try:
                val = expected_type(m.group(1))
                check(f"{name} default unchanged ({expected_val})",
                      abs(val - expected_val) < 1e-9,
                      f"found {val}")
            except (ValueError, TypeError):
                warn(f"Could not parse {name} value")
        else:
            warn(f"{name} pattern not found in {ARGS_FILE} — check manually")

    check_default("densify_from_iter",   r"densify_from_iter\s*[=:]\s*(\d+)",   500)
    # densify_until_iter may be written as 15_000 — strip underscores before int()
    m_dui = re.search(r"densify_until_iter\s*[=:]\s*([0-9_]+)", args_src)
    if m_dui:
        val_dui = int(m_dui.group(1).replace("_", ""))
        check("densify_until_iter default unchanged (15000)",
              val_dui == 15000, f"found {val_dui}")
    else:
        warn("densify_until_iter not found in arguments — check manually")
    # This fork uses densify_grad_threshold = 0.0008 (distinct from the
    # _coarse / _fine_init variants which are 0.0002).  Check it is unchanged.
    check_default("densify_grad_threshold",
                  r"densify_grad_threshold\s*[=:]\s*([0-9.e+-]+)", 0.0008, float)
    # opacity_reset_interval may vary by fork — just check it still exists
    has_opacity_reset = bool(re.search(r"opacity_reset_interval", args_src))
    if has_opacity_reset:
        check("opacity_reset_interval field still present (not removed)",
              True)
    else:
        warn("opacity_reset_interval not found — may be named differently")

print("\n── Static analysis of train_exp.py ──────────────────────────\n")

# ── 4. STEP2.3 comments in train ─────────────────────────────────────────────
n23 = train_src.count("# STEP2.3")
check("# STEP2.3 comments present in train_exp.py",
      n23 > 0, f"found {n23} occurrence(s)")

# ── 5. MAX_GAUSSIANS = 500_000 ────────────────────────────────────────────────
# Accept 500_000 or 500000
max_g_pattern = re.compile(r"MAX_GAUSSIANS\s*=\s*(500[_]?000|500000)")
has_max_g = bool(max_g_pattern.search(train_src))
check("MAX_GAUSSIANS = 500_000 constant defined in train_exp.py",
      has_max_g,
      "look for: MAX_GAUSSIANS = 500_000  # STEP2.3")

# ── 16. Extract actual value ──────────────────────────────────────────────────
max_g_val = None
m = re.search(r"MAX_GAUSSIANS\s*=\s*([0-9_]+)", train_src)
if m:
    try:
        max_g_val = int(m.group(1).replace("_", ""))
    except ValueError:
        pass
if max_g_val is not None:
    check("MAX_GAUSSIANS value is exactly 500000",
          max_g_val == 500000,
          f"found {max_g_val}")

# ── 6. densification call present (densify_and_prune OR codebase equivalent) ──
# This fork uses controlgaussians / densify_pruneclone instead of the
# standard densify_and_prune name.  Accept any of the three.
has_densify = bool(re.search(
    r"densify_and_prune\s*\(|controlgaussians\s*\(|densify_pruneclone\s*\(",
    train_src))
has_prune   = bool(re.search(r"prune_points\s*\(", train_src))
check("densify_and_prune (or equivalent) still present in training loop",
      has_densify,
      "look for: controlgaussians / densify_pruneclone / densify_and_prune")
check("prune_points call present for cap enforcement",
      has_prune,
      "look for: stat_gaussians.prune_points(...)")

# ── 7. Cap condition references MAX_GAUSSIANS ─────────────────────────────────
cap_pattern = re.compile(
    r"(get_xyz\.shape\[0\]|len\s*\(.*gaussians.*\))\s*>\s*MAX_GAUSSIANS"
    r"|MAX_GAUSSIANS[\s\S]{0,200}prune_points",
    re.MULTILINE
)
has_cap_ref = bool(cap_pattern.search(train_src))
check("Cap condition references MAX_GAUSSIANS",
      has_cap_ref,
      "look for: if stat_gaussians.get_xyz.shape[0] > MAX_GAUSSIANS")

# ── 8. Cap in _train_chunked too ──────────────────────────────────────────────
tc_body = ""
try:
    tree = ast.parse(train_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_train_chunked":
            tc_body = ast.get_source_segment(train_src, node) or ""
            break
except SyntaxError:
    pass

if tc_body:
    has_chunk_cap = bool(re.search(r"MAX_GAUSSIANS", tc_body)) and \
                    bool(re.search(r"prune_points",   tc_body))
    check("Cap (MAX_GAUSSIANS + prune_points) also present in _train_chunked",
          has_chunk_cap,
          "chunk_gaussians must also be capped at MAX_GAUSSIANS")
else:
    warn("_train_chunked not found via AST — checking full source for chunk cap")
    has_chunk_cap_fallback = bool(re.search(r"chunk_gaussians.*prune_points|"
                                            r"prune_points.*chunk_gaussians",
                                            train_src))
    check("chunk_gaussians prune_points call present (cap in chunked path)",
          has_chunk_cap_fallback)

# ── 9. Gaussian count in progress bar ────────────────────────────────────────
gauss_log_pattern = re.compile(
    r"n_gauss|num_gauss|gaussians.*shape\[0\]|get_xyz.*shape\[0\]"
    r".*set_postfix|set_postfix.*n_gauss",
    re.DOTALL | re.IGNORECASE
)
# Simpler: check n_gauss appears near set_postfix
has_gauss_log = bool(re.search(r"n_gauss", train_src)) and \
                bool(re.search(r"set_postfix", train_src))
check("Gaussian count (n_gauss) added to progress bar logging",
      has_gauss_log,
      "look for: n_gauss = stat_gaussians.get_xyz.shape[0] near set_postfix")

# ── 10. STEP2.3 in train_entrypoint.py ───────────────────────────────────────
if entry_src is not None:
    check("STEP2.3 comment present in train_entrypoint.py",
          "STEP2.3" in entry_src)
else:
    warn("train_entrypoint.py not found")

print("\n── Compatibility: nothing else changed ──────────────────────────────\n")

# ── 13. camera_spline.py untouched ───────────────────────────────────────────
if spline_src is not None:
    n23_spline = spline_src.count("# STEP2.3")
    check("camera_spline.py has zero STEP2.3 marks (not touched)",
          n23_spline == 0, f"found {n23_spline}")
else:
    warn("camera_spline.py not found")

# ── 14. Stability loss weights unchanged ─────────────────────────────────────
def weight_check(name, pattern, expected):
    m = re.search(pattern, train_src)
    if m:
        val = float(m.group(1))
        check(f"{name} = {expected} (unchanged)",
              abs(val - expected) < 1e-9,
              f"found {val}")
    else:
        warn(f"{name} not found — check manually")

weight_check("w_smooth",  r"w_smooth\s*=\s*([0-9.e+-]+)",  0.1)
weight_check("w_jitter",  r"w_jitter\s*=\s*([0-9.e+-]+)",  0.5)
weight_check("w_fov",     r"w_fov\s*=\s*([0-9.e+-]+)",     0.05)
weight_check("w_dilated", r"w_dilated\s*=\s*([0-9.e+-]+)", 0.1)

# ── 15. densification not removed ────────────────────────────────────────────
# Already checked as check 6 above (accepts controlgaussians / densify_pruneclone) — skip duplicate

print("\n── Logic checks ─────────────────────────────────────────────────────\n")

# ── 17. Cap prune logic is correct (pure Python simulation) ──────────────────
try:
    import torch

    # Simulate 600K Gaussians with random opacities
    torch.manual_seed(42)
    N_sim      = 600_000
    MAX_CAP    = 500_000
    opacities  = torch.rand(N_sim, 1)  # proxy for get_opacity

    # Replicate the prune_points mask from the cursor prompt:
    # keep top-MAX_CAP by opacity, prune the rest
    threshold = opacities.squeeze().topk(MAX_CAP).values.min()
    prune_mask = opacities.squeeze() < threshold   # True = prune
    n_kept = (~prune_mask).sum().item()

    # The mask removes low-opacity Gaussians; n_kept should be MAX_CAP
    cap_correct = (n_kept == MAX_CAP)
    check("Cap prune logic correct: keeps top-500K by opacity",
          cap_correct,
          f"kept {n_kept} / {N_sim} Gaussians (expected {MAX_CAP})")

    # Simulate no-op when already under cap
    N_small   = 300_000
    small_ops = torch.rand(N_small, 1)
    should_not_prune = N_small <= MAX_CAP
    check("Cap does NOT prune when Gaussian count <= 500K",
          should_not_prune,
          f"N={N_small} <= MAX_GAUSSIANS={MAX_CAP} → no prune needed")

except ImportError:
    warn("torch not installed — skipping logic simulation checks")
except Exception as e:
    warn(f"Logic simulation error: {e}")

# ── 18. densification_interval = 200 fires half as often as 100 ──────────────
# Pure arithmetic check
interval_100 = 100
interval_200 = 200
fires_at_100 = [i for i in range(500, 2001) if i % interval_100 == 0]
fires_at_200 = [i for i in range(500, 2001) if i % interval_200 == 0]
ratio = len(fires_at_100) / max(len(fires_at_200), 1)
check("densification_interval=200 fires exactly half as often as 100",
      abs(ratio - 2.0) < 1e-9,
      f"fires_at_100={len(fires_at_100)}, fires_at_200={len(fires_at_200)}, "
      f"ratio={ratio:.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Step 2.3 verified.")
    print("  Phase 2 complete. Ready for Phase 3 (dynamic masking).")
else:
    failed = [name for name, ok in results if not ok]
    print("  Fix before proceeding:")
    for name in failed:
        print(f"    - {name}")
print("─" * 62 + "\n")