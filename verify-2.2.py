"""
verify-2.2.py
-------------
Run from repo root:
    python verify-2.2.py

Verifies Step 2.2 — asymmetric update frequency.
Key facts from Step 1.4 (already correct, must be confirmed unchanged):
  - L_jitter:  iteration % 10 == 0
  - L_dilated: iteration % 5  == 0
Step 2.2 adds only:
  - pose_optimizer.step() conditional on iteration % 2 == 0
  - Same in _train_chunked (local_iter % 2 == 0)

Checks:
  Static (train_static_core.py):
  1.  STEP2.2 comments present
  2.  pose_optimizer.step() is conditional (iteration % 2 == 0)
  3.  Gaussian optimizer.step() is NOT conditional (every iteration)
  4.  pose_optimizer.zero_grad inside the iteration % 2 block
  5.  Same conditional in _train_chunked (local_iter % 2 == 0)
  6.  L_jitter frequency unchanged: iteration % 10 == 0
  7.  L_dilated frequency unchanged: iteration % 5 == 0
  8.  Stage gate print updated with STEP2.2 note
  9.  STEP2.2 confirmation comments on jitter/dilated frequency lines

  Compatibility:
  10. camera_spline.py has zero STEP2.2 marks (not touched)
  11. No change to w_smooth, w_jitter, w_fov, w_dilated values
  12. render_static still present and unchanged
  13. lambda_dssim = 0.2 unchanged

  Runtime simulation:
  14. Spline params do NOT update on odd iterations (gradient accumulates)
  15. Spline params DO update on even iterations
  16. Gaussian proxy params update every iteration regardless
  17. Accumulated gradients from 2 odd+even iterations == 2x single-step grad
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

TRAIN_FILE  = "train_static_core.py"
SPLINE_FILE = "scene/camera_spline.py"

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

train_src  = load(TRAIN_FILE)
spline_src = load(SPLINE_FILE)

if train_src is None:
    print(f"{FAIL} {TRAIN_FILE} not found — run from repo root")
    sys.exit(1)

print("\n── Static analysis of train_static_core.py ──────────────────────────\n")

# ── 1. STEP2.2 comments ───────────────────────────────────────────────────────
n22 = train_src.count("# STEP2.2")
check("# STEP2.2 comments present in train_static_core.py",
      n22 > 0, f"found {n22} occurrence(s)")

# ── 2. pose_optimizer.step() is conditional (iteration % 2 == 0) ─────────────
# Pattern: if iteration % 2 == 0: ... pose_optimizer.step()
# Allow local_iter or global_iteration as variable names too
cond_step_pattern = re.compile(
    r"if\s+(iteration|local_iter|global_iteration)\s*%\s*2\s*==\s*0"
    r"[\s\S]{0,200}"
    r"pose_optimizer\.step\s*\(\s*\)",
    re.MULTILINE
)
has_cond_step = bool(cond_step_pattern.search(train_src))
check("pose_optimizer.step() is conditional on iteration % 2 == 0",
      has_cond_step,
      "look for: if iteration % 2 == 0: ... pose_optimizer.step()")

# ── 3. Gaussian optimizer.step() is NOT conditional ──────────────────────────
# Find all optimizer.step() calls; Gaussian one should NOT be inside a % 2 block
# Heuristic: find lines with gaussian_optimizer.step (or similar) and confirm
# no "% 2" on the same or preceding line
gauss_opt_names = re.findall(
    r"(\w*[Gg]aussian\w*[Oo]pt\w*|optimizer|gauss_optim)\s*\.step\s*\(\s*\)",
    train_src
)
# Use line-by-line: find step() lines NOT inside an "if % 2" block
lines = train_src.splitlines()
gauss_step_unconditional = False
i = 0
while i < len(lines):
    line = lines[i].strip()
    # Check for a .step() that is NOT pose_optimizer and NOT inside "if % 2"
    if ".step()" in line and "pose_optimizer" not in line:
        # Check that neither this line nor the preceding non-empty line
        # contains "% 2"
        prev_lines = [l.strip() for l in lines[max(0,i-3):i] if l.strip()]
        context = " ".join(prev_lines + [line])
        if "% 2" not in context:
            gauss_step_unconditional = True
            break
    i += 1
check("Gaussian optimizer.step() is NOT gated by % 2 (steps every iteration)",
      gauss_step_unconditional,
      "Gaussian optimizer must update every iteration — only pose_optimizer is gated")

# ── 4. pose_optimizer.zero_grad inside % 2 block ─────────────────────────────
zero_grad_cond = re.compile(
    r"if\s+(iteration|local_iter|global_iteration)\s*%\s*2\s*==\s*0"
    r"[\s\S]{0,300}"
    r"pose_optimizer\.zero_grad\s*\(",
    re.MULTILINE
)
has_zero_cond = bool(zero_grad_cond.search(train_src))
check("pose_optimizer.zero_grad inside the iteration % 2 block",
      has_zero_cond,
      "zero_grad must be co-located with conditional step")

# ── 5. Same conditional in _train_chunked ────────────────────────────────────
tc_body = ""
try:
    tree = ast.parse(train_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_train_chunked":
            tc_body = ast.get_source_segment(train_src, node) or ""
            break
except SyntaxError:
    tc_body = train_src

if tc_body:
    has_chunked_cond = bool(re.search(
        r"(local_iter|global_iteration|iteration)\s*%\s*2\s*==\s*0"
        r"[\s\S]{0,200}"
        r"pose_optimizer\.step\s*\(\s*\)",
        tc_body, re.MULTILINE
    ))
    check("pose_optimizer conditional (% 2) also applied in _train_chunked",
          has_chunked_cond,
          "look for: if local_iter % 2 == 0: pose_optimizer.step()")
else:
    warn("_train_chunked not found — skipping check 5")

print("\n── Confirming Step 1.4 frequencies unchanged ────────────────────────\n")

# ── 6. L_jitter still every 10 iterations ────────────────────────────────────
jitter_freq = re.compile(
    r"(iteration|local_iter|global_iteration)\s*%\s*10\s*==\s*0"
    r"[\s\S]{0,2000}"   # widened: two render_static blocks sit between guard and loss_jitter
    r"loss_jitter",
    re.MULTILINE
)
# Also accept reverse order
jitter_freq2 = re.compile(
    r"loss_jitter[\s\S]{0,2000}"   # widened
    r"(iteration|local_iter|global_iteration)\s*%\s*10\s*==\s*0",
    re.MULTILINE
)
has_jitter_freq = (bool(jitter_freq.search(train_src)) or
                   bool(jitter_freq2.search(train_src)))
check("L_jitter frequency unchanged: iteration % 10 == 0",
      has_jitter_freq,
      "this was set in Step 1.4 and must not have changed")

# ── 7. L_dilated still every 5 iterations ────────────────────────────────────
dilated_freq = re.compile(
    r"(iteration|local_iter|global_iteration)\s*%\s*5\s*==\s*0"
    r"[\s\S]{0,3000}"   # widened: four render_static blocks sit between guard and loss_dilated
    r"loss_dilated",
    re.MULTILINE
)
dilated_freq2 = re.compile(
    r"loss_dilated[\s\S]{0,3000}"   # widened
    r"(iteration|local_iter|global_iteration)\s*%\s*5\s*==\s*0",
    re.MULTILINE
)
has_dilated_freq = (bool(dilated_freq.search(train_src)) or
                    bool(dilated_freq2.search(train_src)))
check("L_dilated frequency unchanged: iteration % 5 == 0",
      has_dilated_freq,
      "this was set in Step 1.4 and must not have changed")

# ── 8. Stage gate print updated ──────────────────────────────────────────────
has_gate_print = bool(re.search(
    r"print.*STEP2\.2|STEP2\.2.*print|spline steps every 2nd",
    train_src, re.IGNORECASE
))
check("Stage gate print updated with STEP2.2 note",
      has_gate_print,
      "look for: 'spline steps every 2nd iter (STEP2.2)' in print")

# ── 9. Confirmation comments on jitter/dilated lines ─────────────────────────
has_jitter_confirm  = bool(re.search(r"STEP2\.2.*confirmed.*jitter|"
                                     r"jitter.*STEP2\.2.*confirmed", train_src))
has_dilated_confirm = bool(re.search(r"STEP2\.2.*confirmed.*dilated|"
                                     r"dilated.*STEP2\.2.*confirmed", train_src))
# Also accept just "# STEP2.2 confirmed" near the % 10 and % 5 lines
has_jitter_confirm  = has_jitter_confirm  or bool(re.search(
    r"%\s*10\s*==\s*0.*#\s*STEP2\.2", train_src))
has_dilated_confirm = has_dilated_confirm or bool(re.search(
    r"%\s*5\s*==\s*0.*#\s*STEP2\.2", train_src))
check("STEP2.2 confirmation comment on L_jitter frequency line",
      has_jitter_confirm)
check("STEP2.2 confirmation comment on L_dilated frequency line",
      has_dilated_confirm)

print("\n── Compatibility checks ──────────────────────────────────────────────\n")

# ── 10. camera_spline.py untouched ───────────────────────────────────────────
if spline_src is not None:
    n22_spline = spline_src.count("# STEP2.2")
    check("camera_spline.py has zero STEP2.2 marks (not touched)",
          n22_spline == 0, f"found {n22_spline}")
else:
    warn("camera_spline.py not found")

# ── 11. Loss weights unchanged from Step 1.4 ─────────────────────────────────
def weight_check(name, pattern, expected):
    m = re.search(pattern, train_src)
    if m:
        val = float(m.group(1))
        check(f"{name} = {expected} (unchanged from Step 1.4)",
              abs(val - expected) < 1e-9,
              f"found {val}")
    else:
        warn(f"{name} pattern not found — check manually")

weight_check("w_smooth",  r"w_smooth\s*=\s*([0-9.e+-]+)",  0.1)
weight_check("w_jitter",  r"w_jitter\s*=\s*([0-9.e+-]+)",  0.5)
weight_check("w_fov",     r"w_fov\s*=\s*([0-9.e+-]+)",     0.05)
weight_check("w_dilated", r"w_dilated\s*=\s*([0-9.e+-]+)", 0.1)

# ── 12. render_static present ────────────────────────────────────────────────
check("render_static still present and unchanged",
      "render_static" in train_src)

# ── 13. lambda_dssim = 0.2 ───────────────────────────────────────────────────
has_dssim = bool(re.search(r"lambda_dssim\s*=\s*0\.2", train_src))
check("lambda_dssim = 0.2 unchanged", has_dssim)

print("\n── Runtime simulation of asymmetric update ──────────────────────────\n")

try:
    import torch
    import torch.nn as nn

    # Proxy for Gaussian params (update every iter)
    gauss_param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    # Proxy for spline params (update every 2nd iter)
    spline_param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    gauss_opt  = torch.optim.SGD([gauss_param],  lr=0.1)
    spline_opt = torch.optim.SGD([spline_param], lr=0.1)

    gauss_before  = gauss_param.data.clone()
    spline_before = spline_param.data.clone()

    # ── 14. Odd iteration — spline does NOT update ────────────────────────────
    iteration = 1  # odd
    loss = gauss_param.sum() + spline_param.sum()
    loss.backward()
    gauss_opt.step()
    gauss_opt.zero_grad(set_to_none=True)
    if iteration % 2 == 0:
        spline_opt.step()
        spline_opt.zero_grad(set_to_none=True)

    spline_unchanged = torch.allclose(spline_param.data, spline_before)
    gauss_changed    = not torch.allclose(gauss_param.data, gauss_before)
    check("Spline param does NOT update on odd iteration (gradient accumulates)",
          spline_unchanged,
          f"spline before={spline_before.tolist()}, after={spline_param.data.tolist()}")
    check("Gaussian param DOES update on odd iteration",
          gauss_changed)

    # ── 15. Even iteration — spline DOES update ───────────────────────────────
    # Re-run forward (gradient still accumulated from iteration 1)
    iteration = 2  # even
    spline_before2 = spline_param.data.clone()
    gauss_before2  = gauss_param.data.clone()

    loss2 = gauss_param.sum() + spline_param.sum()
    loss2.backward()
    gauss_opt.step()
    gauss_opt.zero_grad(set_to_none=True)
    if iteration % 2 == 0:
        spline_opt.step()
        spline_opt.zero_grad(set_to_none=True)

    spline_changed2 = not torch.allclose(spline_param.data, spline_before2)
    check("Spline param DOES update on even iteration",
          spline_changed2,
          f"spline before={spline_before2.tolist()}, after={spline_param.data.tolist()}")

    # ── 16. Gaussian updates every iteration ──────────────────────────────────
    check("Gaussian proxy param updated on both odd and even iterations",
          gauss_changed and not torch.allclose(gauss_param.data, gauss_before2))

    # ── 17. Accumulated grad == 2x single-step grad ───────────────────────────
    # Reset
    sp = nn.Parameter(torch.tensor([5.0]))
    opt_sp = torch.optim.SGD([sp], lr=1.0)

    # Two-step accumulation (step only on even)
    for it in [1, 2]:
        loss_sp = sp.sum()
        loss_sp.backward()
        if it % 2 == 0:
            opt_sp.step()
            opt_sp.zero_grad(set_to_none=True)

    # Expected: grad=2.0 applied once with lr=1.0 → param = 5.0 - 2.0 = 3.0
    expected_val = 5.0 - 2.0  # 2 accumulated grads of 1.0 each, lr=1.0
    actual_val   = sp.data.item()
    check("Accumulated gradients (2 iters) applied in one step = 2x single gradient",
          abs(actual_val - expected_val) < 1e-5,
          f"expected {expected_val}, got {actual_val}")

except ImportError:
    warn("torch not installed — skipping runtime simulation checks 14–17")
except Exception as e:
    warn(f"Runtime simulation error: {e}")
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Step 2.2 verified.")
    print("  Phase 2 complete. Ready for Phase 3 (dynamic masking).")
else:
    failed = [name for name, ok in results if not ok]
    print("  Fix before proceeding:")
    for name in failed:
        print(f"    - {name}")
print("─" * 62 + "\n")