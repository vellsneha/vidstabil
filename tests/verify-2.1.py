"""
verify-2.1.py
-------------
Run from repo root:
    python verify-2.1.py

Verifies Step 2.1 — chunked windowed optimization.
Architecture context (from Steps 1.1–1.4):
  - train_static_core.py is the active file
  - cam_spline (CameraSpline) is global, shared across chunks
  - render_static is the renderer
  - stability losses and weights are unchanged from Step 1.4

Checks:
  Static analysis (train_static_core.py):
  1.  STEP2.1 comments present
  2.  CHUNK_THRESHOLD, CHUNK_SIZE, OVERLAP constants defined
  3.  build_chunk_indices function exists (top-level)
  4.  use_chunked = (total_frames > CHUNK_THRESHOLD) branch exists
  5.  _train_chunked function exists (top-level)
  6.  Per-chunk GaussianModel instantiation inside _train_chunked
  7.  chunk_cameras filtering by frame range inside _train_chunked
  8.  pose_optimizer.step() called inside _train_chunked (global spline updated)
  9.  chunk_optimizer (per-chunk Gaussian optimizer) exists in _train_chunked
  10. set_camera_pose_from_spline called inside _train_chunked
  11. render_static called inside _train_chunked
  12. Photometric loss (l1_loss + lambda_dssim * ssim) inside _train_chunked
  13. Stability loss block inside _train_chunked (all four terms)
  14. Chunk warm-up freeze/unfreeze (requires_grad) inside _train_chunked
  15. Short-video else-branch present and structurally unchanged
  16. STEP2.1 comment in train_entrypoint.py

  Unit tests (build_chunk_indices):
  17. build_chunk_indices(150, 70, 20) == [(0,70),(50,120),(100,150)]
  18. build_chunk_indices covers entire range (no frame gaps)
  19. build_chunk_indices overlap is correct between consecutive chunks
  20. build_chunk_indices handles total_frames <= chunk_size (single chunk)
  21. build_chunk_indices last chunk ends exactly at total_frames

  Compatibility:
  22. camera_spline.py has zero STEP2.1 marks (not touched)
  23. verify-1.4.py still importable (no structural breakage)
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

TRAIN_FILE   = "train_static_core.py"
ENTRY_FILE   = "train_entrypoint.py"
SPLINE_FILE  = "scene/camera_spline.py"

# ── load sources ──────────────────────────────────────────────────────────────
def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

train_src  = load(TRAIN_FILE)
entry_src  = load(ENTRY_FILE)
spline_src = load(SPLINE_FILE)

if train_src is None:
    print(f"{FAIL} {TRAIN_FILE} not found — run from repo root")
    sys.exit(1)

print("\n── Static analysis of train_static_core.py ──────────────────────────\n")

# ── 1. STEP2.1 comments ───────────────────────────────────────────────────────
n21 = train_src.count("# STEP2.1")
check("# STEP2.1 comments present in train_static_core.py",
      n21 > 0, f"found {n21} occurrence(s)")

# ── 2. Constants defined ──────────────────────────────────────────────────────
has_threshold = bool(re.search(r"CHUNK_THRESHOLD\s*=\s*\d+", train_src))
has_size      = bool(re.search(r"CHUNK_SIZE\s*=\s*\d+",      train_src))
has_overlap   = bool(re.search(r"OVERLAP\s*=\s*\d+",         train_src))
check("CHUNK_THRESHOLD constant defined", has_threshold)
check("CHUNK_SIZE constant defined",      has_size)
check("OVERLAP constant defined",         has_overlap)

# extract values for later use
def extract_int(pattern, src, default=None):
    m = re.search(pattern, src)
    return int(m.group(1)) if m else default

CHUNK_THRESHOLD = extract_int(r"CHUNK_THRESHOLD\s*=\s*(\d+)", train_src, 150)
CHUNK_SIZE      = extract_int(r"CHUNK_SIZE\s*=\s*(\d+)",      train_src, 70)
OVERLAP         = extract_int(r"OVERLAP\s*=\s*(\d+)",         train_src, 20)

# ── 3. build_chunk_indices exists ─────────────────────────────────────────────
has_bci_def = bool(re.search(r"^def\s+build_chunk_indices\s*\(", train_src, re.MULTILINE))
check("build_chunk_indices function defined (top-level)", has_bci_def)

# ── 4. use_chunked branch ─────────────────────────────────────────────────────
has_use_chunked = bool(re.search(
    r"use_chunked\s*=\s*\(?\s*total_frames\s*>\s*CHUNK_THRESHOLD", train_src))
check("use_chunked = (total_frames > CHUNK_THRESHOLD) branch exists",
      has_use_chunked)

# ── 5. _train_chunked function exists ─────────────────────────────────────────
has_tc_def = bool(re.search(r"^def\s+_train_chunked\s*\(", train_src, re.MULTILINE))
check("_train_chunked function defined (top-level)", has_tc_def)

# ── Extract _train_chunked body for focused checks ────────────────────────────
tc_body = ""
if has_tc_def:
    try:
        tree = ast.parse(train_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_train_chunked":
                tc_body = ast.get_source_segment(train_src, node) or ""
                break
    except SyntaxError:
        warn("AST parse failed — falling back to full-source search for checks 6–14")
        tc_body = train_src  # degrade gracefully

print("\n── Checks inside _train_chunked body ────────────────────────────────\n")

search = tc_body if tc_body else train_src

# ── 6. Per-chunk GaussianModel ────────────────────────────────────────────────
has_chunk_gm = bool(re.search(r"GaussianModel\s*\(", search))
check("Per-chunk GaussianModel instantiated inside _train_chunked", has_chunk_gm)

# ── 7. chunk_cameras filtering ────────────────────────────────────────────────
has_cam_filter = bool(re.search(
    r"chunk_cameras|c_start\s*<=.*uid|uid.*c_start", search))
check("chunk_cameras filtered by frame range inside _train_chunked",
      has_cam_filter,
      "look for: [cam for cam in ... if c_start <= cam.uid < c_end]")

# ── 8. pose_optimizer.step() inside _train_chunked ───────────────────────────
has_pose_step = bool(re.search(r"pose_optimizer\.step\s*\(\s*\)", search))
check("pose_optimizer.step() called inside _train_chunked (global spline updated)",
      has_pose_step)

# ── 9. chunk_optimizer (per-chunk Gaussian optimizer) ────────────────────────
has_chunk_opt = bool(re.search(r"chunk_optimizer", search))
check("chunk_optimizer (per-chunk) exists in _train_chunked", has_chunk_opt)

# ── 10. set_camera_pose_from_spline called ───────────────────────────────────
has_set_pose = bool(re.search(r"set_camera_pose_from_spline\s*\(", search))
check("set_camera_pose_from_spline called inside _train_chunked", has_set_pose)

# ── 11. render_static called ─────────────────────────────────────────────────
has_render = bool(re.search(r"render_static\s*\(", search))
check("render_static called inside _train_chunked", has_render)

# ── 12. Photometric loss ──────────────────────────────────────────────────────
has_photo = bool(re.search(
    r"l1_loss\s*\(|ll1\s*=|lambda_dssim", search))
check("Photometric loss (l1_loss + lambda_dssim * ssim) inside _train_chunked",
      has_photo)

# ── 13. Stability losses inside _train_chunked ───────────────────────────────
has_smooth  = bool(re.search(r"loss_smooth",  search))
has_jitter  = bool(re.search(r"loss_jitter",  search))
has_fov     = bool(re.search(r"loss_fov",     search))
has_dilated = bool(re.search(r"loss_dilated", search))
all_stability = has_smooth and has_jitter and has_fov and has_dilated
check("All four stability losses present inside _train_chunked",
      all_stability,
      f"smooth={has_smooth}, jitter={has_jitter}, "
      f"fov={has_fov}, dilated={has_dilated}")

# ── 14. Chunk warm-up freeze/unfreeze ────────────────────────────────────────
has_chunk_freeze   = bool(re.search(r"requires_grad_\s*\(\s*False\s*\)", search))
has_chunk_unfreeze = bool(re.search(r"requires_grad_\s*\(\s*True\s*\)",  search))
check("Chunk warm-up freeze (requires_grad_(False)) inside _train_chunked",
      has_chunk_freeze)
check("Chunk warm-up unfreeze (requires_grad_(True)) inside _train_chunked",
      has_chunk_unfreeze)

print("\n── Checking short-video else-branch ─────────────────────────────────\n")

# ── 15. Short-video else-branch present ──────────────────────────────────────
has_else = bool(re.search(
    r"else\s*:.*#\s*STEP2\.1.*short video|"
    r"else\s*:.*short video|"
    r"#\s*STEP2\.1.*short video.*else",
    train_src, re.IGNORECASE | re.DOTALL))
# Fallback: just check else: exists near use_chunked
if not has_else:
    has_else = bool(re.search(r"else\s*:\s*#\s*STEP2\.1", train_src))
check("Short-video else-branch present and labelled",
      has_else,
      "look for: else:  # STEP2.1 — short video, original path unchanged")

# ── 16. train_entrypoint.py comment ──────────────────────────────────────────
if entry_src is not None:
    has_entry_comment = "STEP2.1" in entry_src
    check("STEP2.1 comment present in train_entrypoint.py",
          has_entry_comment)
else:
    warn("train_entrypoint.py not found")

print("\n── Unit tests: build_chunk_indices ──────────────────────────────────\n")

# ── Import build_chunk_indices ────────────────────────────────────────────────
bci = None
if has_bci_def:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tsc", TRAIN_FILE)
        mod  = importlib.util.module_from_spec(spec)
        # Don't exec the full module (has heavy deps) — extract function via AST + exec
        tree = ast.parse(train_src)
        func_src = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_chunk_indices":
                func_src = ast.get_source_segment(train_src, node) or ""
                break
        if func_src:
            ns = {}
            exec(compile(func_src, TRAIN_FILE, "exec"), ns)
            bci = ns.get("build_chunk_indices")
    except Exception as e:
        warn(f"Could not extract build_chunk_indices for unit tests: {e}")

if bci is not None:
    # ── 17. Specific output for (150, 70, 20) ─────────────────────────────────
    expected = [(0, 70), (50, 120), (100, 150)]
    got      = bci(150, 70, 20)
    check("build_chunk_indices(150, 70, 20) == [(0,70),(50,120),(100,150)]",
          got == expected,
          f"got {got}")

    # ── 18. Full coverage — no frame gaps ────────────────────────────────────
    def no_gaps(total, size, overlap):
        chunks = bci(total, size, overlap)
        covered = set()
        for s, e in chunks:
            covered.update(range(s, e))
        return covered == set(range(total))

    for total in [100, 150, 200, 300, 500]:
        ok = no_gaps(total, CHUNK_SIZE, OVERLAP)
        check(f"build_chunk_indices covers all frames for total={total}",
              ok, f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}")

    # ── 19. Overlap between consecutive chunks ───────────────────────────────
    def check_overlap(total, size, overlap):
        chunks = bci(total, size, overlap)
        if len(chunks) < 2:
            return True
        for (s1, e1), (s2, e2) in zip(chunks[:-1], chunks[1:]):
            actual_overlap = e1 - s2
            if actual_overlap < overlap - 1:  # allow off-by-one for boundary
                return False
        return True

    ok_ov = check_overlap(300, CHUNK_SIZE, OVERLAP)
    check("Consecutive chunks have correct overlap",
          ok_ov,
          f"expected ~{OVERLAP} frames overlap between adjacent chunks")

    # ── 20. Single chunk when total <= chunk_size ────────────────────────────
    single = bci(50, 70, 20)
    check("build_chunk_indices returns single chunk when total <= chunk_size",
          len(single) == 1 and single[0] == (0, 50),
          f"got {single}")

    # ── 21. Last chunk ends at total_frames ──────────────────────────────────
    for total in [150, 200, 333]:
        chunks = bci(total, CHUNK_SIZE, OVERLAP)
        last_end = chunks[-1][1]
        check(f"Last chunk ends at total_frames={total}",
              last_end == total,
              f"got last chunk end={last_end}")

else:
    warn("build_chunk_indices not importable — skipping unit tests 17–21")
    for i in range(17, 22):
        results.append((f"Unit test {i} (skipped — import failed)", False))

print("\n── Compatibility checks ──────────────────────────────────────────────\n")

# ── 22. camera_spline.py untouched ───────────────────────────────────────────
if spline_src is not None:
    n21_spline = spline_src.count("# STEP2.1")
    check("camera_spline.py has zero STEP2.1 marks (not touched)",
          n21_spline == 0,
          f"found {n21_spline} STEP2.1 marks")
else:
    warn("camera_spline.py not found")

# ── 23. verify-1.4.py still importable ───────────────────────────────────────
if os.path.exists("verify-1.4.py"):
    with open("verify-1.4.py") as f:
        v14_src = f.read()
    has_syntax_error = False
    try:
        ast.parse(v14_src)
    except SyntaxError:
        has_syntax_error = True
    check("verify-1.4.py parses without SyntaxError (no structural breakage)",
          not has_syntax_error)
else:
    warn("verify-1.4.py not found — skipping compatibility check")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Step 2.1 verified. Ready for Step 2.2.")
else:
    failed = [name for name, ok in results if not ok]
    print("  Fix before proceeding:")
    for name in failed:
        print(f"    - {name}")
print("─" * 62 + "\n")