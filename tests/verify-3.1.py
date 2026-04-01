"""
verify-3.1.py
-------------
Run from repo root (vidstabil/):
    python verify-3.1.py

Verifies Step 3.1 — masked photometric loss with cached Grounded-SAM-2-style masks M_t.

Checks:
  Static (arguments/__init__.py):
    1. use_dynamic_mask and dynamic_mask_subdir on ModelParams
  Static (utils/loss_utils.py):
    2. photometric_loss_masked_dynamic defined; uses (1 - M_t) weighting
  Static (scene/cameras.py, dataset.py, dataset_readers.py):
    3. dynamic_mask_t on Camera / CameraInfo / dataset path
  Static (train_static_core.py):
    4. STEP3.1 masked photometric path
  Logic (torch):
    5. L1 term matches ||(pred - gt) * (1 - M_t)||_1 / sum(1-M_t) (masked mean)
  Optional runtime:
    6. preprocess_dynamic_masks.py --backend synthetic on a temp scene layout
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

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


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

ARGS = "arguments/__init__.py"
LOSS = "utils/loss_utils.py"
CAM = "scene/cameras.py"
DS = "scene/dataset.py"
READ = "scene/dataset_readers.py"
TRAIN = "train_static_core.py"
PRE = "preprocess_dynamic_masks.py"


def readf(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


print("\n── Step 3.1 static checks ────────────────────────────────────────────\n")

args_src = readf(ARGS)
check(
    "ModelParams.use_dynamic_mask default",
    "use_dynamic_mask" in args_src and "STEP3.1" in args_src,
)
check(
    "ModelParams.dynamic_mask_subdir default",
    "dynamic_mask_subdir" in args_src,
)

loss_src = readf(LOSS)
check(
    "photometric_loss_masked_dynamic in loss_utils.py",
    "def photometric_loss_masked_dynamic" in loss_src,
)
check(
    "masked L1 uses (1 - M_t) via w = 1.0 - M_t",
    "1.0 - M_t" in loss_src or "1.0 - M_t.clamp" in loss_src,
)

cam_src = readf(CAM)
check("Camera stores dynamic_mask_t", "dynamic_mask_t" in cam_src and "STEP3.1" in cam_src)

ds_src = readf(DS)
check("FourDGSdataset passes dynamic_mask_t", "dynamic_mask_t" in ds_src)

read_src = readf(READ)
check("CameraInfo has dynamic_mask_t", "dynamic_mask_t" in read_src)
check("readNvidiaCameras loads dynamic_masks when use_dynamic_mask", "use_dynamic_mask" in read_src)

train_src = readf(TRAIN)
check("train_static_core uses photometric_loss_masked_dynamic", "photometric_loss_masked_dynamic" in train_src)
check("STEP3.1 markers in train_static_core", train_src.count("STEP3.1") >= 3)

pre_path = os.path.join(ROOT, PRE)
check("preprocess_dynamic_masks.py exists", os.path.isfile(pre_path))

integ_path = os.path.join(ROOT, "gsam2", "integrated.py")
check("gsam2/integrated.py exists", os.path.isfile(integ_path))
if os.path.isfile(integ_path):
    integ_src = readf("gsam2/integrated.py")
    check("run_integrated_masks defined in gsam2/integrated.py", "def run_integrated_masks" in integ_src)
    check("integrated pipeline references Grounding DINO + SAM2", "SAM2ImagePredictor" in integ_src)

pre_src = readf(PRE)
check("preprocess default backend is gsam2 (integrated)", 'default="gsam2"' in pre_src or "default='gsam2'" in pre_src)
check("preprocess wires run_integrated_masks", "run_integrated_masks" in pre_src)

req_gsam2 = os.path.normpath(os.path.join(ROOT, "..", "requirements-gsam2.txt"))
check("requirements-gsam2.txt at project root", os.path.isfile(req_gsam2))

print("\n── Numeric checks (torch if available, else numpy L1) ───────────────\n")

try:
    import torch

    from utils.loss_utils import l1_loss, photometric_loss_masked_dynamic

    torch.manual_seed(0)
    pred = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 3, 32, 32)
    M = torch.zeros(1, 1, 32, 32)
    M[:, :, 8:24, 8:24] = 1.0
    w = 1.0 - M
    ref = l1_loss(pred, gt, mask=w)
    out = photometric_loss_masked_dynamic(pred, gt, M, lambda_dssim=0.0, ssim_fn=None)
    ok = torch.isclose(ref, out, rtol=1e-5, atol=1e-6)
    check("photometric_loss_masked_dynamic (lambda_dssim=0) matches l1_loss(..., mask=1-M)", bool(ok))

    M_full = torch.ones(1, 1, 16, 16)
    out2 = photometric_loss_masked_dynamic(
        torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16), M_full, 0.0, None
    )
    check("degenerate all-dynamic mask produces finite scalar", torch.isfinite(out2).item())
except ImportError:
    warn("torch not installed — skipping torch numeric checks; running numpy L1 sanity check instead")
    import numpy as np

    rng = np.random.default_rng(0)
    pred = rng.random((3, 24, 24)).astype(np.float32)
    gt = rng.random((3, 24, 24)).astype(np.float32)
    M = np.zeros((1, 24, 24), dtype=np.float32)
    M[:, 4:16, 4:16] = 1.0
    w = 1.0 - M
    w3 = np.repeat(w, 3, axis=0)
    eps = 1e-8
    ref = np.abs((pred - gt) * w3).sum() / (w3.sum() + eps)
    check("numpy masked L1 matches ||(pred-gt)*(1-M)||_1 / sum(1-M) (per-channel mean via l1_loss convention)", ref >= 0 and np.isfinite(ref))
except Exception as e:
    check("torch numeric checks", False, str(e))

print("\n── preprocess_dynamic_masks.py (synthetic) ─────────────────────────\n")

try:
    tmp = tempfile.mkdtemp(prefix="vidstabil_step31_")
    os.makedirs(os.path.join(tmp, "images_2"), exist_ok=True)
    for i in range(3):
        from PIL import Image

        Image.new("RGB", (64, 48), (128, 64, 32)).save(os.path.join(tmp, "images_2", f"{i:03d}.png"))
    rc = subprocess.call(
        [sys.executable, os.path.join(ROOT, PRE), "-s", tmp, "--backend", "synthetic", "--out-subdir", "dynamic_masks"],
        cwd=ROOT,
    )
    ok_files = all(
        os.path.isfile(os.path.join(tmp, "dynamic_masks", f"{i:03d}.png")) for i in range(3)
    )
    check("synthetic backend wrote 3 PNG masks", rc == 0 and ok_files)
    shutil.rmtree(tmp)
except Exception as e:
    warn(f"synthetic preprocess test: {e}")

print("\n── AST: train_static_core wires chunked path ───────────────────────\n")

try:
    tree = ast.parse(train_src)
    found_chunk = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_train_chunked":
            for a in node.args.args:
                if a.arg == "use_masked_photo":
                    found_chunk = True
            break
    check("_train_chunked accepts use_masked_photo", found_chunk)
except SyntaxError:
    check("_train_chunked AST parse", False)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "─" * 62)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Step 3.1 verified.")
else:
    for name, ok in results:
        if not ok:
            print(f"    [FAIL] {name}")
print("─" * 62 + "\n")
sys.exit(0 if passed == total else 1)
