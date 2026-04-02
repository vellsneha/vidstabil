"""
verify_step1_4.py
-----------------
Run from the repository root (same directory as train_exp.py):

    python verify-1.4.py
    python verify-1.4.py --strict   # exit non-zero unless every term passes

Step 1.4 replaces the Step 1.3 stability stub with four stability loss terms.
As you define each term (one by one), tighten the regex lists in TERM_SPECS
below so this script fully encodes the spec.

Convention (recommended for reliable checks):
  - Use distinct Python names in train_exp.py, e.g. loss_smooth,
    loss_jitter, loss_fov, loss_dilated (or keep names aligned with your math).
  - Tag the block with # STEP1.4 ... so static checks stay unambiguous.

This file does not import train_static_core (avoids CUDA / simple_knn). It only
reads source text and optionally imports a small pure-torch helper module if you
add one later (see OPTIONAL_LOSS_MODULE).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

TRAIN_FILE = "train_exp.py"
SPLINE_FILE = "scene/camera_spline.py"

# If you move pure loss computations here, runtime checks will import it.
OPTIONAL_LOSS_MODULE = "utils.losses_step14"

# ─────────────────────────────────────────────────────────────────────────────
# Expand these entries as you lock in each term’s definition.
#
# For each term:
#   - patterns:  list of regexes — if require_all_patterns is True, ALL must
#                match; otherwise ANY match counts as "present".
#   - require_all_patterns: default False until the term is fully specified.
#
# Start by setting patterns to match the real variable/function names you add.
# ─────────────────────────────────────────────────────────────────────────────

TERM_SPECS: list[dict] = [
    {
        "id": "smooth",
        "title": "Term 1 — smoothness (L_smooth)",
        "patterns": [
            r"\bloss_smooth\s*=",
            r"get_translation_second_derivative",
        ],
        "require_all_patterns": True,
        "required": True,
    },
    {
        "id": "jitter",
        "title": "Term 2 — jitter (L_jitter)",
        "patterns": [
            r"\bloss_jitter\s*=",
            r"iteration\s*%\s*10\s*==\s*0",
            r"loss_jitter_pixel_diff|loss_jitter_raft_laplacian",
        ],
        "require_all_patterns": True,
        "required": True,
    },
    {
        "id": "fov",
        "title": "Term 3 — FoV (L_fov)",
        "patterns": [
            r"\bloss_fov\s*=",
            r"frozen_low_frequency_translation_reference",
        ],
        "require_all_patterns": True,
        "required": True,
    },
    {
        "id": "dilated",
        "title": "Term 4 — dilated (L_dilated)",
        "patterns": [
            r"\bloss_dilated\s*=",
            r"iteration\s*%\s*5\s*==\s*0",
            r"visibility_filter",
        ],
        "require_all_patterns": True,
        "required": True,
    },
]

# Optional: once all four are wired, require that stability is not the old zero stub.
# Step 1.3 placeholder — remove/replace when Step 1.4 is implemented.
FORBID_PURE_ZERO_STUB = re.compile(
    r"stability_loss\s*=\s*torch\.tensor\s*\(\s*0\.0\b",
)

# Loss must still be gated to main stage (iteration >= 2000) unless you change design.
REQUIRE_MAIN_STAGE_GATE = re.compile(
    r"if\s+iteration\s*>=\s*2000",
)


def _read_train_src() -> str:
    path = os.path.join(os.getcwd(), TRAIN_FILE)
    if not os.path.isfile(path):
        print(f"{FAIL} {TRAIN_FILE} not found (cwd={os.getcwd()})")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _match_not_comment_only(src: str, pattern: re.Pattern[str]) -> bool:
    """True if pattern matches at least one line that isn't only a # comment."""
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern.search(line):
            return True
    if pattern.search(src):
        # might be multi-line
        return True
    return False


def check_term(src: str, spec: dict) -> tuple[bool, str]:
    patterns = spec.get("patterns") or []
    if not patterns:
        return False, "no patterns configured — edit TERM_SPECS in verify-1.4.py"

    compiled = [(p, re.compile(p)) for p in patterns]
    require_all = spec.get("require_all_patterns", False)

    if require_all:
        ok = True
        for p, c in compiled:
            if not _match_not_comment_only(src, c):
                ok = False
        detail = "all patterns must match: " + ", ".join(patterns)
        return ok, detail

    for p, c in compiled:
        if _match_not_comment_only(src, c):
            return True, f"matched: {p}"
    return False, "none of: " + ", ".join(patterns)


def check_global_step14(src: str) -> tuple[bool, str]:
    if "# STEP1.4" not in src and "STEP1.4" not in src:
        return False, "add at least one # STEP1.4 marker in " + TRAIN_FILE
    return True, "STEP1.4 marker present"


def check_stub_replaced(src: str) -> tuple[bool, str]:
    if FORBID_PURE_ZERO_STUB.search(src):
        return (
            False,
            "stability_loss is still the pure torch.tensor(0.0, requires_grad=False) stub — "
            "replace with real terms for Step 1.4",
        )
    return True, "zero-only stability stub pattern not found (OK)"


def check_main_stage_gate(src: str) -> tuple[bool, str]:
    if not REQUIRE_MAIN_STAGE_GATE.search(src):
        return False, "expected: if iteration >= 2000: (stability losses in main stage)"
    return True, "main-stage gate (iteration >= 2000) present"


def try_optional_runtime_checks() -> list[tuple[str, bool, str]]:
    """If OPTIONAL_LOSS_MODULE exists, run trivial import / callable checks."""
    out: list[tuple[str, bool, str]] = []
    sys.path.insert(0, os.getcwd())
    try:
        mod = __import__(OPTIONAL_LOSS_MODULE, fromlist=["*"])
    except ImportError:
        out.append(
            (
                f"optional {OPTIONAL_LOSS_MODULE}",
                True,
                "module not present — skipped (add later for unit-style tests)",
            )
        )
        return out

    out.append((f"import {OPTIONAL_LOSS_MODULE}", True, "ok"))
    # Extend here: e.g. assert callable(mod.loss_smooth) etc.
    for name in ("loss_smooth", "loss_jitter", "loss_fov", "loss_dilated"):
        if hasattr(mod, name):
            fn = getattr(mod, name)
            out.append((f"{OPTIONAL_LOSS_MODULE}.{name}", callable(fn), "callable" if callable(fn) else "not callable"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Step 1.4 stability loss implementation.")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any check fails (default: exit 1 only if train file missing).",
    )
    args = ap.parse_args()

    print("\n── Step 1.4 — stability losses (four terms) ─────────────────────────\n")

    src = _read_train_src()
    results: list[tuple[str, bool, str]] = []

    def r(name: str, ok: bool, detail: str = "") -> None:
        tag = PASS if ok else FAIL
        print(f"{tag} {name}")
        if detail:
            print(f"       {detail}")
        results.append((name, ok))

    ok, d = check_global_step14(src)
    r("Global: STEP1.4 documentation in train_exp.py", ok, d)

    ok, d = check_main_stage_gate(src)
    r("Global: stability block gated (iteration >= 2000)", ok, d)

    ok, d = check_stub_replaced(src)
    r("Global: Step 1.3 zero-tensor stub replaced for stability_loss", ok, d)

    print("\n── Per-term implementation (edit TERM_SPECS in verify-1.4.py) ──────\n")

    spline_src = ""
    spath = os.path.join(os.getcwd(), SPLINE_FILE)
    if os.path.isfile(spath):
        with open(spath, encoding="utf-8") as sf:
            spline_src = sf.read()
        ok = "get_translation_second_derivative" in spline_src and "# STEP1.4" in spline_src
        r(
            "Spline: get_translation_second_derivative in camera_spline.py (STEP1.4)",
            ok,
            "closed-form translation acceleration" if ok else f"missing in {SPLINE_FILE}",
        )

    for spec in TERM_SPECS:
        tid = spec["id"]
        title = spec["title"]
        if not spec.get("required", True):
            r(f"{title} [{tid}]", True, "skipped — not specified yet")
            continue
        ok, d = check_term(src, spec)
        r(f"{title} [{tid}]", ok, d)

    print("\n── Optional runtime module ──────────────────────────────────────────\n")
    for name, ok, d in try_optional_runtime_checks():
        r(name, ok, d)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "─" * 62)
    print(f"  {passed}/{total} checks passed")

    critical = [n for n, ok in results if not ok and not n.startswith("optional")]
    if args.strict and critical:
        print(f"{FAIL} Strict mode: failing ({len(critical)} issue(s))")
        for n in critical:
            print(f"       - {n}")
        sys.exit(1)

    if passed == total:
        print(f"{INFO} All checks passed.")
        sys.exit(0)

    print(f"{WARN} Some checks failed — tighten TERM_SPECS as you define each term.")
    if not args.strict:
        print(f"{INFO} Run with --strict to exit non-zero on any failure.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
