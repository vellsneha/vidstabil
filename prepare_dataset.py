"""
We now need to convert your extracted frames into the dataset format that VidStabil expects:
1. images_2/ → 3-digit PNG frames (000.png, 001.png …)
2. gt/ → copy of images renamed v000_t000.png …
3. uni_depth/ → dummy .npy depth files for each frame
4. instance_mask/ → dummy masks (must exist, at least one per frame)
5. bootscotracker_dynamic/ & bootscotracker_static/ → dummy track files for every pair

So,
1. Take your extracted frames (%05d.png)
2. Rename & copy to images_2/
3. Copy frames to gt/
4. Create dummy instance_mask/
5. Create dummy uni_depth/
6. Create dummy bootscotracker_dynamic/ and bootscotracker_static/

then to train, run this

python train_entrypoint.py -s "$SCENE" --expname regular_fast_run
"""

#!/usr/bin/env python3
import os, glob, shutil
import numpy as np
from PIL import Image
#!/usr/bin/env python3
"""
Converts extracted frames into VidStabil dataset format.
Fixed: Step 6 uses a single shared dummy file instead of n² writes.
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
SRC_FRAMES = "/workspace/vidstabil/data3/test_clip/images"
SCENE      = "/workspace/vidstabil/data3/crowd9_scene"

# ── Step 1: folder structure ──────────────────────────────────────────────────
for d in ["images_2", "gt", "uni_depth", "instance_mask",
          "bootscotracker_dynamic", "bootscotracker_static"]:
    os.makedirs(os.path.join(SCENE, d), exist_ok=True)

# ── Step 2: copy frames → images_2/000.png ────────────────────────────────────
frame_paths = sorted(glob.glob(os.path.join(SRC_FRAMES, "*.png")))
assert frame_paths, f"No PNGs found in {SRC_FRAMES}"
n = len(frame_paths)

for i, p in enumerate(frame_paths):
    shutil.copy2(p, os.path.join(SCENE, "images_2", f"{i:03d}.png"))
print(f"[1/5] Copied {n} frames to images_2/")

# ── Step 3: copy frames → gt/v000_t###.png ───────────────────────────────────
for i in range(n):
    shutil.copy2(
        os.path.join(SCENE, "images_2", f"{i:03d}.png"),
        os.path.join(SCENE, "gt", f"v000_t{i:03d}.png")
    )
print(f"[2/5] Copied {n} frames to gt/")

# ── Step 4: dummy instance_mask ───────────────────────────────────────────────
im0  = Image.open(frame_paths[0])
w, h = im0.size
mask = Image.new("L", (w, h), 0)

# Save one shared mask file, symlink the rest
shared_mask_dir = os.path.join(SCENE, "instance_mask", "000")
os.makedirs(shared_mask_dir, exist_ok=True)
shared_mask_path = os.path.join(shared_mask_dir, "000.png")
mask.save(shared_mask_path)

for i in range(1, n):
    mask_dir = os.path.join(SCENE, "instance_mask", f"{i:03d}")
    os.makedirs(mask_dir, exist_ok=True)
    dst = os.path.join(mask_dir, "000.png")
    if not os.path.exists(dst):
        os.symlink(shared_mask_path, dst)  # symlink, not copy
print(f"[3/5] Created dummy instance_mask for {n} frames")

# ── Step 5: dummy uni_depth ───────────────────────────────────────────────────
depth = np.ones((h, w, 1), dtype=np.float32)
shared_depth = os.path.join(SCENE, "uni_depth", "shared_depth.npy")
np.save(shared_depth, depth)

for i in range(n):
    dst = os.path.join(SCENE, "uni_depth", f"{i:03d}.npy")
    if not os.path.exists(dst):
        os.symlink(shared_depth, dst)  # symlink, not copy
print(f"[4/5] Created dummy uni_depth for {n} frames")

# ── Step 6: dummy bootscotracker — ONE shared file, symlinks for all pairs ────
# Critical fix: original wrote n² files. This writes 1 and symlinks the rest.
track       = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
shared_dyn  = os.path.join(SCENE, "bootscotracker_dynamic", "shared.npy")
shared_stat = os.path.join(SCENE, "bootscotracker_static",  "shared.npy")
np.save(shared_dyn,  track)
np.save(shared_stat, track)

for q in range(n):
    qn = f"{q:03d}"
    for t in range(n):
        tn  = f"{t:03d}"
        dyn = os.path.join(SCENE, "bootscotracker_dynamic", f"{qn}_{tn}.npy")
        sta = os.path.join(SCENE, "bootscotracker_static",  f"{qn}_{tn}.npy")
        if not os.path.exists(dyn):
            os.symlink(shared_dyn,  dyn)
        if not os.path.exists(sta):
            os.symlink(shared_stat, sta)

print(f"[5/5] Created bootscotracker symlinks ({n*n} pairs, 2 real files)")

print(f"\nDone. Dataset ready at: {SCENE}")
print(f"Frames: {n} | Resolution: {w}x{h}")
print(f"\nRun training:")
print(f"  python train_entrypoint.py -s {SCENE} --expname my_run")