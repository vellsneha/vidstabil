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

# ========================
# CONFIG — change paths
# ========================
SRC_FRAMES = "/workspace/vidstabil/data/test_clip/images"  # your ffmpeg PNGs
SCENE = "/workspace/vidstabil/data/regular_scene"          # dataset folder to create

# ========================
# Step 1: create folder structure
# ========================
os.makedirs(os.path.join(SCENE, "images_2"), exist_ok=True)
os.makedirs(os.path.join(SCENE, "gt"), exist_ok=True)
os.makedirs(os.path.join(SCENE, "uni_depth"), exist_ok=True)
os.makedirs(os.path.join(SCENE, "instance_mask"), exist_ok=True)
os.makedirs(os.path.join(SCENE, "bootscotracker_dynamic"), exist_ok=True)
os.makedirs(os.path.join(SCENE, "bootscotracker_static"), exist_ok=True)

# ========================
# Step 2: copy & rename frames -> images_2/000.png
# ========================
frame_paths = sorted(glob.glob(os.path.join(SRC_FRAMES, "*.png")))
assert frame_paths, f"No PNGs found in {SRC_FRAMES}"

for i, p in enumerate(frame_paths):
    dst_path = os.path.join(SCENE, "images_2", f"{i:03d}.png")
    shutil.copy2(p, dst_path)
print(f"Copied {len(frame_paths)} frames to images_2/")

# ========================
# Step 3: copy frames to gt/v000_t###.png
# ========================
for i, p in enumerate(sorted(glob.glob(os.path.join(SCENE, "images_2", "*.png")))):
    dst_path = os.path.join(SCENE, "gt", f"v000_t{i:03d}.png")
    shutil.copy2(p, dst_path)
print(f"Copied {len(frame_paths)} frames to gt/")

# ========================
# Step 4: create dummy instance_mask
# ========================
for i, p in enumerate(sorted(glob.glob(os.path.join(SCENE, "images_2", "*.png")))):
    im = Image.open(p)
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    mask_dir = os.path.join(SCENE, "instance_mask", f"{i:03d}")
    os.makedirs(mask_dir, exist_ok=True)
    mask.save(os.path.join(mask_dir, "000.png"))
print(f"Created dummy instance_mask for {len(frame_paths)} frames")

# ========================
# Step 5: create dummy uni_depth
# ========================
for i, p in enumerate(sorted(glob.glob(os.path.join(SCENE, "images_2", "*.png")))):
    im = Image.open(p)
    w, h = im.size
    depth = np.ones((h, w, 1), dtype=np.float32)
    np.save(os.path.join(SCENE, "uni_depth", f"{i:03d}.npy"), depth)
print(f"Created dummy uni_depth for {len(frame_paths)} frames")

# ========================
# Step 6: create dummy bootscotracker files
# ========================
n = len(frame_paths)
track = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

for q in range(n):
    qn = f"{q:03d}"
    for t in range(n):
        tn = f"{t:03d}"
        np.save(os.path.join(SCENE, "bootscotracker_dynamic", f"{qn}_{tn}.npy"), track)
        np.save(os.path.join(SCENE, "bootscotracker_static", f"{qn}_{tn}.npy"), track)
print(f"Created dummy bootscotracker files for {n} frames (pairs: {n*n})")

print("\nDataset preparation complete!")
print(f"Your dataset is ready at: {SCENE}")