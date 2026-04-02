
""" Code borrowed from 
https://github.com/vye16/shape-of-motion/blob/main/preproc/compute_tracks_torch.py
"""
import argparse
import glob
import os

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cotracker.utils.visualizer import Visualizer

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path)).transpose(2, 0, 1)[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def read_mask(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate(
        [np.array(Image.open(frame_path))[None, None, None] for frame_path in frame_paths],
        axis=1,
    )
    video = torch.from_numpy(video).float()
    return video


def maybe_resize_video_and_masks(video, masks, max_hw=None):
    if max_hw is None or max_hw <= 0:
        return video, masks, 1.0, 1.0

    _, _, _, height, width = video.shape
    longest = max(height, width)
    if longest <= max_hw:
        return video, masks, 1.0, 1.0

    scale = max_hw / float(longest)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    video_rs = F.interpolate(
        video.reshape(-1, video.shape[2], height, width),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).reshape(video.shape[0], video.shape[1], video.shape[2], new_h, new_w)
    masks_rs = F.interpolate(
        masks.reshape(-1, 1, height, width),
        size=(new_h, new_w),
        mode="nearest",
    ).reshape(masks.shape[0], masks.shape[1], 1, new_h, new_w)
    return video_rs, masks_rs, width / float(new_w), height / float(new_h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--max_hw",
        type=int,
        default=768,
        help="Downscale longest image side before tracking for speed/memory; 0 disables (default: 768)",
    )
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    print(f"{done}")
    if done:
        print("Already done")
        return

    ## Load model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEFAULT_DEVICE)
    video = read_video(folder_path).to(DEFAULT_DEVICE)
    
    masks = read_mask(mask_dir).to(DEFAULT_DEVICE)
    
    masks[masks>0] = 1.
    if args.is_static:
        masks = 1.0 - masks

    video, masks, scale_x, scale_y = maybe_resize_video_and_masks(video, masks, args.max_hw)
    _, num_frames,_, height, width = video.shape
    print(f"Tracking resolution: {width}x{height} | scale back: x={scale_x:.4f}, y={scale_y:.4f}")
    vis = Visualizer(save_dir=os.path.join(out_dir, "vis"), pad_value=120, linewidth=3)

    for t in tqdm(range(num_frames), desc="query frames"):
        name_t = os.path.splitext(frame_names[t])[0]
        file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
        if len(file_matches) == num_frames:
            print(f"Already computed tracks with query {t} {name_t}")
            continue

        current_mask = masks[:, t]
        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video,
                grid_size=args.grid_size,
                grid_query_frame=t,
                backward_tracking=True,
                segm_mask=current_mask,
            )

        pred_tracks[..., 0] *= scale_x
        pred_tracks[..., 1] *= scale_y
        pred = torch.cat([pred_tracks, pred_visibility.unsqueeze(-1)], dim=-1)

        for j in range(num_frames):
            name_j = os.path.splitext(frame_names[j])[0]
            current_pred = pred[0, j]
            np.save(f"{out_dir}/{name_t}_{name_j}.npy", current_pred.cpu().numpy())



if __name__ == "__main__":
    main()
