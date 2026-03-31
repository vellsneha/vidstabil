import argparse
import numpy as np
import torch
from PIL import Image

from submodules.UniDepth.unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from submodules.UniDepth.unidepth.utils import colorize, image_grid

import glob
import os

def gen_depth(model, args):
    focal = 500 # default 500 for all datasets

    images_list = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    os.makedirs(args.out_dir, exist_ok=True)
    for image in images_list:
        rgb = np.array(Image.open(image).convert('RGB'))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        
        H, W = rgb_torch.shape[1], rgb_torch.shape[2]
        intrinsics_torch = torch.from_numpy(np.array([[focal, 0, H/2],
                                                    [0, focal, W/2],
                                                    [0, 0 ,1]])).float()
        # predict
        predictions = model.infer(rgb_torch, intrinsics_torch)
        depth_pred = predictions["depth"].squeeze().cpu().numpy()[..., None]
        
        fname = os.path.basename(image)
        np.save(os.path.join(args.out_dir, fname.replace('png', 'npy')), depth_pred)

        # # colorize
        depth_pred_col = colorize(depth_pred)
        Image.fromarray(depth_pred_col).save(os.path.join(args.out_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--depth_type", type=str, help="depth type disp or depth", default="depth")  
    parser.add_argument(
        "--depth_model",
        type=str,
        help="unidepth model (v2 recommended; v2old may require xformers)",
        default="v2",
    )
    args = parser.parse_args()
    
    print("Torch version:", torch.__version__)
    
    if args.depth_type == "disp":  
        os.makedirs(args.out_dir, exist_ok=True)     
        cmd = f"python submodules/mega-sam/Depth-Anything/run_videos.py --encoder vitl \
        --load-from submodules/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
        --img-path {args.image_dir} \
        --outdir {args.out_dir}"
        os.system(cmd)
    elif args.depth_type == "depth":
        type_ = "l"  # available types: s, b, l
        name = f"unidepth-{args.depth_model}-vit{type_}14"
        if args.depth_model == "v2":
            model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
            model.interpolation_mode = "bilinear"
        elif args.depth_model == "v2old":
            try:
                model = UniDepthV2old.from_pretrained(f"lpiccinelli/{name}")
            except Exception as e:
                raise
        else:
            raise ValueError("--depth_model must be one of: v2, v2old")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        try:
            gen_depth(model, args)
        except NotImplementedError as e:
            msg = str(e)
            if "NystromAttention" in msg or "xformers" in msg:
                raise SystemExit(
                    "UniDepth depth inference failed because xformers/NystromAttention is unavailable.\n"
                    "Fix options:\n"
                    "  - Use UniDepth v2: pass --depth_model v2 (recommended)\n"
                    "  - Or install xformers compatible with your CUDA/PyTorch\n"
                ) from e
            raise
    else:
        raise ValueError("depth_type must be either 'disp' or 'depth'")

