import os
import random
import sys
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_static
from scene import GaussianModel, Scene
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def prepare_output_dir(expname):
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def train_static_core(dataset, hyper, opt):
    stat_gaussians = GaussianModel(dataset)
    pose_holder = GaussianModel(dataset)
    dataset.model_path = args.model_path
    scene = Scene(dataset, pose_holder, stat_gaussians, load_coarse=None)

    # Keep COLMAP-free pose initialization.
    pose_holder.create_pose_network(hyper, scene.getTrainCameras())
    stat_gaussians.training_setup(opt, stage="fine_static")

    # Separate optimizer for pose network.
    pose_params = list(pose_holder._posenet.get_mlp_parameters())
    pose_params += list(pose_holder._posenet.get_focal_parameters())
    pose_optimizer = torch.optim.Adam(pose_params, lr=opt.pose_lr_init, eps=1e-15)

    bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = [i for i in scene.getTrainCameras()]
    viewpoint_stack_ids = []
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Static core training")

    # Camera rays are required for rgbdecoder and update_cam.
    pixels = train_cams[0].metadata.get_pixels(normalize=True)
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))

    for iteration in progress_bar:
        stat_gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0 and iteration > 2000:
            stat_gaussians.oneupSHdegree()

        if not viewpoint_stack_ids:
            viewpoint_stack_ids = list(range(len(train_cams)))
        cam_id = viewpoint_stack_ids.pop(randint(0, len(viewpoint_stack_ids) - 1))
        viewpoint_cam = train_cams[cam_id]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.depth[None].cuda()
        depth_in = gt_depth.view(-1, 1)
        time_in = torch.tensor(viewpoint_cam.time).float().cuda().view(1, 1)

        pred_R, pred_T, _ = pose_holder._posenet(time_in, depth=depth_in)
        focal = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()

        y = (pixels[..., 1] - viewpoint_cam.metadata.principal_point_y) / focal
        x = (pixels[..., 0] - viewpoint_cam.metadata.principal_point_x) / focal
        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

        R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
        t_ = pred_T.detach().cpu().numpy()
        viewpoint_cam.update_cam(R_[0], t_[0], local_viewdirs, batch_shape, focal)

        render_pkg = render_static(
            viewpoint_cam=viewpoint_cam,
            stat_pc=stat_gaussians,
            dyn_pc=stat_gaussians,
            bg_color=background,
            get_static=True,
        )
        pred_image = render_pkg["render"]

        # Basic photometric loss (L1 + optional DSSIM).
        ll1 = l1_loss(pred_image, gt_image[:3, :, :])
        ssim_loss = ssim(pred_image, gt_image) if opt.lambda_dssim != 0 else 0.0
        photo_loss = ll1 + opt.lambda_dssim * (1.0 - ssim_loss)

        photo_loss.backward()

        if torch.isnan(photo_loss).any():
            raise RuntimeError("NaN in loss; stopping training.")

        stat_gaussians.optimizer.step()
        stat_gaussians.optimizer.zero_grad(set_to_none=True)
        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)

        current_psnr = psnr(pred_image, gt_image).detach().mean().item()
        progress_bar.set_postfix(
            {
                "photo_loss": f"{photo_loss.detach().item():.6f}",
                "psnr": f"{current_psnr:.2f}",
                "num_gaussians": stat_gaussians.get_xyz.shape[0],
                "focal": f"{pose_holder._posenet.focal_bias.exp().detach().item():.2f}",
            }
        )

    # Save as static-only output checkpoint.
    point_cloud_path = os.path.join(args.model_path, "point_cloud", "static_core_final")
    os.makedirs(point_cloud_path, exist_ok=True)
    stat_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_static.ply"))
    torch.save(pose_holder._posenet.state_dict(), os.path.join(point_cloud_path, "posenet.pth"))

    # Write pose trajectory for quick sanity checks.
    trajectory = []
    for cam in train_cams:
        gt_Rt = getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
        trajectory.append(np.linalg.inv(gt_Rt))
    np.save(os.path.join(point_cloud_path, "train_cam_c2w_gt.npy"), np.stack(trajectory, axis=0))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Static-core 3DGS training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expname", type=str, default="static_core")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet, seed=args.seed)
    setup_seed(args.seed)
    torch.set_num_threads(16)
    prepare_output_dir(args.expname)

    train_static_core(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
    )
    print("Static-core training complete.")
