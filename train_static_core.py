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
from scene.camera_spline import CameraSpline  # STEP1.2
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2, getWorld2View2_torch  # STEP1.2
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def prepare_output_dir(expname):
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def train_static_core(dataset, hyper, opt):
    lambda_dssim = 0.2  # STEP1.3 L_photo weight for L1 + 0.2*L_SSIM
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

    # STEP1.2 — warm-start: collect initial per-frame R, T from pose_network
    total_frames = len(train_cams)  # STEP1.2
    _Rs_init, _Ts_init = [], []  # STEP1.2
    with torch.no_grad():  # STEP1.2
        for _cam in train_cams:  # STEP1.2
            _gt_depth = _cam.depth[None].cuda()  # STEP1.2
            _depth_in = _gt_depth.view(-1, 1)  # STEP1.2
            _time_in = torch.tensor(_cam.time).float().cuda().view(1, 1)  # STEP1.2
            _pred_R, _pred_T, _ = pose_holder._posenet(_time_in, depth=_depth_in)  # STEP1.2
            # Transpose to match the convention expected by update_cam (torch path)  # STEP1.2
            _Rs_init.append(torch.transpose(_pred_R[0], 1, 0).cpu())  # STEP1.2
            _Ts_init.append(_pred_T[0].cpu())  # STEP1.2
    _Rs_init = torch.stack(_Rs_init)  # [N, 3, 3]  # STEP1.2
    _Ts_init = torch.stack(_Ts_init)  # [N, 3]  # STEP1.2

    cam_spline = CameraSpline(N=total_frames)  # STEP1.2
    cam_spline.initialize_from_poses(_Rs_init, _Ts_init)  # STEP1.2
    cam_spline = cam_spline.cuda()  # STEP1.2
    pose_optimizer.add_param_group(  # STEP1.2
        {"params": list(cam_spline.parameters()), "lr": opt.pose_lr_init}  # STEP1.2
    )  # STEP1.2
    for p in cam_spline.parameters():  # STEP1.3
        p.requires_grad_(False)  # STEP1.3 warm-up: spline frozen

    viewpoint_stack_ids = []
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Static core training")
    iteration = 0  # STEP1.3

    for iteration in progress_bar:
        stat_gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0 and iteration > 2000:
            stat_gaussians.oneupSHdegree()

        if not viewpoint_stack_ids:
            viewpoint_stack_ids = list(range(len(train_cams)))
        cam_id = viewpoint_stack_ids.pop(randint(0, len(viewpoint_stack_ids) - 1))
        viewpoint_cam = train_cams[cam_id]

        gt_image = viewpoint_cam.original_image.cuda()
        focal = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP1.2
        R, T = cam_spline.get_pose(float(cam_id))  # STEP1.2
        # Inline SE(3) camera update — directly sets transforms from spline pose  # STEP1.2
        viewpoint_cam.FoVy = focal2fov(float(focal), viewpoint_cam.image_height)  # STEP1.2
        viewpoint_cam.FoVx = focal2fov(float(focal), viewpoint_cam.image_width)  # STEP1.2
        viewpoint_cam.world_view_transform = getWorld2View2_torch(R, T).transpose(0, 1)  # STEP1.2
        viewpoint_cam.projection_matrix = getProjectionMatrix(  # STEP1.2
            znear=viewpoint_cam.znear, zfar=viewpoint_cam.zfar,  # STEP1.2
            fovX=viewpoint_cam.FoVx, fovY=viewpoint_cam.FoVy,  # STEP1.2
        ).transpose(0, 1).cuda()  # STEP1.2
        viewpoint_cam.full_proj_transform = (  # STEP1.2
            viewpoint_cam.world_view_transform.unsqueeze(0)  # STEP1.2
            .bmm(viewpoint_cam.projection_matrix.unsqueeze(0))  # STEP1.2
        ).squeeze(0)  # STEP1.2
        viewpoint_cam.camera_center = torch.inverse(  # STEP1.2
            viewpoint_cam.world_view_transform  # STEP1.2
        )[3, :3]  # STEP1.2

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
        ssim_loss = ssim(pred_image, gt_image) if lambda_dssim != 0 else 0.0
        photo_loss = ll1 + lambda_dssim * (1.0 - ssim_loss)  # STEP1.3 L_photo = L1 + 0.2*L_SSIM
        loss = photo_loss  # STEP1.3

        # STEP1.3 — stage gate
        if iteration == 2000:  # STEP1.3
            for p in cam_spline.parameters():  # STEP1.3
                p.requires_grad_(True)  # STEP1.3 unfreeze spline
            print("[STEP1.3] Main stage: spline unfrozen at iteration 2000")  # STEP1.3

        # STEP1.3 stub — stability losses injected here in Step 1.4
        if iteration >= 2000:  # STEP1.3
            stability_loss = torch.tensor(0.0, device=pred_image.device, requires_grad=False)  # STEP1.3
            # L_smooth, L_jitter, L_fov, L_dilated will replace this stub  # STEP1.3
            loss = loss + stability_loss  # STEP1.3

        loss.backward()  # STEP1.3

        if torch.isnan(loss).any():  # STEP1.3
            raise RuntimeError("NaN in loss; stopping training.")

        stat_gaussians.optimizer.step()
        stat_gaussians.optimizer.zero_grad(set_to_none=True)
        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)

        current_psnr = psnr(pred_image, gt_image).detach().mean().item()
        stage = "warmup" if iteration < 2000 else "main"  # STEP1.3
        progress_bar.set_postfix(
            {
                "loss": f"{loss.detach().item():.6f}",  # STEP1.3
                "psnr": f"{current_psnr:.2f}",
                "stage": stage,  # STEP1.3
                "num_gaussians": stat_gaussians.get_xyz.shape[0],
                "focal": f"{pose_holder._posenet.focal_bias.exp().detach().item():.2f}",
            }
        )
        iteration += 1  # STEP1.3

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
