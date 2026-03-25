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
from utils.fov_loss import frozen_low_frequency_translation_reference  # STEP1.4 L_fov
from utils.jitter_loss import loss_jitter_pixel_diff, loss_jitter_raft_laplacian  # STEP1.4 L_jitter
from utils.loss_utils import l1_loss, ssim


def set_camera_pose_from_spline(viewpoint_cam, cam_spline, focal, frame_idx):  # STEP1.2 STEP1.4
    """Apply `cam_spline` pose for integer frame index (shared by main render & jitter)."""  # STEP1.4
    R, T = cam_spline.get_pose(float(frame_idx))  # STEP1.2
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
    viewpoint_cam.camera_center = torch.inverse(viewpoint_cam.world_view_transform)[3, :3]  # STEP1.2


def world_to_camera_points(xyz_world, world_view_transform):  # STEP1.4 L_dilated
    """Project world points to camera coordinates using current view transform."""  # STEP1.4
    viewmat = world_view_transform.transpose(0, 1)  # STEP1.4 renderer convention
    ones = torch.ones((xyz_world.shape[0], 1), dtype=xyz_world.dtype, device=xyz_world.device)  # STEP1.4
    xyz_h = torch.cat([xyz_world, ones], dim=1)  # STEP1.4 [G,4]
    xyz_cam_h = xyz_h @ viewmat.transpose(0, 1)  # STEP1.4 row-vector batch multiply
    return xyz_cam_h[:, :3]  # STEP1.4


def prepare_output_dir(expname):
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def train_static_core(dataset, hyper, opt):
    lambda_dssim = 0.2  # STEP1.3 L_photo weight for L1 + 0.2*L_SSIM
    # STEP1.4 tuned defaults (from spec slide): lambda1..lambda4
    w_smooth = 1e-1
    w_jitter = 5e-1
    w_fov = 5e-2
    w_dilated = 1e-1
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
    with torch.no_grad():  # STEP1.4 L_fov — frozen low-frequency reference from initial translations
        T_ref_fov = frozen_low_frequency_translation_reference(_Ts_init.cuda().float())  # STEP1.4 [N,3] detached
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
        set_camera_pose_from_spline(viewpoint_cam, cam_spline, focal, cam_id)  # STEP1.2

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

        # STEP1.4 — stability losses (main stage; spline trainable)
        if iteration >= 2000:  # STEP1.3
            # STEP1.4 L_smooth: (1/N) sum_t || d²T/dt²(t) ||² — translation Hermite, closed form
            acc_smooth = torch.zeros((), device=cam_spline.ctrl_trans.device, dtype=cam_spline.ctrl_trans.dtype)
            for t_idx in range(total_frames):  # STEP1.4
                d2T = cam_spline.get_translation_second_derivative(float(t_idx))  # STEP1.4
                acc_smooth = acc_smooth + (d2T * d2T).sum()  # STEP1.4 ||·||² per frame
            loss_smooth = acc_smooth / float(total_frames)  # STEP1.4
            loss = loss + w_smooth * loss_smooth  # STEP1.4

            # STEP1.4 L_fov: (1/N) sum_t || T(t) - T_bar(t) ||^2 — T_bar frozen from smoothed init poses
            Tbar = T_ref_fov.to(device=cam_spline.ctrl_trans.device, dtype=cam_spline.ctrl_trans.dtype)  # STEP1.4
            acc_fov = torch.zeros((), device=cam_spline.ctrl_trans.device, dtype=cam_spline.ctrl_trans.dtype)  # STEP1.4
            for t_idx in range(total_frames):  # STEP1.4
                _, T_cur = cam_spline.get_pose(float(t_idx))  # STEP1.4
                diff = T_cur - Tbar[t_idx]  # STEP1.4 translation difference (same convention as get_pose)
                acc_fov = acc_fov + (diff * diff).sum()  # STEP1.4
            loss_fov = acc_fov / float(total_frames)  # STEP1.4
            loss = loss + w_fov * loss_fov  # STEP1.4

            # STEP1.4 L_jitter: || nabla^2 (I_{t+1}-I_t) ||_F (pixel diff); RAFT+Laplacian after iter 7000
            # Expensive: only every 10 iterations.
            if total_frames >= 2 and iteration % 10 == 0:  # STEP1.4
                t_pair = randint(0, total_frames - 2)  # STEP1.4
                vc = train_cams[t_pair]  # STEP1.4
                focal_np = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP1.4
                set_camera_pose_from_spline(vc, cam_spline, focal_np, t_pair)  # STEP1.4
                I0 = render_static(  # STEP1.4
                    viewpoint_cam=vc,  # STEP1.4
                    stat_pc=stat_gaussians,  # STEP1.4
                    dyn_pc=stat_gaussians,  # STEP1.4
                    bg_color=background,  # STEP1.4
                    get_static=True,  # STEP1.4
                )["render"]  # STEP1.4
                set_camera_pose_from_spline(vc, cam_spline, focal_np, t_pair + 1)  # STEP1.4
                I1 = render_static(  # STEP1.4
                    viewpoint_cam=vc,  # STEP1.4
                    stat_pc=stat_gaussians,  # STEP1.4
                    dyn_pc=stat_gaussians,  # STEP1.4
                    bg_color=background,  # STEP1.4
                    get_static=True,  # STEP1.4
                )["render"]  # STEP1.4
                if iteration < 7000:  # STEP1.4 pixel-difference proxy
                    loss_jitter = loss_jitter_pixel_diff(I0, I1)  # STEP1.4
                else:  # STEP1.4 RAFT flow + Laplacian (RAFT forward no_grad — no grad through flow)
                    loss_jitter = loss_jitter_raft_laplacian(I0, I1, I0.device)  # STEP1.4
                loss = loss + w_jitter * loss_jitter  # STEP1.4

            # STEP1.4 L_dilated: co-visible Gaussians at (t, t+k); k default 5; run every 5 iterations
            dilated_k = 5  # STEP1.4 default window
            if total_frames > dilated_k and iteration % 5 == 0:  # STEP1.4
                t0 = randint(0, total_frames - dilated_k - 1)  # STEP1.4
                t1 = t0 + dilated_k  # STEP1.4
                vc0 = train_cams[t0]  # STEP1.4
                vc1 = train_cams[t1]  # STEP1.4
                focal_np = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP1.4

                set_camera_pose_from_spline(vc0, cam_spline, focal_np, t0)  # STEP1.4
                pkg0 = render_static(  # STEP1.4
                    viewpoint_cam=vc0,
                    stat_pc=stat_gaussians,
                    dyn_pc=stat_gaussians,
                    bg_color=background,
                    get_static=True,
                )
                set_camera_pose_from_spline(vc1, cam_spline, focal_np, t1)  # STEP1.4
                pkg1 = render_static(  # STEP1.4
                    viewpoint_cam=vc1,
                    stat_pc=stat_gaussians,
                    dyn_pc=stat_gaussians,
                    bg_color=background,
                    get_static=True,
                )

                vis0 = pkg0["visibility_filter"]  # STEP1.4 existing rasterizer visibility
                vis1 = pkg1["visibility_filter"]  # STEP1.4 existing rasterizer visibility
                covis = vis0 & vis1  # STEP1.4 V(t) ∩ V(t+k)

                if torch.any(covis):  # STEP1.4
                    xyz_world = stat_gaussians.get_xyz  # STEP1.4
                    mu_t = world_to_camera_points(xyz_world, vc0.world_view_transform)  # STEP1.4
                    mu_tk = world_to_camera_points(xyz_world, vc1.world_view_transform)  # STEP1.4
                    pos_term = ((mu_t[covis] - mu_tk[covis]) ** 2).sum(dim=1).mean()  # STEP1.4 ||mu^t - mu^{t+k}||^2

                    alpha = stat_gaussians.get_opacity.squeeze(-1)  # STEP1.4
                    alpha_term = ((alpha[covis] - alpha[covis]) ** 2).mean()  # STEP1.4 ||alpha^t - alpha^{t+k}||^2
                    loss_dilated = pos_term + alpha_term  # STEP1.4
                else:
                    loss_dilated = torch.zeros((), device=cam_spline.ctrl_trans.device, dtype=cam_spline.ctrl_trans.dtype)  # STEP1.4
                loss = loss + w_dilated * loss_dilated  # STEP1.4

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
