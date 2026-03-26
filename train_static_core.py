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
from scene.gaussian_model import controlgaussians  # STEP2.3
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


def build_chunk_indices(total_frames, chunk_size, overlap):  # STEP2.1
    """
    Returns list of (start, end) tuples (end is exclusive) covering
    [0, total_frames) with the requested overlap.
    The last chunk always ends at total_frames (may be smaller than chunk_size).
    Example: total=150, size=70, overlap=20
      -> [(0,70), (50,120), (100,150)]
    """  # STEP2.1
    step = chunk_size - overlap  # STEP2.1
    chunks = []  # STEP2.1
    start = 0  # STEP2.1
    while start < total_frames:  # STEP2.1
        end = min(start + chunk_size, total_frames)  # STEP2.1
        chunks.append((start, end))  # STEP2.1
        if end == total_frames:  # STEP2.1
            break  # STEP2.1
        start += step  # STEP2.1
    return chunks  # STEP2.1


def _train_chunked(  # STEP2.1
    chunk_list,         # STEP2.1 list of (c_start, c_end) tuples
    cam_spline,         # STEP2.1 global CameraSpline (shared, updated by all chunks)
    pose_optimizer,     # STEP2.1 global optimizer (pose_network + cam_spline params)
    all_train_cameras,  # STEP2.1 full list of Camera objects for all frames
    pose_holder,        # STEP2.1 GaussianModel carrying _posenet / focal_bias
    dataset,            # STEP2.1 ModelParams (used to construct per-chunk Scene)
    opt,                # STEP2.1 OptimizationParams
    background,         # STEP2.1 background color tensor
    T_ref_fov,          # STEP2.1 frozen low-frequency translation reference [N, 3]
    model_path,         # STEP2.1 output directory for per-chunk checkpoints
    scene,              # STEP2.3 global Scene (provides cameras_extent for densification)
):  # STEP2.1
    """Per-chunk windowed training for long videos (total_frames > CHUNK_THRESHOLD)."""  # STEP2.1
    lambda_dssim = 0.2  # STEP2.1
    w_smooth  = 1e-1    # STEP2.1
    w_jitter  = 5e-1    # STEP2.1
    w_fov     = 5e-2    # STEP2.1
    w_dilated = 1e-1    # STEP2.1
    CHUNK_WARMUP = 500  # STEP2.1 — shorter warm-up per chunk

    global_iteration = 0  # STEP2.1 — cross-chunk iteration counter
    iters_per_chunk = opt.iterations // len(chunk_list)  # STEP2.1

    for chunk_idx, (c_start, c_end) in enumerate(chunk_list):  # STEP2.1
        chunk_n_frames = c_end - c_start  # STEP2.1

        # 4a. Per-chunk GaussianModel — Scene initialises it from pcd; frame_range
        #     not supported so we filter cameras manually.
        chunk_gaussians = GaussianModel(dataset)  # STEP2.1
        _chunk_scene = Scene(dataset, chunk_gaussians, chunk_gaussians,  # STEP2.1
                             load_coarse=None)                           # STEP2.1
        chunk_cameras = [cam for cam in all_train_cameras  # STEP2.1
                         if c_start <= cam.uid < c_end]    # STEP2.1
        chunk_gaussians.training_setup(opt, stage="fine_static")  # STEP2.1
        chunk_optimizer = chunk_gaussians.optimizer  # STEP2.1
        chunk_flag_s = 0  # STEP2.3 — per-chunk densification counter (resets each chunk)

        # 4b. Warm-start from global scene skipped (GaussianModel has no clone);
        #     TODO: copy stat_gaussians state when a clone/copy utility is added.

        # 4c. Freeze spline for this chunk's warm-up phase.
        for p in cam_spline.parameters():  # STEP2.1
            p.requires_grad_(False)        # STEP2.1 chunk warm-up

        # 4e. Per-chunk training loop.
        for local_iter in range(iters_per_chunk):  # STEP2.1
            global_iteration += 1  # STEP2.1
            chunk_gaussians.update_learning_rate(local_iter + 1)  # STEP2.1

            if local_iter % 1000 == 0 and local_iter > 2000:  # STEP2.1
                chunk_gaussians.oneupSHdegree()  # STEP2.1

            # Stage gate (chunk-local): unfreeze spline after CHUNK_WARMUP iters.
            if local_iter == CHUNK_WARMUP:  # STEP2.1
                for p in cam_spline.parameters():  # STEP2.1
                    p.requires_grad_(True)          # STEP2.1 unfreeze spline for chunk
                print(f"[STEP2.1] Chunk {chunk_idx}: spline unfrozen "  # STEP2.1
                      f"at local_iter {local_iter}")                     # STEP2.1

            # Sample a camera from this chunk's frame range.
            viewpoint_cam = random.choice(chunk_cameras)  # STEP2.1
            cam_id = viewpoint_cam.uid  # STEP2.1 absolute frame index in [c_start, c_end)

            gt_image = viewpoint_cam.original_image.cuda()  # STEP2.1
            focal = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP2.1
            set_camera_pose_from_spline(viewpoint_cam, cam_spline, focal, cam_id)  # STEP2.1

            render_pkg = render_static(             # STEP2.1
                viewpoint_cam=viewpoint_cam,        # STEP2.1
                stat_pc=chunk_gaussians,            # STEP2.1
                dyn_pc=chunk_gaussians,             # STEP2.1
                bg_color=background,                # STEP2.1
                get_static=True,                    # STEP2.1
            )                                       # STEP2.1
            pred_image = render_pkg["render"]       # STEP2.1

            # Photometric loss — identical formula to single-scene path.
            ll1 = l1_loss(pred_image, gt_image[:3, :, :])                              # STEP2.1
            ssim_loss = ssim(pred_image, gt_image) if lambda_dssim != 0 else 0.0       # STEP2.1
            loss = ll1 + lambda_dssim * (1.0 - ssim_loss)                              # STEP2.1

            # Stability losses — same terms, weights, and frequencies as single-scene
            # path; active after the chunk warm-up gate (local_iter >= CHUNK_WARMUP).
            if local_iter >= CHUNK_WARMUP:  # STEP2.1

                # L_smooth: (1/N) sum_t || d²T/dt²(t) ||² over this chunk's frames.
                acc_smooth = torch.zeros(  # STEP2.1
                    (), device=cam_spline.ctrl_trans.device,  # STEP2.1
                    dtype=cam_spline.ctrl_trans.dtype)        # STEP2.1
                for t_idx in range(c_start, c_end):  # STEP2.1
                    d2T = cam_spline.get_translation_second_derivative(float(t_idx))  # STEP2.1
                    acc_smooth = acc_smooth + (d2T * d2T).sum()                       # STEP2.1
                loss_smooth = acc_smooth / float(chunk_n_frames)  # STEP2.1
                loss = loss + w_smooth * loss_smooth              # STEP2.1

                # L_fov: (1/N) sum_t || T(t) - T_bar(t) ||² over this chunk's frames.
                Tbar_chunk = T_ref_fov[c_start:c_end].to(  # STEP2.1
                    device=cam_spline.ctrl_trans.device,    # STEP2.1
                    dtype=cam_spline.ctrl_trans.dtype)      # STEP2.1
                acc_fov = torch.zeros(  # STEP2.1
                    (), device=cam_spline.ctrl_trans.device,  # STEP2.1
                    dtype=cam_spline.ctrl_trans.dtype)        # STEP2.1
                for t_idx in range(c_start, c_end):  # STEP2.1
                    _, T_cur = cam_spline.get_pose(float(t_idx))  # STEP2.1
                    diff = T_cur - Tbar_chunk[t_idx - c_start]   # STEP2.1
                    acc_fov = acc_fov + (diff * diff).sum()       # STEP2.1
                loss_fov = acc_fov / float(chunk_n_frames)  # STEP2.1
                loss = loss + w_fov * loss_fov              # STEP2.1

                # L_jitter: pixel-diff or RAFT+Laplacian; expensive — every 10 iters.
                if chunk_n_frames >= 2 and local_iter % 10 == 0:  # STEP2.1
                    t_pair = randint(0, chunk_n_frames - 2)  # STEP2.1
                    vc      = chunk_cameras[t_pair]          # STEP2.1
                    vc_next = chunk_cameras[t_pair + 1]      # STEP2.1
                    focal_np = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP2.1
                    set_camera_pose_from_spline(vc, cam_spline, focal_np, vc.uid)            # STEP2.1
                    I0 = render_static(              # STEP2.1
                        viewpoint_cam=vc,            # STEP2.1
                        stat_pc=chunk_gaussians,     # STEP2.1
                        dyn_pc=chunk_gaussians,      # STEP2.1
                        bg_color=background,         # STEP2.1
                        get_static=True,             # STEP2.1
                    )["render"]                      # STEP2.1
                    set_camera_pose_from_spline(vc_next, cam_spline, focal_np, vc_next.uid)  # STEP2.1
                    I1 = render_static(              # STEP2.1
                        viewpoint_cam=vc_next,       # STEP2.1
                        stat_pc=chunk_gaussians,     # STEP2.1
                        dyn_pc=chunk_gaussians,      # STEP2.1
                        bg_color=background,         # STEP2.1
                        get_static=True,             # STEP2.1
                    )["render"]                      # STEP2.1
                    if global_iteration < 7000:  # STEP2.1 pixel-difference proxy
                        loss_jitter = loss_jitter_pixel_diff(I0, I1)              # STEP2.1
                    else:                        # STEP2.1 RAFT flow + Laplacian
                        loss_jitter = loss_jitter_raft_laplacian(I0, I1, I0.device)  # STEP2.1
                    loss = loss + w_jitter * loss_jitter  # STEP2.1

                # L_dilated: co-visible Gaussians at (t, t+k); k=5; every 5 iters.
                dilated_k = 5  # STEP2.1
                if chunk_n_frames > dilated_k and local_iter % 5 == 0:  # STEP2.1
                    t0  = randint(0, chunk_n_frames - dilated_k - 1)  # STEP2.1
                    t1  = t0 + dilated_k                               # STEP2.1
                    vc0 = chunk_cameras[t0]                            # STEP2.1
                    vc1 = chunk_cameras[t1]                            # STEP2.1
                    focal_np = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # STEP2.1

                    set_camera_pose_from_spline(vc0, cam_spline, focal_np, vc0.uid)  # STEP2.1
                    pkg0 = render_static(          # STEP2.1
                        viewpoint_cam=vc0,         # STEP2.1
                        stat_pc=chunk_gaussians,   # STEP2.1
                        dyn_pc=chunk_gaussians,    # STEP2.1
                        bg_color=background,       # STEP2.1
                        get_static=True,           # STEP2.1
                    )                              # STEP2.1
                    set_camera_pose_from_spline(vc1, cam_spline, focal_np, vc1.uid)  # STEP2.1
                    pkg1 = render_static(          # STEP2.1
                        viewpoint_cam=vc1,         # STEP2.1
                        stat_pc=chunk_gaussians,   # STEP2.1
                        dyn_pc=chunk_gaussians,    # STEP2.1
                        bg_color=background,       # STEP2.1
                        get_static=True,           # STEP2.1
                    )                              # STEP2.1

                    vis0  = pkg0["visibility_filter"]  # STEP2.1
                    vis1  = pkg1["visibility_filter"]  # STEP2.1
                    covis = vis0 & vis1                # STEP2.1 V(t) ∩ V(t+k)

                    if torch.any(covis):  # STEP2.1
                        xyz_world  = chunk_gaussians.get_xyz                                       # STEP2.1
                        mu_t       = world_to_camera_points(xyz_world, vc0.world_view_transform)   # STEP2.1
                        mu_tk      = world_to_camera_points(xyz_world, vc1.world_view_transform)   # STEP2.1
                        pos_term   = ((mu_t[covis] - mu_tk[covis]) ** 2).sum(dim=1).mean()        # STEP2.1
                        alpha      = chunk_gaussians.get_opacity.squeeze(-1)                       # STEP2.1
                        alpha_term = ((alpha[covis] - alpha[covis]) ** 2).mean()                  # STEP2.1
                        loss_dilated = pos_term + alpha_term                                       # STEP2.1
                    else:  # STEP2.1
                        loss_dilated = torch.zeros(  # STEP2.1
                            (), device=cam_spline.ctrl_trans.device,  # STEP2.1
                            dtype=cam_spline.ctrl_trans.dtype)        # STEP2.1
                    loss = loss + w_dilated * loss_dilated  # STEP2.1

            loss.backward()  # STEP2.1

            if torch.isnan(loss).any():  # STEP2.1
                raise RuntimeError(  # STEP2.1
                    f"[STEP2.1] NaN in loss at chunk {chunk_idx}, "  # STEP2.1
                    f"local_iter {local_iter}")                       # STEP2.1

            chunk_optimizer.step()                           # STEP2.1
            chunk_optimizer.zero_grad(set_to_none=True)     # STEP2.1
            if local_iter % 2 == 0:                         # STEP2.2 — spline steps every 2nd iter
                pose_optimizer.step()                        # STEP2.2
                pose_optimizer.zero_grad(set_to_none=True)  # STEP2.2

            # STEP2.3 — densification + hard Gaussian cap for this chunk
            with torch.no_grad():  # STEP2.3
                if local_iter < opt.densify_until_iter:                          # STEP2.3
                    c_vis  = render_pkg["visibility_filter"]                     # STEP2.3
                    c_rad  = render_pkg["radii"]                                 # STEP2.3
                    chunk_gaussians.max_radii2D[c_vis] = torch.max(              # STEP2.3
                        chunk_gaussians.max_radii2D[c_vis], c_rad[c_vis])        # STEP2.3
                    c_vpts = render_pkg["viewspace_points"]                      # STEP2.3
                    if c_vpts.absgrad is not None:                               # STEP2.3
                        c_grad = c_vpts.absgrad.squeeze(0)                       # STEP2.3 [N, 2]
                        c_grad = c_grad * torch.tensor(                          # STEP2.3
                            [viewpoint_cam.image_width * 0.5,                   # STEP2.3
                             viewpoint_cam.image_height * 0.5],                 # STEP2.3
                            device=c_grad.device)                               # STEP2.3
                        chunk_gaussians.add_densification_stats(                 # STEP2.3
                            c_grad, c_vis)                                       # STEP2.3
                    if (local_iter > opt.densify_from_iter                       # STEP2.3
                            and local_iter % opt.densification_interval == 0):  # STEP2.3
                        chunk_flag_s = controlgaussians(                         # STEP2.3
                            opt, chunk_gaussians, opt.densify,                   # STEP2.3
                            local_iter, scene, chunk_flag_s)                     # STEP2.3
                        # STEP2.3 — hard cap at MAX_GAUSSIANS
                        if chunk_gaussians.get_xyz.shape[0] > MAX_GAUSSIANS:    # STEP2.3
                            chunk_gaussians.prune_points(                        # STEP2.3
                                chunk_gaussians.get_opacity.squeeze() <         # STEP2.3
                                chunk_gaussians.get_opacity.squeeze()           # STEP2.3
                                .topk(MAX_GAUSSIANS).values.min()              # STEP2.3
                            )  # STEP2.3 keep only top-MAX_GAUSSIANS by opacity

        print(f"[STEP2.1] Chunk {chunk_idx} done "          # STEP2.1
              f"(frames {c_start}–{c_end - 1})")             # STEP2.1

        # Save per-chunk checkpoint (optional; appends chunk_idx to path).
        if model_path:  # STEP2.1
            chunk_ply_path = os.path.join(  # STEP2.1
                model_path, "point_cloud",  # STEP2.1
                f"chunk_{chunk_idx:02d}_final")              # STEP2.1
            os.makedirs(chunk_ply_path, exist_ok=True)       # STEP2.1
            chunk_gaussians.save_ply(                        # STEP2.1
                os.path.join(chunk_ply_path,                 # STEP2.1
                             "point_cloud_static.ply"))      # STEP2.1


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

    # STEP2.1 — chunked windowed path constants
    CHUNK_THRESHOLD = 150    # STEP2.1 use chunked path if total_frames > this
    CHUNK_SIZE      = 70     # STEP2.1 frames per chunk
    OVERLAP         = 20     # STEP2.1 overlap between adjacent chunks
    MAX_GAUSSIANS   = 500_000  # STEP2.3 — Gaussian count cap

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

    # STEP2.1 — branch: long videos use chunked windowed path; short videos run
    # the original single-scene loop unchanged (else-branch below).
    use_chunked = (total_frames > CHUNK_THRESHOLD)  # STEP2.1

    if use_chunked:  # STEP2.1 — long video path
        chunk_list = build_chunk_indices(total_frames, CHUNK_SIZE, OVERLAP)  # STEP2.1
        print(f"[STEP2.1] Chunked mode: {len(chunk_list)} chunks, "          # STEP2.1
              f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}")                  # STEP2.1
        _train_chunked(                # STEP2.1
            chunk_list,                # STEP2.1
            cam_spline,                # STEP2.1
            pose_optimizer,            # STEP2.1
            train_cams,                # STEP2.1
            pose_holder,               # STEP2.1
            dataset,                   # STEP2.1
            opt,                       # STEP2.1
            background,                # STEP2.1
            T_ref_fov,                 # STEP2.1
            args.model_path,           # STEP2.1
            scene,                     # STEP2.3
        )                              # STEP2.1
    else:  # STEP2.1 — short video, original single-scene path unchanged
        viewpoint_stack_ids = []
        progress_bar = tqdm(range(1, opt.iterations + 1), desc="Static core training")
        iteration = 0  # STEP1.3
        flag_s = 0      # STEP2.3 — densification counter for single-scene path

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
                print("[STEP1.3] Main stage: spline unfrozen at iteration 2000 "
                      "| spline steps every 2nd iter (STEP2.2)")  # STEP2.2

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
                if total_frames >= 2 and iteration % 10 == 0:  # STEP2.2 confirmed — already correct per spec
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
                if total_frames > dilated_k and iteration % 5 == 0:  # STEP2.2 confirmed — already correct per spec
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

            stat_gaussians.optimizer.step()                      # unchanged — every iter
            stat_gaussians.optimizer.zero_grad(set_to_none=True)  # unchanged
            if iteration % 2 == 0:                               # STEP2.2 — spline steps every 2nd iter
                pose_optimizer.step()                             # STEP2.2
                pose_optimizer.zero_grad(set_to_none=True)       # STEP2.2

            # STEP2.3 — densification + hard Gaussian cap
            with torch.no_grad():  # STEP2.3
                if iteration < opt.densify_until_iter:  # STEP2.3
                    visibility_filter = render_pkg["visibility_filter"]  # STEP2.3
                    radii             = render_pkg["radii"]               # STEP2.3
                    stat_gaussians.max_radii2D[visibility_filter] = torch.max(  # STEP2.3
                        stat_gaussians.max_radii2D[visibility_filter],           # STEP2.3
                        radii[visibility_filter],                                # STEP2.3
                    )                                                            # STEP2.3
                    viewspace_pts = render_pkg["viewspace_points"]               # STEP2.3
                    if viewspace_pts.absgrad is not None:                        # STEP2.3
                        vpt_grad = viewspace_pts.absgrad.squeeze(0)              # STEP2.3 [N, 2]
                        vpt_grad = vpt_grad * torch.tensor(                      # STEP2.3
                            [viewpoint_cam.image_width * 0.5,                   # STEP2.3
                             viewpoint_cam.image_height * 0.5],                 # STEP2.3
                            device=vpt_grad.device)                             # STEP2.3
                        stat_gaussians.add_densification_stats(                  # STEP2.3
                            vpt_grad, visibility_filter)                         # STEP2.3
                    if (iteration > opt.densify_from_iter                        # STEP2.3
                            and iteration % opt.densification_interval == 0):   # STEP2.3
                        flag_s = controlgaussians(                               # STEP2.3
                            opt, stat_gaussians, opt.densify,                    # STEP2.3
                            iteration, scene, flag_s)                            # STEP2.3
                        # STEP2.3 — hard cap at MAX_GAUSSIANS
                        if stat_gaussians.get_xyz.shape[0] > MAX_GAUSSIANS:     # STEP2.3
                            stat_gaussians.prune_points(                         # STEP2.3
                                stat_gaussians.get_opacity.squeeze() <           # STEP2.3
                                stat_gaussians.get_opacity.squeeze()             # STEP2.3
                                .topk(MAX_GAUSSIANS).values.min()               # STEP2.3
                            )  # STEP2.3 keep only top-MAX_GAUSSIANS by opacity

            n_gauss = stat_gaussians.get_xyz.shape[0]  # STEP2.3
            current_psnr = psnr(pred_image, gt_image).detach().mean().item()
            stage = "warmup" if iteration < 2000 else "main"  # STEP1.3
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.6f}",  # STEP1.3
                    "psnr": f"{current_psnr:.2f}",
                    "n_gauss": n_gauss,                     # STEP2.3
                    "stage": stage,                         # STEP1.3
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
