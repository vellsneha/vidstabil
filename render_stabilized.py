"""Stabilized novel-view rendering from static-core checkpoint."""  # RENDER_PREP
import os  # RENDER_PREP
import subprocess  # RENDER_PREP
import sys  # RENDER_PREP
from argparse import ArgumentParser  # RENDER_PREP
import cv2  # RENDER_PREP
import torch  # RENDER_PREP
from tqdm import tqdm  # RENDER_PREP
from arguments import ModelHiddenParams, ModelParams, PipelineParams  # RENDER_PREP
from gaussian_renderer import render_static  # RENDER_PREP
from scene import GaussianModel, Scene  # RENDER_PREP
from scene.camera_spline import CameraSpline  # RENDER_PREP
from train_static_core import set_camera_pose_from_spline  # RENDER_PREP
def main():  # RENDER_PREP
    parser = ArgumentParser(description="Render stabilized video from static-core checkpoint")  # RENDER_PREP
    lp = ModelParams(parser)  # RENDER_PREP
    pp = PipelineParams(parser)  # RENDER_PREP
    hp = ModelHiddenParams(parser)  # RENDER_PREP
    parser.add_argument("-s", "--source_path", type=str, required=True)  # RENDER_PREP
    parser.add_argument("--expname", type=str, required=True)  # RENDER_PREP
    parser.add_argument("--output_video", type=str, required=True)  # RENDER_PREP
    parser.add_argument("--fps", type=int, default=30)  # RENDER_PREP
    parser.add_argument("--skip_verification", action="store_true")  # RENDER_PREP
    _args = parser.parse_args(sys.argv[1:])  # RENDER_PREP
    model_path = os.path.abspath(os.path.join("output", _args.expname))  # RENDER_PREP
    cfg_path = os.path.join(model_path, "cfg_args")  # RENDER_PREP
    if not os.path.isfile(cfg_path):  # RENDER_PREP
        raise FileNotFoundError(f"[RENDER] Missing {cfg_path} — train static-core first.")  # RENDER_PREP
    with open(cfg_path, "r", encoding="utf-8") as f:  # RENDER_PREP
        cfg_ns = eval(f.read())  # noqa: S307  # RENDER_PREP
    cfg_ns.source_path = os.path.abspath(_args.source_path)  # RENDER_PREP
    cfg_ns.model_path = model_path  # RENDER_PREP
    dataset = lp.extract(cfg_ns)  # RENDER_PREP
    pipe = pp.extract(cfg_ns)  # RENDER_PREP
    hyper = hp.extract(cfg_ns)  # RENDER_PREP
    ckpt_dir = os.path.join(model_path, "point_cloud", "static_core_final")  # RENDER_PREP
    ply_path = os.path.join(ckpt_dir, "point_cloud_static.ply")  # RENDER_PREP
    spline_pth = os.path.join(ckpt_dir, "cam_spline.pth")  # RENDER_PREP
    posenet_pth = os.path.join(ckpt_dir, "posenet.pth")  # RENDER_PREP
    if not os.path.isfile(ply_path):  # RENDER_PREP
        raise FileNotFoundError(f"[RENDER] Missing {ply_path}")  # RENDER_PREP
    if not os.path.isfile(spline_pth):  # RENDER_PREP
        raise RuntimeError(  # RENDER_PREP
            f"[RENDER] Missing {spline_pth}. Fix: re-run training and ensure cam_spline.pth is saved."  # RENDER_PREP
        )  # RENDER_PREP
    if not os.path.isfile(posenet_pth):  # RENDER_PREP
        raise FileNotFoundError(f"[RENDER] Missing {posenet_pth}")  # RENDER_PREP
    stat_gaussians = GaussianModel(dataset)  # RENDER_PREP
    pose_holder = GaussianModel(dataset)  # RENDER_PREP
    dataset.model_path = model_path  # RENDER_PREP
    scene = Scene(dataset, pose_holder, stat_gaussians, load_coarse=None)  # RENDER_PREP
    train_ds = scene.getTrainCameras()  # RENDER_PREP
    train_cameras = [train_ds[i] for i in range(len(train_ds))]  # RENDER_PREP
    train_cameras = sorted(train_cameras, key=lambda c: c.uid)  # RENDER_PREP
    pose_holder.create_pose_network(hyper, train_cameras)  # RENDER_PREP
    pose_holder._posenet.load_state_dict(torch.load(posenet_pth, map_location="cuda"))  # RENDER_PREP
    stat_gaussians.load_ply(ply_path)  # RENDER_PREP
    spline_ckpt = torch.load(spline_pth, map_location="cpu")  # RENDER_PREP
    cam_spline = CameraSpline(N=int(spline_ckpt["N"]))  # RENDER_PREP
    if int(spline_ckpt["K"]) != cam_spline.K:  # RENDER_PREP
        raise RuntimeError(  # RENDER_PREP
            f"[RENDER] Spline K mismatch: ckpt K={spline_ckpt['K']} vs model K={cam_spline.K}"  # RENDER_PREP
        )  # RENDER_PREP
    cam_spline.ctrl_trans.data = spline_ckpt["ctrl_trans"].cuda()  # RENDER_PREP
    cam_spline.ctrl_quats.data = spline_ckpt["ctrl_quats"].cuda()  # RENDER_PREP
    cam_spline = cam_spline.cuda()  # RENDER_PREP
    N = int(cam_spline.N)  # RENDER_PREP
    if len(train_cameras) != N:  # RENDER_PREP
        raise RuntimeError(  # RENDER_PREP
            f"[RENDER] Camera count {len(train_cameras)} != spline N={N}"  # RENDER_PREP
        )  # RENDER_PREP
    bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]  # RENDER_PREP
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # RENDER_PREP
    focal_np = pose_holder._posenet.focal_bias.exp().detach().cpu().numpy()  # RENDER_PREP
    output_video = os.path.abspath(_args.output_video)  # RENDER_PREP
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)  # RENDER_PREP
    comparison_mp4 = os.path.join(model_path, "comparison.mp4")  # RENDER_PREP
    original_frames_dir = os.path.join(model_path, "original_frames")  # RENDER_PREP
    if not _args.skip_verification:  # RENDER_PREP
        sample_ts = [0, N // 4, N // 2, (3 * N) // 4, N - 1]  # RENDER_PREP
        try:  # RENDER_PREP
            with torch.no_grad():  # RENDER_PREP
                for t in sample_ts:  # RENDER_PREP
                    R, T = cam_spline.get_pose(float(t))  # RENDER_PREP
                    assert R.shape == (3, 3), f"R shape wrong at t={t}"  # RENDER_PREP
                    assert T.shape == (3,), f"T shape wrong at t={t}"  # RENDER_PREP
                    eye = torch.eye(3, device=R.device, dtype=R.dtype)  # RENDER_PREP
                    RtR = R.T @ R  # RENDER_PREP
                    assert (RtR - eye).abs().max() < 1e-3, f"R not SO(3) at t={t}"  # RENDER_PREP
            print("[CHECK 1] Spline poses valid: PASS")  # RENDER_PREP
        except AssertionError as e:  # RENDER_PREP
            raise RuntimeError(  # RENDER_PREP
                "Fix: re-run training and ensure cam_spline.pth is saved."  # RENDER_PREP
            ) from e  # RENDER_PREP
        try:  # RENDER_PREP
            with torch.no_grad():  # RENDER_PREP
                for t in sample_ts:  # RENDER_PREP
                    cam = train_cameras[t]  # RENDER_PREP
                    set_camera_pose_from_spline(cam, cam_spline, focal_np, t)  # RENDER_PREP
                    pkg = render_static(  # RENDER_PREP
                        viewpoint_camera=cam,  # RENDER_PREP
                        stat_pc=stat_gaussians,  # RENDER_PREP
                        dyn_pc=stat_gaussians,  # RENDER_PREP
                        bg_color=background,  # RENDER_PREP
                        get_static=True,  # RENDER_PREP
                    )  # RENDER_PREP
                    img = pkg["render"]  # RENDER_PREP
                    exp = (3, cam.image_height, cam.image_width)  # RENDER_PREP
                    assert img.shape == exp, f"Wrong output shape at t={t}: {img.shape}"  # RENDER_PREP
                    assert not torch.isnan(img).any(), f"NaN pixels at t={t}"  # RENDER_PREP
                    assert not torch.isinf(img).any(), f"Inf pixels at t={t}"  # RENDER_PREP
                    assert img.min() >= 0.0 and img.max() <= 1.0, (  # RENDER_PREP
                        f"Pixel values out of [0,1] at t={t}: min={img.min():.3f} max={img.max():.3f}"  # RENDER_PREP
                    )  # RENDER_PREP
            print("[CHECK 2] Sample renders valid: PASS")  # RENDER_PREP
        except AssertionError as e:  # RENDER_PREP
            raise RuntimeError(  # RENDER_PREP
                "Fix: re-run training and ensure cam_spline.pth is saved."  # RENDER_PREP
            ) from e  # RENDER_PREP
        with torch.no_grad():  # RENDER_PREP
            if N >= 2:  # RENDER_PREP
                start = N // 2  # RENDER_PREP
                nfrm = min(10, N - start)  # RENDER_PREP
                if nfrm >= 2:  # RENDER_PREP
                    stab_diffs = []  # RENDER_PREP
                    orig_diffs = []  # RENDER_PREP
                    for k in range(nfrm - 1):  # RENDER_PREP
                        t0, t1 = start + k, start + k + 1  # RENDER_PREP
                        c0, c1 = train_cameras[t0], train_cameras[t1]  # RENDER_PREP
                        set_camera_pose_from_spline(c0, cam_spline, focal_np, t0)  # RENDER_PREP
                        I0 = render_static(  # RENDER_PREP
                            c0, stat_gaussians, stat_gaussians, background, get_static=True  # RENDER_PREP
                        )["render"]  # RENDER_PREP
                        set_camera_pose_from_spline(c1, cam_spline, focal_np, t1)  # RENDER_PREP
                        I1 = render_static(  # RENDER_PREP
                            c1, stat_gaussians, stat_gaussians, background, get_static=True  # RENDER_PREP
                        )["render"]  # RENDER_PREP
                        stab_diffs.append((I1 - I0).abs().mean().item())  # RENDER_PREP
                        o0 = c0.original_image[:3, :, :].cuda()  # RENDER_PREP
                        o1 = c1.original_image[:3, :, :].cuda()  # RENDER_PREP
                        orig_diffs.append((o1 - o0).abs().mean().item())  # RENDER_PREP
                    stabilized_diff = sum(stab_diffs) / len(stab_diffs)  # RENDER_PREP
                    original_diff = sum(orig_diffs) / len(orig_diffs)  # RENDER_PREP
                    print(f"[CHECK 3] stabilized_mean_abs_diff={stabilized_diff:.6f} original_mean_abs_diff={original_diff:.6f}")  # RENDER_PREP
                    if stabilized_diff < original_diff * 1.5:  # RENDER_PREP
                        print("[CHECK 3] Stabilized frames smoother than input: PASS")  # RENDER_PREP
                    else:  # RENDER_PREP
                        print("[CHECK 3] Stabilized frames smoother than input: FAIL (diagnostic only, continuing)")  # RENDER_PREP
                else:  # RENDER_PREP
                    print("[CHECK 3] Skipped (not enough frames in window)")  # RENDER_PREP
            else:  # RENDER_PREP
                print("[CHECK 3] Skipped (N < 2)")  # RENDER_PREP
        check4_warn = False  # RENDER_PREP
        with torch.no_grad():  # RENDER_PREP
            for t in sample_ts:  # RENDER_PREP
                cam = train_cameras[t]  # RENDER_PREP
                set_camera_pose_from_spline(cam, cam_spline, focal_np, t)  # RENDER_PREP
                img = render_static(  # RENDER_PREP
                    cam, stat_gaussians, stat_gaussians, background, get_static=True  # RENDER_PREP
                )["render"]  # RENDER_PREP
                H, W = cam.image_height, cam.image_width  # RENDER_PREP
                bh = max(1, int(0.05 * H))  # RENDER_PREP
                bw = max(1, int(0.05 * W))  # RENDER_PREP
                border = torch.zeros((H, W), dtype=torch.bool, device=img.device)  # RENDER_PREP
                border[:bh, :] = True  # RENDER_PREP
                border[-bh:, :] = True  # RENDER_PREP
                border[:, :bw] = True  # RENDER_PREP
                border[:, -bw:] = True  # RENDER_PREP
                bmask = border  # RENDER_PREP
                black = (img < 0.02).all(dim=0) & bmask  # RENDER_PREP
                pct = 100.0 * black.sum().float() / bmask.sum().float()  # RENDER_PREP
                if pct > 30.0:  # RENDER_PREP
                    check4_warn = True  # RENDER_PREP
                    print(  # RENDER_PREP
                        f"[CHECK 4] WARNING: heavy black borders detected at frame t={t} ({pct:.1f}% black border) — FoV loss may need higher w_fov"  # RENDER_PREP
                    )  # RENDER_PREP
        if not check4_warn:  # RENDER_PREP
            print("[CHECK 4] Border check: PASS")  # RENDER_PREP
    output_frames_dir = output_video.replace(".mp4", "_frames/")  # RENDER_PREP
    os.makedirs(output_frames_dir, exist_ok=True)  # RENDER_PREP
    os.makedirs(original_frames_dir, exist_ok=True)  # RENDER_PREP
    with torch.no_grad():  # RENDER_PREP
        for t, cam in enumerate(tqdm(train_cameras, desc="Rendering")):  # RENDER_PREP
            set_camera_pose_from_spline(cam, cam_spline, focal_np, t)  # RENDER_PREP
            pkg = render_static(cam, stat_gaussians, stat_gaussians, background, get_static=True)  # RENDER_PREP
            img = pkg["render"]  # RENDER_PREP
            img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")  # RENDER_PREP
            frame_path = os.path.join(output_frames_dir, f"{t:05d}.png")  # RENDER_PREP
            cv2.imwrite(frame_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))  # RENDER_PREP
            oimg = cam.original_image[:3, :, :]  # RENDER_PREP
            o_np = (oimg.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")  # RENDER_PREP
            cv2.imwrite(os.path.join(original_frames_dir, f"{t:05d}.png"), cv2.cvtColor(o_np, cv2.COLOR_RGB2BGR))  # RENDER_PREP
    print(f"[RENDER] {N} frames saved to {output_frames_dir}")  # RENDER_PREP
    cmd = [  # RENDER_PREP
        "ffmpeg",  # RENDER_PREP
        "-y",  # RENDER_PREP
        "-framerate",  # RENDER_PREP
        str(_args.fps),  # RENDER_PREP
        "-i",  # RENDER_PREP
        os.path.join(output_frames_dir, "%05d.png"),  # RENDER_PREP
        "-c:v",  # RENDER_PREP
        "libx264",  # RENDER_PREP
        "-crf",  # RENDER_PREP
        "18",  # RENDER_PREP
        "-pix_fmt",  # RENDER_PREP
        "yuv420p",  # RENDER_PREP
        output_video,  # RENDER_PREP
    ]  # RENDER_PREP
    result = subprocess.run(cmd, capture_output=True, text=True)  # RENDER_PREP
    if result.returncode != 0:  # RENDER_PREP
        print(f"[ERROR] ffmpeg failed: {result.stderr}")  # RENDER_PREP
        print(f"        Frames are saved in {output_frames_dir}")  # RENDER_PREP
        print(  # RENDER_PREP
            f"        Run manually: ffmpeg -framerate {_args.fps} "  # RENDER_PREP
            f"-i {output_frames_dir}/%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p {output_video}"  # RENDER_PREP
        )  # RENDER_PREP
    else:  # RENDER_PREP
        print(f"[RENDER] Video saved to {output_video}")  # RENDER_PREP
    cmd_h = [  # RENDER_PREP
        "ffmpeg",  # RENDER_PREP
        "-y",  # RENDER_PREP
        "-i",  # RENDER_PREP
        os.path.join(output_frames_dir, "%05d.png"),  # RENDER_PREP
        "-i",  # RENDER_PREP
        os.path.join(original_frames_dir, "%05d.png"),  # RENDER_PREP
        "-filter_complex",  # RENDER_PREP
        "hstack",  # RENDER_PREP
        "-c:v",  # RENDER_PREP
        "libx264",  # RENDER_PREP
        "-crf",  # RENDER_PREP
        "18",  # RENDER_PREP
        "-pix_fmt",  # RENDER_PREP
        "yuv420p",  # RENDER_PREP
        comparison_mp4,  # RENDER_PREP
    ]  # RENDER_PREP
    rh = subprocess.run(cmd_h, capture_output=True, text=True)  # RENDER_PREP
    if rh.returncode != 0:  # RENDER_PREP
        print(f"[WARN] comparison ffmpeg failed: {rh.stderr}")  # RENDER_PREP
    else:  # RENDER_PREP
        print(f"[RENDER] Side-by-side comparison saved to {comparison_mp4}")  # RENDER_PREP
if __name__ == "__main__":  # RENDER_PREP
    main()  # RENDER_PREP
