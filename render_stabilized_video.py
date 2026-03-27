#!/usr/bin/env python3
"""Render a stabilized video from a saved trajectory and static-core checkpoint."""

import argparse
import os
from argparse import Namespace
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from arguments import ModelParams
from gaussian_renderer import render_static
from scene import GaussianModel, Scene
from scene.camera_spline import CameraSpline
from train_static_core import set_camera_pose_from_spline


def _read_cfg_args(run_dir: str) -> Namespace:
    cfg_path = os.path.join(run_dir, "cfg_args")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing cfg_args at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    return eval(txt, {"Namespace": Namespace})


def _make_dataset_from_cfg(parser: argparse.ArgumentParser, cfg_ns: Namespace, run_dir: str):
    lp = ModelParams(parser)
    merged = vars(cfg_ns).copy()
    merged["model_path"] = run_dir
    args = Namespace(**merged)
    return lp.extract(args)


def _resolve_source_path(cfg_source_path: str, run_dir: str, override_source_path: Optional[str]) -> str:
    candidates = []
    if override_source_path:
        candidates.append(os.path.abspath(override_source_path))
    if cfg_source_path:
        candidates.append(os.path.abspath(cfg_source_path))

    base = os.path.basename(os.path.abspath(cfg_source_path)) if cfg_source_path else ""
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    if base:
        candidates.extend(
            [
                os.path.join(workspace_root, "data2", base),
                os.path.join(run_dir, "..", "..", "data2", base),
            ]
        )

    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(os.path.join(p, "images_2")):
            return p

    raise FileNotFoundError(
        "Could not find a valid dataset source path with images_2/. "
        f"Checked: {candidates}. "
        "Pass --source-path explicitly."
    )


def _infer_focal_from_posenet(posenet_path: str) -> Optional[float]:
    if not os.path.isfile(posenet_path):
        return None
    state = torch.load(posenet_path, map_location="cpu")
    if "focal_bias" not in state:
        return None
    return float(torch.exp(state["focal_bias"]).reshape(-1)[0].item())


def _trajectory_to_c2w(
    trajectory_path: str,
    num_frames: int,
    device: torch.device,
) -> np.ndarray:
    data = np.load(trajectory_path, allow_pickle=True)

    if isinstance(data, np.lib.npyio.NpzFile):
        keys = set(data.files)
        if "c2w" in keys:
            c2w = data["c2w"]
        elif "w2c" in keys:
            c2w = np.linalg.inv(data["w2c"])
        elif {"ctrl_trans", "ctrl_quats"}.issubset(keys):
            spline = CameraSpline(N=num_frames).to(device)
            with torch.no_grad():
                spline.ctrl_trans.copy_(torch.from_numpy(data["ctrl_trans"]).float().to(device))
                spline.ctrl_quats.copy_(torch.from_numpy(data["ctrl_quats"]).float().to(device))
            mats = []
            with torch.no_grad():
                for t in range(num_frames):
                    R, T = spline.get_pose(float(t))
                    w2c = torch.eye(4, device=device, dtype=torch.float32)
                    # Keep the same convention as training-time camera update path.
                    w2c[:3, :3] = R.transpose(0, 1)
                    w2c[:3, 3] = T
                    mats.append(torch.linalg.inv(w2c).cpu().numpy())
            c2w = np.stack(mats, axis=0)
        else:
            raise ValueError(
                "Unsupported NPZ trajectory. Expected one of: "
                "{c2w}, {w2c}, or {ctrl_trans, ctrl_quats}."
            )
    else:
        c2w = data

    if c2w.ndim != 3:
        raise ValueError(f"Trajectory must be rank-3, got shape {c2w.shape}")
    if c2w.shape[1:] == (3, 4):
        pad = np.zeros((c2w.shape[0], 1, 4), dtype=c2w.dtype)
        pad[:, 0, 3] = 1.0
        c2w = np.concatenate([c2w, pad], axis=1)
    if c2w.shape[1:] != (4, 4):
        raise ValueError(f"Trajectory must be [N,4,4] or [N,3,4], got {c2w.shape}")
    if c2w.shape[0] < num_frames:
        raise ValueError(f"Trajectory has {c2w.shape[0]} frames, expected at least {num_frames}")
    return c2w[:num_frames].astype(np.float32)


def _load_spline_from_controls_npz(
    trajectory_path: str,
    num_frames: int,
    device: torch.device,
) -> Optional[CameraSpline]:
    if not trajectory_path.endswith(".npz"):
        return None
    data = np.load(trajectory_path, allow_pickle=True)
    keys = set(data.files)
    if not {"ctrl_trans", "ctrl_quats"}.issubset(keys):
        return None
    spline = CameraSpline(N=num_frames).to(device)
    with torch.no_grad():
        spline.ctrl_trans.copy_(torch.from_numpy(data["ctrl_trans"]).float().to(device))
        spline.ctrl_quats.copy_(torch.from_numpy(data["ctrl_quats"]).float().to(device))
    return spline


def _resolve_trajectory_path(run_dir: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path
    candidates = [
        os.path.join(run_dir, "point_cloud", "static_core_final", "cam_spline_controls.npz"),
        os.path.join(run_dir, "point_cloud", "static_core_final", "cam_spline_c2w.npy"),
        os.path.join(run_dir, "point_cloud", "static_core_final", "train_cam_c2w_pred.npy"),
        os.path.join(run_dir, "point_cloud", "static_core_final", "train_cam_c2w_spline.npy"),
        os.path.join(run_dir, "point_cloud", "static_core_final", "train_cam_c2w_gt.npy"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No trajectory file provided and no default trajectory file found. "
        "Pass --trajectory explicitly."
    )


def main():
    parser = argparse.ArgumentParser(description="Render stabilized video from trajectory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir (contains cfg_args and point_cloud/...).")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to trajectory (.npy or .npz).")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory containing point_cloud_static.ply (defaults to point_cloud/static_core_final).",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Dataset source path override (must contain images_2).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output .mp4 path.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_ns = _read_cfg_args(run_dir)
    dataset = _make_dataset_from_cfg(argparse.ArgumentParser(), cfg_ns, run_dir)
    dataset.source_path = _resolve_source_path(dataset.source_path, run_dir, args.source_path)

    ckpt_dir = (
        os.path.abspath(args.checkpoint_dir)
        if args.checkpoint_dir
        else os.path.join(run_dir, "point_cloud", "static_core_final")
    )
    ply_path = os.path.join(ckpt_dir, "point_cloud_static.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Missing static point cloud: {ply_path}")

    output_path = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(run_dir, "stabilized_from_spline.mp4")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traj_path = _resolve_trajectory_path(run_dir, args.trajectory)
    print(f"[info] run_dir: {run_dir}")
    print(f"[info] checkpoint: {ckpt_dir}")
    print(f"[info] source_path: {dataset.source_path}")
    print(f"[info] trajectory: {traj_path}")

    device = torch.device("cuda")
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, gaussians, load_coarse=None)
    gaussians.load_ply(ply_path)

    cameras = [cam for cam in scene.getTrainCameras()]
    cameras.sort(key=lambda c: c.uid)
    num_frames = len(cameras)
    spline = _load_spline_from_controls_npz(traj_path, num_frames=num_frames, device=device)
    c2w = None if spline is not None else _trajectory_to_c2w(traj_path, num_frames=num_frames, device=device)

    posenet_path = os.path.join(ckpt_dir, "posenet.pth")
    focal_override = _infer_focal_from_posenet(posenet_path)
    if focal_override is not None:
        print(f"[info] focal from posenet: {focal_override:.4f}")
    else:
        print("[warn] no focal_bias in posenet; using camera metadata focal.")

    h = int(cameras[0].image_height)
    w = int(cameras[0].image_width)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")

    bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            focal = focal_override if focal_override is not None else float(cam.metadata.focal_length.reshape(-1)[0])
            if spline is not None:
                # Use the exact same camera pose application path as training.
                set_camera_pose_from_spline(cam, spline, np.array([focal], dtype=np.float32), i)
            else:
                w2c = np.linalg.inv(c2w[i])
                # update_cam internally builds view with getWorld2View2_torch(R, T),
                # where R is transposed inside that function.
                R = torch.from_numpy(c2w[i][:3, :3]).float().to(device)
                T = torch.from_numpy(w2c[:3, 3]).float().to(device)
                cam.update_cam(R, T, local_viewdirs=None, batch_shape=None, focal=focal)

            pkg = render_static(
                viewpoint_camera=cam,
                stat_pc=gaussians,
                dyn_pc=gaussians,
                bg_color=background,
                get_static=True,
            )
            img = torch.clamp(pkg["render"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            writer.write(cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if i == 0:
                vis_count = int(pkg["visibility_filter"].sum().item())
                print(f"[debug] frame0 visible gaussians: {vis_count}")

            if (i + 1) % 25 == 0 or (i + 1) == num_frames:
                print(f"[render] {i + 1}/{num_frames}")

    writer.release()
    print(f"[done] wrote stabilized video: {output_path}")


if __name__ == "__main__":
    main()
