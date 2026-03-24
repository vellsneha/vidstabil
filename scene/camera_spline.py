"""SE(3) cubic Hermite spline for camera trajectory.  # STEP1.2

Replaces per-frame pose_network with K = N // 5 differentiable control
points (translations + unit quaternions).  All forward-path operations
are pure PyTorch; no numpy calls and no .detach() in get_pose /
get_all_poses.
"""  # STEP1.2

import torch  # STEP1.2
import torch.nn as nn  # STEP1.2


class CameraSpline(nn.Module):  # STEP1.2
    """Cubic Hermite spline over SE(3) with K = N // 5 control points.

    Control-point parameterisation
    --------------------------------
    ctrl_trans : nn.Parameter [K, 3]   -- translation for each knot
    ctrl_quats : nn.Parameter [K, 4]   -- unit quaternion (wxyz) for each knot

    The spline maps a continuous frame index t in [0, N-1] to a pose
    (R [3,3], T [3]) by:
      1. mapping t linearly onto the control-point index axis,
      2. cubic-Hermite interpolating translations, and
      3. Squad-interpolating rotations (two nested Slerps).
    """  # STEP1.2

    def __init__(self, N: int):  # STEP1.2
        super().__init__()  # STEP1.2
        self.N = N  # STEP1.2
        self.K = max(N // 5, 2)  # STEP1.2  at least 2 so we always have one interval
        self.ctrl_trans = nn.Parameter(torch.zeros(self.K, 3))  # STEP1.2
        self.ctrl_quats = nn.Parameter(torch.zeros(self.K, 4))  # STEP1.2
        with torch.no_grad():  # STEP1.2
            self.ctrl_quats[:, 0] = 1.0  # wxyz identity initialisation  # STEP1.2

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _slerp(  # STEP1.2
        self,
        q0: torch.Tensor,  # STEP1.2
        q1: torch.Tensor,  # STEP1.2
        u: torch.Tensor,  # STEP1.2
    ) -> torch.Tensor:  # STEP1.2
        """Differentiable slerp between unit quaternions q0, q1 at scalar u in [0,1].

        Shortest-path convention: q1 is flipped if dot(q0,q1) < 0.
        Falls back to linear interpolation when the angle is near zero.
        No detach or numpy calls; fully differentiable w.r.t. q0, q1.
        """  # STEP1.2
        dot = (q0 * q1).sum().clamp(-1.0, 1.0)  # STEP1.2
        # Shortest-path: negate q1 when dot < 0 (arithmetic, no branch)  # STEP1.2
        neg_mask = (dot < 0.0).float()  # STEP1.2
        q1_adj = q1 * (1.0 - 2.0 * neg_mask)  # flip sign if needed  # STEP1.2
        dot_adj = dot.abs()  # STEP1.2
        theta = torch.acos(dot_adj.clamp(0.0, 1.0))  # STEP1.2
        sin_theta = torch.sin(theta)  # STEP1.2
        # Stable slerp: replace denominator with 1 when sin_theta is tiny  # STEP1.2
        safe_sin = torch.where(  # STEP1.2
            sin_theta.abs() > 1e-6,  # STEP1.2
            sin_theta,  # STEP1.2
            torch.ones_like(sin_theta),  # STEP1.2
        )  # STEP1.2
        w0_slerp = torch.sin((1.0 - u) * theta) / safe_sin  # STEP1.2
        w1_slerp = torch.sin(u * theta) / safe_sin  # STEP1.2
        # Blend: use linear fallback when angle is near zero  # STEP1.2
        mask = (sin_theta.abs() > 1e-6).float()  # STEP1.2
        w0 = mask * w0_slerp + (1.0 - mask) * (1.0 - u)  # STEP1.2
        w1 = mask * w1_slerp + (1.0 - mask) * u  # STEP1.2
        return w0 * q0 + w1 * q1_adj  # STEP1.2

    def _quat_to_matrix(self, q: torch.Tensor) -> torch.Tensor:  # STEP1.2
        """Convert unit quaternion (wxyz) to 3×3 rotation matrix.

        Standard formula — no external libraries, fully differentiable.
        """  # STEP1.2
        w, x, y, z = q[0], q[1], q[2], q[3]  # STEP1.2
        return torch.stack(  # STEP1.2
            [  # STEP1.2
                torch.stack(  # STEP1.2
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)]  # STEP1.2
                ),  # STEP1.2
                torch.stack(  # STEP1.2
                    [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)]  # STEP1.2
                ),  # STEP1.2
                torch.stack(  # STEP1.2
                    [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]  # STEP1.2
                ),  # STEP1.2
            ]  # STEP1.2
        )  # STEP1.2

    def _rotmat_to_quat(self, Rs: torch.Tensor) -> torch.Tensor:  # STEP1.2
        """Convert batch of rotation matrices [K, 3, 3] → unit quaternions [K, 4] (wxyz).

        Uses Shepperd's method with .item() comparisons.
        Only ever called from initialize_from_poses() inside torch.no_grad().
        """  # STEP1.2
        quats = []  # STEP1.2
        for k in range(Rs.shape[0]):  # STEP1.2
            R = Rs[k]  # STEP1.2
            trace = R[0, 0] + R[1, 1] + R[2, 2]  # STEP1.2
            if trace.item() > 0:  # STEP1.2
                s = 0.5 / torch.sqrt(trace + 1.0)  # STEP1.2
                w = 0.25 / s  # STEP1.2
                x = (R[2, 1] - R[1, 2]) * s  # STEP1.2
                y = (R[0, 2] - R[2, 0]) * s  # STEP1.2
                z = (R[1, 0] - R[0, 1]) * s  # STEP1.2
            elif R[0, 0].item() > R[1, 1].item() and R[0, 0].item() > R[2, 2].item():  # STEP1.2
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])  # STEP1.2
                w = (R[2, 1] - R[1, 2]) / s  # STEP1.2
                x = 0.25 * s  # STEP1.2
                y = (R[0, 1] + R[1, 0]) / s  # STEP1.2
                z = (R[0, 2] + R[2, 0]) / s  # STEP1.2
            elif R[1, 1].item() > R[2, 2].item():  # STEP1.2
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])  # STEP1.2
                w = (R[0, 2] - R[2, 0]) / s  # STEP1.2
                x = (R[0, 1] + R[1, 0]) / s  # STEP1.2
                y = 0.25 * s  # STEP1.2
                z = (R[1, 2] + R[2, 1]) / s  # STEP1.2
            else:  # STEP1.2
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])  # STEP1.2
                w = (R[1, 0] - R[0, 1]) / s  # STEP1.2
                x = (R[0, 2] + R[2, 0]) / s  # STEP1.2
                y = (R[1, 2] + R[2, 1]) / s  # STEP1.2
                z = 0.25 * s  # STEP1.2
            q = torch.stack([w, x, y, z])  # STEP1.2
            quats.append(q / q.norm())  # STEP1.2
        return torch.stack(quats)  # STEP1.2

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_pose(self, t: float):  # STEP1.2
        """Return (R [3,3], T [3]) at continuous frame index t in [0, N-1].

        All operations are differentiable; no detach or numpy calls.
        Gradients flow through ctrl_trans and ctrl_quats.
        """  # STEP1.2
        # Map frame index to fractional control-point coordinate  # STEP1.2
        t_ctrl = t * (self.K - 1) / max(self.N - 1, 1)  # STEP1.2
        i = int(t_ctrl)  # STEP1.2
        i = max(0, min(i, self.K - 2))  # clamp interval index  # STEP1.2
        u_val = t_ctrl - i  # local parameter in [0, 1)  # STEP1.2
        u = torch.tensor(  # STEP1.2
            u_val, dtype=self.ctrl_trans.dtype, device=self.ctrl_trans.device  # STEP1.2
        )  # STEP1.2

        # ---- Translation: cubic Hermite interpolation ---- #  # STEP1.2
        p0 = self.ctrl_trans[i]  # STEP1.2
        p1 = self.ctrl_trans[i + 1]  # STEP1.2
        # Finite-difference tangents; clamp at boundaries  # STEP1.2
        if i > 0:  # STEP1.2
            m0 = 0.5 * (self.ctrl_trans[i + 1] - self.ctrl_trans[i - 1])  # STEP1.2
        else:  # STEP1.2
            m0 = p1 - p0  # boundary clamp  # STEP1.2
        if i + 1 < self.K - 1:  # STEP1.2
            m1 = 0.5 * (self.ctrl_trans[i + 2] - self.ctrl_trans[i])  # STEP1.2
        else:  # STEP1.2
            m1 = p1 - p0  # boundary clamp  # STEP1.2
        u2 = u * u  # STEP1.2
        u3 = u2 * u  # STEP1.2
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0  # STEP1.2
        h10 = u3 - 2.0 * u2 + u  # STEP1.2
        h01 = -2.0 * u3 + 3.0 * u2  # STEP1.2
        h11 = u3 - u2  # STEP1.2
        T_out = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1  # STEP1.2

        # ---- Rotation: Squad (two nested Slerps) ---- #  # STEP1.2
        q_i = self.ctrl_quats[i] / self.ctrl_quats[i].norm()  # STEP1.2
        q_i1 = self.ctrl_quats[i + 1] / self.ctrl_quats[i + 1].norm()  # STEP1.2
        q_im1_idx = max(0, i - 1)  # STEP1.2
        q_i2_idx = min(self.K - 1, i + 2)  # STEP1.2
        q_im1 = self.ctrl_quats[q_im1_idx] / self.ctrl_quats[q_im1_idx].norm()  # STEP1.2
        q_i2 = self.ctrl_quats[q_i2_idx] / self.ctrl_quats[q_i2_idx].norm()  # STEP1.2
        slerp_main = self._slerp(q_i, q_i1, u)  # STEP1.2
        slerp_aux = self._slerp(q_im1, q_i2, u)  # STEP1.2
        u_squad = 2.0 * u * (1.0 - u)  # STEP1.2
        q_out = self._slerp(slerp_main, slerp_aux, u_squad)  # STEP1.2
        q_out = q_out / q_out.norm()  # STEP1.2

        R_out = self._quat_to_matrix(q_out)  # STEP1.2
        return R_out, T_out  # STEP1.2

    def get_all_poses(self, N: int):  # STEP1.2
        """Return [(R_0, T_0), ..., (R_{N-1}, T_{N-1})] for frame indices 0..N-1."""  # STEP1.2
        return [self.get_pose(float(t)) for t in range(N)]  # STEP1.2

    def initialize_from_poses(  # STEP1.2
        self,
        Rs: torch.Tensor,  # STEP1.2  [N, 3, 3]
        Ts: torch.Tensor,  # STEP1.2  [N, 3]
    ) -> None:  # STEP1.2
        """Warm-start: uniformly subsample N per-frame poses to K control points.

        Rs and Ts are CPU tensors; no gradient tracking needed here.
        """  # STEP1.2
        N = Rs.shape[0]  # STEP1.2
        indices = torch.linspace(0, N - 1, self.K).long()  # STEP1.2
        sampled_Rs = Rs[indices]  # [K, 3, 3]  # STEP1.2
        sampled_Ts = Ts[indices]  # [K, 3]  # STEP1.2
        quats = self._rotmat_to_quat(sampled_Rs)  # [K, 4]  # STEP1.2
        with torch.no_grad():  # STEP1.2
            self.ctrl_trans.data.copy_(sampled_Ts)  # STEP1.2
            self.ctrl_quats.data.copy_(quats)  # STEP1.2
