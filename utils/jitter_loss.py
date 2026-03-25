"""Rendered jitter loss (Step 1.4 Term 2) — Laplacian on pixel diff or RAFT flow."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LAPL_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


def _laplacian_channels(x: torch.Tensor) -> torch.Tensor:
    """x: [C, H, W] -> Laplacian per channel [C, H, W]."""
    c, h, w = x.shape
    device, dtype = x.device, x.dtype
    k = _LAPL_KERNEL.to(device=device, dtype=dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    x4 = x.unsqueeze(0)
    return F.conv2d(x4, k, padding=1, groups=c).squeeze(0)


def frobenius_norm_squared(tensor: torch.Tensor) -> torch.Tensor:
    """Scalar ||T||_F^2 (sum of squares)."""
    return (tensor * tensor).sum()


def loss_jitter_pixel_diff(I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
    """L_jitter = || nabla^2 (I1 - I0) ||_F^2  (Frobenius as sum of squares, matches ||·||_F^2 training).

    I0, I1: [3, H, W] in same value range as renderer output.
    Differentiable w.r.t. both images.
    """
    diff = I1 - I0
    lap = _laplacian_channels(diff)
    return torch.sqrt(frobenius_norm_squared(lap) + 1e-12)


# --- RAFT (optional; torchvision) ----------------------------------------------

_RAFT_BUNDLE = None  # (model, transforms, device) or None


def _get_raft_bundle(device: torch.device):
    global _RAFT_BUNDLE
    if _RAFT_BUNDLE is not None:
        model, transforms, dev = _RAFT_BUNDLE
        if dev != device:
            model = model.to(device)
            _RAFT_BUNDLE = (model, transforms, device)
        return _RAFT_BUNDLE
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except ImportError:
        return None

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    transforms = weights.transforms()
    _RAFT_BUNDLE = (model, transforms, device)
    return _RAFT_BUNDLE


def loss_jitter_raft_laplacian(
    I0: torch.Tensor,
    I1: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """|| nabla^2(flow) ||_F with RAFT flow; entire forward under torch.no_grad (no grad to scene).

    I0, I1: [3, H, W] in [0, 1]. Returns a scalar tensor (detached).
    """
    bundle = _get_raft_bundle(device)
    if bundle is None:
        return loss_jitter_pixel_diff(I0.detach(), I1.detach()).detach()

    model, transforms, _dev = bundle
    # [1,3,H,W]
    a = I0.unsqueeze(0).clamp(0.0, 1.0).to(device)
    b = I1.unsqueeze(0).clamp(0.0, 1.0).to(device)

    with torch.no_grad():
        try:
            tr_out = transforms(a, b)
            if isinstance(tr_out, (tuple, list)) and len(tr_out) == 2:
                a_t, b_t = tr_out[0], tr_out[1]
            else:
                a_t, b_t = a, b
        except Exception:
            a_t, b_t = a, b
        flows = model(a_t, b_t)
        flow = flows[-1]  # [1, 2, H', W']
        lap = _laplacian_channels(flow.squeeze(0))
        loss_scalar = torch.sqrt(frobenius_norm_squared(lap) + 1e-12)
    return loss_scalar.detach()
