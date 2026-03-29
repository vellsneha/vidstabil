"""FoV preservation loss (Step 1.4 Term 3) — frozen low-frequency translation reference."""  # STEP1.4

from __future__ import annotations

import torch
import torch.nn.functional as F


def frozen_low_frequency_translation_reference(
    T_init: torch.Tensor,
    *,
    kernel_halfwidth: int | None = None,
) -> torch.Tensor:
    """Build **T̄**: heavily smoothed per-frame translations, then **frozen** (detached).

    ``T_init`` shape ``[N, 3]`` — initial rough world-to-camera translations (same
    convention as ``CameraSpline.get_pose`` translation output).

    Smoothing: 1D Gaussian convolution along time with replicate padding at ends.
    Default ``kernel_halfwidth`` is ``min(max(3, N // 6), (N-1)//2)`` (clamped),
    giving a wide low-pass for typical sequence lengths.

    Returns ``[N, 3]`` with ``requires_grad=False``; safe to keep as a constant target.
    """
    N = T_init.shape[0]
    if N < 1:
        raise ValueError("N must be >= 1")
    device, dtype = T_init.device, T_init.dtype
    T = T_init.to(device=device, dtype=dtype)

    if N == 1:
        return T.detach()

    if kernel_halfwidth is None:
        # Wide low-pass along time; cap so kernel fits with replicate padding.
        max_half = max((N - 1) // 2, 1)
        kh = max(3, min(N // 6, max_half))
        kernel_halfwidth = min(kh, max_half)

    k = 2 * kernel_halfwidth + 1
    x = torch.arange(k, device=device, dtype=dtype) - float(kernel_halfwidth)
    sigma = max(float(kernel_halfwidth) / 2.0, 0.5)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    w = w / (w.sum() + 1e-12)

    T_perm = T.T.unsqueeze(0)  # [1, 3, N]
    pad = kernel_halfwidth
    T_pad = F.pad(T_perm, (pad, pad), mode="replicate")
    weight = w.view(1, 1, k).expand(3, 1, k).contiguous()
    out = F.conv1d(T_pad, weight, groups=3)
    ref = out.squeeze(0).T.contiguous()
    return ref.detach()
