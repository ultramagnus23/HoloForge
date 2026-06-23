"""
metrics.py
----------
Perceptual and physical image quality metrics.

  psnr()   — Peak Signal-to-Noise Ratio  (higher = better)
  ssim()   — Structural Similarity Index (higher = better, max 1.0)
  mse()    — Mean Squared Error
  lpips_proxy() — LPIPS approximation via gradient magnitude difference
                  (no deep-learning dependency required)

All functions accept float32 numpy arrays in [0, 1].
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as _psnr_skimage


# ─────────────────────────────────────────────────────────────────────────────
#  Basic metrics
# ─────────────────────────────────────────────────────────────────────────────

def mse(ref: np.ndarray, deg: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((ref.astype(np.float32) - deg.astype(np.float32)) ** 2))


def psnr(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Returns np.inf if images are identical.
    Assumes data range [0, 1].
    """
    return float(_psnr_skimage(ref, deg, data_range=1.0))


def ssim(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Structural Similarity Index.
    Handles both grayscale (H,W) and colour (H,W,3).
    """
    if ref.ndim == 3:
        return float(_ssim_skimage(ref, deg, channel_axis=2, data_range=1.0))
    return float(_ssim_skimage(ref, deg, data_range=1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Perceptual proxy (no torch/tensorflow needed)
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude via finite differences (Sobel-like)."""
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def lpips_gradient_proxy(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Lightweight perceptual similarity proxy using gradient-magnitude MSE.

    NOTE: This is NOT LPIPS. It is a cheap, dependency-free surrogate that
    correlates loosely with true perceptual distance for holographic
    reconstructions. It is reported alongside — never instead of — the real
    LPIPS metric (`lpips_real`). Lower = more similar (like true LPIPS).
    """
    gm_ref = _gradient_magnitude(ref)
    gm_deg = _gradient_magnitude(deg)
    return float(np.mean((gm_ref - gm_deg) ** 2))


# Backward-compatible alias (old name).
lpips_proxy = lpips_gradient_proxy


def lpips_real(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    True LPIPS perceptual distance using AlexNet features (Zhang et al. 2018).

    Requires: pip install lpips torch torchvision

    Parameters
    ----------
    ref, deg : float32 (H, W) in [0, 1] — grayscale holographic reconstructions

    Returns
    -------
    float — LPIPS distance (lower = more similar, 0 = identical)
    """
    try:
        import torch
        import lpips as lpips_lib
        # Lazy-load model (cached after first call)
        if not hasattr(lpips_real, '_fn'):
            lpips_real._fn = lpips_lib.LPIPS(net='alex', verbose=False)

        def _to_tensor(arr):
            # (H,W) float32 [0,1] → (1,3,H,W) tensor in [-1,1]
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            t = t.repeat(1, 3, 1, 1)    # grayscale → 3-channel
            return t * 2 - 1            # [0,1] → [-1,1]

        with torch.no_grad():
            score = lpips_real._fn(_to_tensor(ref), _to_tensor(deg))
        return float(score.item())
    except ImportError:
        # Graceful fallback to gradient proxy with a warning
        import warnings
        warnings.warn("lpips not installed — falling back to gradient proxy. "
                      "Run: pip install lpips torch torchvision")
        return lpips_gradient_proxy(ref, deg)


# ─────────────────────────────────────────────────────────────────────────────
#  Bundle — compute all at once
# ─────────────────────────────────────────────────────────────────────────────

def all_metrics(ref: np.ndarray, deg: np.ndarray) -> dict:
    """
    Compute all quality metrics and return as a dict.

    Example
    -------
    >>> scores = all_metrics(ref_img, degraded_img)
    >>> print(scores)
    {'mse': 0.003, 'psnr': 25.2, 'ssim': 0.87, 'lpips_proxy': 0.001, 'lpips': 0.12}
    """
    return {
        "mse":         mse(ref, deg),
        "psnr":        psnr(ref, deg),
        "ssim":        ssim(ref, deg),
        "lpips_proxy": lpips_gradient_proxy(ref, deg),
        "lpips":       lpips_real(ref, deg),
    }
