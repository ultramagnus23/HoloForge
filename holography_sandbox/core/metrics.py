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


def lpips_proxy(ref: np.ndarray, deg: np.ndarray) -> float:
    """
    Lightweight perceptual similarity proxy using gradient-magnitude MSE.

    Correlates reasonably with true LPIPS for holographic reconstructions
    without requiring PyTorch. Lower = more similar (like true LPIPS).

    For real LPIPS you'd do:
        pip install lpips
        import lpips; loss_fn = lpips.LPIPS(net='alex')
    """
    gm_ref = _gradient_magnitude(ref)
    gm_deg = _gradient_magnitude(deg)
    return float(np.mean((gm_ref - gm_deg) ** 2))


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
    {'mse': 0.003, 'psnr': 25.2, 'ssim': 0.87, 'lpips_proxy': 0.001}
    """
    return {
        "mse":         mse(ref, deg),
        "psnr":        psnr(ref, deg),
        "ssim":        ssim(ref, deg),
        "lpips_proxy": lpips_proxy(ref, deg),
    }
