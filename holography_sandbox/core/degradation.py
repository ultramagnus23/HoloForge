"""
degradation.py
--------------
All degradation knobs described in Phase 0 / Project 2.

Each function takes an input (hologram phase array, or RGB image)
and returns a degraded version of the same shape / type.

Knob catalogue
--------------
  1. Resolution downscale        — degrade_resolution()
  2. Phase quantisation          — quantise_phase()
  3. Number of depth planes      — (handled in experiment runner)
  4. Color channels              — degrade_color()
  5. Viewing angle (bandwidth)   — limit_viewing_angle()
  6. Speckle noise               — add_speckle()
"""

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  1. Resolution
# ─────────────────────────────────────────────────────────────────────────────

def degrade_resolution(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsample `image` to (target_size × target_size) then upsample
    back to original size using nearest-neighbour — simulates SLM pixel budget.

    Works for 2-D (H,W) float arrays or 3-D (H,W,C) uint8 arrays.
    """
    original_h, original_w = image.shape[:2]
    is_float = image.dtype in (np.float32, np.float64)
    is_3d = image.ndim == 3

    if is_float:
        # Convert to uint8 for PIL, process, convert back
        arr_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        arr_uint8 = image

    if is_3d:
        pil_img = Image.fromarray(arr_uint8, mode="RGB")
    else:
        pil_img = Image.fromarray(arr_uint8, mode="L")

    small = pil_img.resize((target_size, target_size), Image.NEAREST)
    restored = small.resize((original_w, original_h), Image.NEAREST)
    result = np.array(restored)

    if is_float:
        return result.astype(np.float32) / 255.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  2. Phase quantisation
# ─────────────────────────────────────────────────────────────────────────────

def quantise_phase(phase: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantise a phase array (values in [-π, π]) to `bits` bits.

    bits=8 → 256 levels  (essentially lossless for perception)
    bits=4 → 16 levels
    bits=2 → 4 levels
    bits=1 → 2 levels  (binary hologram)

    Returns float32 phase array in [-π, π].
    """
    n_levels = 2 ** bits
    # Map [-π, π] → [0, 1]
    normalized = (phase + np.pi) / (2 * np.pi)
    # Quantise
    quantized = np.round(normalized * (n_levels - 1)) / (n_levels - 1)
    # Map back to [-π, π]
    return (quantized * 2 * np.pi - np.pi).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Color channel degradation
# ─────────────────────────────────────────────────────────────────────────────

def degrade_color(image_rgb: np.ndarray, mode: str) -> np.ndarray:
    """
    Reduce colour information.

    Parameters
    ----------
    image_rgb : uint8 (H,W,3)
    mode      : 'RGB' | 'RG' | 'mono'

    Returns
    -------
    uint8 (H,W,3) — always 3 channels so comparison is easy
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3

    if mode == "RGB":
        return image_rgb.copy()

    elif mode == "RG":
        out = image_rgb.copy()
        out[:, :, 2] = 0          # zero blue channel
        return out

    elif mode == "mono":
        # Luminance weights (ITU-R BT.601)
        gray = (
            0.299 * image_rgb[:, :, 0].astype(np.float32)
            + 0.587 * image_rgb[:, :, 1].astype(np.float32)
            + 0.114 * image_rgb[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'RGB', 'RG', or 'mono'.")


# ─────────────────────────────────────────────────────────────────────────────
#  4. Viewing angle (spatial-frequency bandwidth limit)
# ─────────────────────────────────────────────────────────────────────────────

def limit_viewing_angle(phase: np.ndarray, bandwidth_fraction: float) -> np.ndarray:
    """
    Limit the spatial-frequency content of a hologram phase to simulate
    a reduced viewing angle (smaller SLM or smaller pixel pitch).

    bandwidth_fraction : float in (0, 1]
        1.0 → full bandwidth (maximum viewing angle)
        0.5 → half bandwidth  (half the viewing angle)
        0.1 → 10% bandwidth   (very narrow viewing angle)

    Strategy:
      Convert phase to complex field → FFT → zero frequencies outside
      circular mask of radius `bandwidth_fraction` → IFFT → take angle.
    """
    H, W = phase.shape
    field = np.exp(1j * phase).astype(np.complex64)
    spectrum = np.fft.fftshift(np.fft.fft2(field))

    # Build circular mask
    cx, cy = W // 2, H // 2
    radius = bandwidth_fraction * min(H, W) / 2
    Y, X = np.ogrid[:H, :W]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

    spectrum_masked = spectrum * mask
    field_back = np.fft.ifft2(np.fft.ifftshift(spectrum_masked))
    return np.angle(field_back).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Speckle noise
# ─────────────────────────────────────────────────────────────────────────────

def add_speckle(image: np.ndarray, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    Add multiplicative speckle noise (common in coherent imaging).

    image : float32 (H,W) in [0,1]
    sigma : standard deviation of the Gaussian noise field

    Returns float32 (H,W) clipped to [0,1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, sigma, image.shape).astype(np.float32)
    noisy = image * noise
    return np.clip(noisy, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Depth-plane simulation helper
# ─────────────────────────────────────────────────────────────────────────────

def depth_planes_to_z_list(n_planes: int, z_near: float = 0.05, z_far: float = 0.30):
    """
    Return a list of z-distances for `n_planes` evenly spaced depth planes.

    n_planes : int  — e.g. 10, 5, 2, 1
    z_near   : float — closest plane [m]
    z_far    : float — farthest plane [m]

    Returns list of floats.
    """
    if n_planes == 1:
        return [(z_near + z_far) / 2]
    return list(np.linspace(z_near, z_far, n_planes))
