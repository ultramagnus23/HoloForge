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


def degrade_resolution_phase(phase: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsample a phase hologram to (target_size × target_size) and restore
    to original resolution — operating in the complex (phasor) domain to
    avoid the uint8 quantization error that would occur in a direct phase
    downsample.

    The correct approach: convert phase to complex phasor, downsample the
    real and imaginary parts separately via nearest-neighbour, then extract
    the angle of the restored complex field.

    Parameters
    ----------
    phase       : float32 (H, W)  — SLM phase in [-π, π]
    target_size : int             — intermediate resolution (e.g. 64 for 64×64)

    Returns
    -------
    float32 (H, W) — phase degraded at target_size resolution, restored to
                     original size
    """
    from PIL import Image as PILImage
    H, W = phase.shape
    # Convert phase to complex phasor
    phasor = np.exp(1j * phase).astype(np.complex64)

    def _resize_channel(arr_real):
        # arr_real is float32 in [-1, 1]
        pil = PILImage.fromarray(((arr_real + 1) / 2 * 255).clip(0, 255).astype(np.uint8), mode='L')
        small = pil.resize((target_size, target_size), PILImage.NEAREST)
        restored = small.resize((W, H), PILImage.NEAREST)
        return np.array(restored).astype(np.float32) / 255.0 * 2 - 1

    real_degraded = _resize_channel(phasor.real)
    imag_degraded = _resize_channel(phasor.imag)
    # Reconstruct complex phasor and extract phase
    phasor_degraded = real_degraded + 1j * imag_degraded
    return np.angle(phasor_degraded).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Phase quantisation
# ─────────────────────────────────────────────────────────────────────────────

def quantise_phase(phase: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantise a phase array to `2**bits` evenly spaced levels over the full
    2π period.

    bits=8 → 256 levels  (essentially lossless for perception)
    bits=4 → 16 levels
    bits=2 → 4 levels
    bits=1 → 2 levels  (genuine binary phase hologram, levels π apart)

    Implementation note
    -------------------
    Phase is periodic, so the quantiser must place `n_levels` bins over the
    *open* interval [-π, π). A naive mapping over the *closed* interval
    [-π, π] using (n_levels - 1) spacing collides the two endpoints
    (e^{-jπ} = e^{+jπ} = -1), yielding only `n_levels - 1` distinct phasors —
    and for bits=1 a degenerate, single-valued (constant) hologram. We instead
    use a mid-rise quantiser: bin centres at (k + 0.5)·(2π/n_levels), which
    gives exactly `n_levels` distinct, evenly spaced phase values with no
    endpoint collision. For bits=1 this yields a true two-level binary phase
    hologram ({-π/2, +π/2}, i.e. phasors π apart).

    Returns float32 phase array in [-π, π).
    """
    n_levels = 2 ** bits
    step = 2 * np.pi / n_levels
    # Map phase to [0, 2π), assign to one of n_levels bins.
    wrapped = np.mod(phase + np.pi, 2 * np.pi)
    idx = np.clip(np.floor(wrapped / step).astype(np.int64), 0, n_levels - 1)
    # Bin centre, shifted back to [-π, π).
    quantised = (idx + 0.5) * step - np.pi
    return quantised.astype(np.float32)


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

def add_speckle_legacy(image: np.ndarray, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    DEPRECATED — physically incorrect speckle model.

    Adds multiplicative Gaussian noise directly to a *reconstructed intensity
    image*. This is NOT how coherent speckle arises: real speckle is a
    coherent-field phenomenon produced by random phase perturbations at the
    aperture (SLM) plane *before* propagation, and it does not reproduce the
    correct Rayleigh-distributed intensity statistics. Retained only for
    backward reference; use `add_speckle_physical` for all new work.

    image : float32 (H,W) in [0,1]
    sigma : standard deviation of the Gaussian noise field

    Returns float32 (H,W) clipped to [0,1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, sigma, image.shape).astype(np.float32)
    noisy = image * noise
    return np.clip(noisy, 0, 1).astype(np.float32)


# Backward-compatible alias (deprecated name).
add_speckle = add_speckle_legacy


def add_speckle_physical(phase: np.ndarray, sigma_rad: float = 0.3, seed: int = 0) -> np.ndarray:
    """
    Physics-correct speckle model: add random phase perturbations to the
    SLM hologram phase BEFORE reconstruction.

    In real coherent systems, surface roughness and SLM phase errors introduce
    random phase noise at the aperture plane. This produces speckle in the
    reconstruction via coherent interference — not additive intensity noise.

    Parameters
    ----------
    phase     : float32 (H,W) — SLM phase hologram in [-π, π]
    sigma_rad : float — std of Gaussian phase noise in radians
                0.0 = no speckle;  0.3 = mild;  1.0 = severe;  π = binary random
    seed      : int

    Returns
    -------
    float32 (H,W) — perturbed phase, wrapped to [-π, π]
    """
    rng = np.random.default_rng(seed)
    phase_noise = rng.normal(0.0, sigma_rad, phase.shape).astype(np.float32)
    perturbed = phase + phase_noise
    # Wrap back to [-π, π]
    return np.angle(np.exp(1j * perturbed)).astype(np.float32)


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
