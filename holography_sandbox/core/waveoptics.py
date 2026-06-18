"""
waveoptics.py
-------------
Core wave-optics engine for the Computational Perceptual Holography Sandbox.

Implements:
  - Angular Spectrum Method (ASM) propagation
  - Gerchberg-Saxton (GS) phase retrieval
  - Fresnel propagation (thin-lens approximation)

All fields are complex numpy arrays: shape (H, W), dtype complex64.
Conventions:
  wavelength, pixel_pitch in metres
  z (propagation distance) in metres, positive = forward
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freq_grid(H: int, W: int, dx: float):
    """
    Return spatial-frequency grids (fx, fy) in cycles/metre.
    dx = pixel pitch in metres.
    """
    fx = np.fft.fftfreq(W, d=dx).astype(np.float32)
    fy = np.fft.fftfreq(H, d=dx).astype(np.float32)
    return np.meshgrid(fx, fy)


def _transfer_function(H: int, W: int, dx: float, wavelength: float, z: float):
    """
    Angular Spectrum transfer function H(fx,fy,z).
    Evanescent waves are zeroed out.
    """
    fx, fy = _freq_grid(H, W, dx)
    k = 1.0 / wavelength                  # wavenumber in cycles/metre
    kz_sq = k**2 - fx**2 - fy**2
    propagating = kz_sq >= 0              # mask out evanescent waves
    kz = np.where(propagating, np.sqrt(np.maximum(kz_sq, 0)), 0).astype(np.float32)
    H_tf = np.exp(1j * 2 * np.pi * kz * z).astype(np.complex64)
    H_tf[~propagating] = 0.0
    return H_tf


# ─────────────────────────────────────────────────────────────────────────────
#  Propagation
# ─────────────────────────────────────────────────────────────────────────────

def propagate_asm(field: np.ndarray, z: float, wavelength: float, dx: float) -> np.ndarray:
    """
    Angular Spectrum Method propagation.

    Parameters
    ----------
    field      : complex64 array (H, W)  — input complex field
    z          : float  — propagation distance [m], can be negative
    wavelength : float  — [m], e.g. 532e-9 for green laser
    dx         : float  — pixel pitch [m], e.g. 8e-6

    Returns
    -------
    complex64 array (H, W) — propagated field
    """
    H, W = field.shape
    tf = _transfer_function(H, W, dx, wavelength, z)
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * tf).astype(np.complex64)


def propagate_fresnel(field: np.ndarray, z: float, wavelength: float, dx: float) -> np.ndarray:
    """
    Fresnel (paraxial) propagation via convolution approach.
    Faster than ASM for large z but less accurate at steep angles.
    """
    H, W = field.shape
    k = 2 * np.pi / wavelength
    x = np.fft.fftfreq(W, d=dx).astype(np.float32)
    y = np.fft.fftfreq(H, d=dx).astype(np.float32)
    fx, fy = np.meshgrid(x, y)
    # Fresnel transfer function
    H_tf = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (fx**2 + fy**2))
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * H_tf).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase retrieval — Gerchberg-Saxton
# ─────────────────────────────────────────────────────────────────────────────

def gerchberg_saxton(
    target_amplitude: np.ndarray,
    n_iter: int = 50,
    wavelength: float = 532e-9,
    dx: float = 8e-6,
    z: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Classic Gerchberg-Saxton algorithm.

    Finds a phase-only hologram (at the SLM plane) that reconstructs
    `target_amplitude` at the image plane (distance z away).

    Parameters
    ----------
    target_amplitude : float32 array (H, W)  — desired intensity pattern [0..1]
    n_iter           : int    — number of GS iterations
    wavelength       : float  — [m]
    dx               : float  — SLM pixel pitch [m]
    z                : float  — propagation distance [m]

    Returns
    -------
    phase_hologram : float32 array (H, W)  — phase in [-π, π]
    """
    rng = np.random.default_rng(seed)
    H, W = target_amplitude.shape

    # Start with random phase at SLM plane, uniform amplitude
    slm_phase = rng.uniform(-np.pi, np.pi, (H, W)).astype(np.float32)
    slm_field = np.exp(1j * slm_phase).astype(np.complex64)

    target_amp = target_amplitude.astype(np.float32)

    for _ in range(n_iter):
        # Forward propagate to image plane
        img_field = propagate_asm(slm_field, z, wavelength, dx)
        # Replace amplitude with target, keep phase
        img_phase = np.angle(img_field)
        img_field_constrained = target_amp * np.exp(1j * img_phase)
        # Back-propagate to SLM plane
        slm_field = propagate_asm(img_field_constrained, -z, wavelength, dx)
        # Enforce phase-only constraint at SLM
        slm_field = np.exp(1j * np.angle(slm_field)).astype(np.complex64)

    return np.angle(slm_field).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Reconstruction from hologram
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(
    phase_hologram: np.ndarray,
    wavelength: float = 532e-9,
    dx: float = 8e-6,
    z: float = 0.1,
) -> np.ndarray:
    """
    Simulate optical reconstruction of a phase hologram.

    Returns the intensity image (float32, H×W) at the image plane.
    """
    slm_field = np.exp(1j * phase_hologram).astype(np.complex64)
    img_field = propagate_asm(slm_field, z, wavelength, dx)
    intensity = np.abs(img_field) ** 2
    # Normalise to [0, 1]
    intensity /= intensity.max() + 1e-12
    return intensity.astype(np.float32)
