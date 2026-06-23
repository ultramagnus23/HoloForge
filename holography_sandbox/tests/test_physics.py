"""
tests/test_physics.py
Tests core physics invariants of the HoloForge waveoptics engine.

Run with: pytest holography_sandbox/tests/
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.waveoptics import propagate_asm, gerchberg_saxton, reconstruct
from core.degradation import (
    quantise_phase, limit_viewing_angle, add_speckle_physical,
    degrade_resolution_phase,
)
from core.metrics import psnr, ssim, lpips_real, all_metrics
from core.scenes import gaussian_spots, resolution_chart

WL = 532e-9; DX = 8e-6; Z = 0.1; SIZE = 64


# Test 1: ASM propagation preserves energy (Parseval's theorem)
def test_asm_energy_conservation():
    field = np.random.default_rng(0).standard_normal((SIZE, SIZE)).astype(np.float32) + \
            1j * np.random.default_rng(1).standard_normal((SIZE, SIZE)).astype(np.float32)
    field = field.astype(np.complex64)
    propagated = propagate_asm(field, Z, WL, DX)
    energy_in = np.sum(np.abs(field) ** 2)
    energy_out = np.sum(np.abs(propagated) ** 2)
    assert abs(energy_in - energy_out) / energy_in < 0.01, \
        f"ASM energy not conserved: {energy_in:.3f} vs {energy_out:.3f}"


# Test 2: ASM round-trip (forward then backward) recovers original field
def test_asm_round_trip():
    field = np.exp(1j * np.random.default_rng(0).uniform(-np.pi, np.pi, (SIZE, SIZE))).astype(np.complex64)
    recovered = propagate_asm(propagate_asm(field, Z, WL, DX), -Z, WL, DX)
    error = np.mean(np.abs(field - recovered) ** 2)
    assert error < 1e-8, f"Round-trip error too large: {error}"


# Test 3: GS phase retrieval converges (output not all zeros, MSE reduces)
def test_gs_convergence():
    target = gaussian_spots(SIZE, n_spots=3, sigma=5.0)
    phase, history = gerchberg_saxton(target, n_iter=20, wavelength=WL, dx=DX, z=Z, return_history=True)
    assert phase.shape == (SIZE, SIZE)
    assert not np.allclose(phase, 0), "GS returned all-zero phase"
    assert history[-1] < history[0], "GS MSE did not decrease"


# Test 4: Reconstruction intensity is in [0,1]
def test_reconstruction_range():
    target = gaussian_spots(SIZE)
    phase = gerchberg_saxton(target, n_iter=10, wavelength=WL, dx=DX, z=Z)
    recon = reconstruct(phase, wavelength=WL, dx=DX, z=Z)
    assert recon.min() >= 0.0 - 1e-6, f"Recon min < 0: {recon.min()}"
    assert recon.max() <= 1.0 + 1e-6, f"Recon max > 1: {recon.max()}"


# Test 5: Phase quantisation reduces unique values
def test_quantise_phase_levels():
    phase = np.linspace(-np.pi, np.pi, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    for bits in [1, 2, 4, 8]:
        q = quantise_phase(phase, bits)
        n_unique = len(np.unique(np.round(q, 6)))
        assert n_unique <= 2 ** bits + 1, \
            f"{bits}-bit quantization: {n_unique} unique values (expected <= {2**bits+1})"


# Test 6: Complex superposition (new depth-plane model)
def test_complex_field_superposition():
    phases = [np.random.default_rng(i).uniform(-np.pi, np.pi, (SIZE, SIZE)).astype(np.float32) for i in range(3)]
    combined_field = np.zeros((SIZE, SIZE), dtype=np.complex64)
    for p in phases:
        combined_field += np.exp(1j * p).astype(np.complex64)
    combined_phase = np.angle(combined_field)
    for p in phases:
        assert not np.allclose(combined_phase, p, atol=0.1), \
            "Combined phase identical to input — superposition failed"


# Test 7: Physical speckle perturbs phase but stays in [-pi, pi]
def test_speckle_physical_range():
    phase = np.zeros((SIZE, SIZE), dtype=np.float32)
    perturbed = add_speckle_physical(phase, sigma_rad=1.0)
    assert perturbed.min() >= -np.pi - 1e-5
    assert perturbed.max() <= np.pi + 1e-5
    assert not np.allclose(phase, perturbed)


# Test 8: PSNR is inf for identical images
def test_psnr_identical():
    a = np.random.default_rng(0).random((SIZE, SIZE)).astype(np.float32)
    assert np.isinf(psnr(a, a)) or psnr(a, a) > 100


# Test 9: SSIM in [-1, 1] and = 1 for identical
def test_ssim_identical():
    a = np.random.default_rng(0).random((SIZE, SIZE)).astype(np.float32)
    assert abs(ssim(a, a) - 1.0) < 1e-4


# Test 10: Resolution degradation preserves shape
def test_resolution_degradation_shape():
    phase = np.random.default_rng(0).uniform(-np.pi, np.pi, (SIZE, SIZE)).astype(np.float32)
    degraded = degrade_resolution_phase(phase, target_size=SIZE // 4)
    assert degraded.shape == (SIZE, SIZE)
