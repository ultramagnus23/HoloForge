"""
Volume diffraction models for recorded index profiles.

Two tiers:
  1. Kogelnik coupled-wave theory (closed form) -- validation tier for pure
     sinusoidal gratings; reproduces DE vs dn*T and angular selectivity.
  2. Split-step beam propagation (BPM) through the recorded slab -- the
     general engine used inside the optimization loop for arbitrary
     exposure patterns. The recorded transverse profile dn(x) is extruded
     through the slab thickness with an optional shrinkage-induced
     longitudinal compression (Bragg detuning).

Scalar, 2D (x, z). Validity bounds and RCWA cross-check discussed in
paper Sec. 6; see tests/test_kogelnik.py for agreement checks.
"""

from __future__ import annotations
import math
import torch


# ------------------------------------------------------------------ Kogelnik
def kogelnik_de(dn: torch.Tensor, thickness_um: float, wavelength_um: float,
                n0: float = 1.5, theta_B: float | None = None,
                dtheta: torch.Tensor | None = None):
    """Diffraction efficiency of an unslanted transmission volume grating.

    eta = sin^2( sqrt(nu^2 + xi^2) ) * nu^2 / (nu^2 + xi^2)
    nu  = pi dn T / (lambda cos(theta_B))
    xi  = detuning parameter (0 at Bragg); xi = dtheta * K * T / 2
    """
    lam = wavelength_um
    if theta_B is None:
        theta_B = math.radians(10.0)
    nu = math.pi * dn * thickness_um / (lam * math.cos(theta_B))
    if dtheta is None:
        return torch.sin(nu) ** 2
    # angular selectivity around Bragg (grating vector K from Bragg condition)
    K = 4.0 * math.pi * n0 * math.sin(theta_B) / lam
    xi = dtheta * K * thickness_um / 2.0
    s = torch.sqrt(nu ** 2 + xi ** 2)
    return (torch.sin(s) ** 2) * (nu ** 2) / (nu ** 2 + xi ** 2 + 1e-12)


# ----------------------------------------------------------------------- BPM
class SlabBPM(torch.nn.Module):
    """Split-step scalar BPM through the recorded volume, then ASM to far plane.

    Field is 1D in x; slab is sliced into n_z steps. Each step:
        phase kick  : exp(i k0 dn(x,z) dz)
        propagation : band-limited angular spectrum over dz inside medium
    After the slab, free-space ASM propagates to the reconstruction plane.
    """

    def __init__(self, n_x: int, dx: float, wavelength_um: float,
                 thickness_um: float, n_z: int = 32, n0: float = 1.5,
                 z_recon_um: float = 5.0e4, dtype=torch.complex128):
        super().__init__()
        self.n_x, self.dx = n_x, dx
        self.lam, self.n0 = wavelength_um, n0
        self.T, self.n_z = thickness_um, n_z
        self.dz = thickness_um / n_z
        self.z_recon = z_recon_um
        self.cdtype = dtype

        fx = torch.fft.fftfreq(n_x, d=dx)
        k0 = 2 * math.pi / wavelength_um

        def asm_kernel(dist, n_medium):
            arg = (n_medium / wavelength_um) ** 2 - fx ** 2
            kz = 2 * math.pi * torch.sqrt(torch.clamp(arg, min=0.0))
            H = torch.exp(1j * kz * dist)
            H = torch.where(arg > 0, H, torch.zeros_like(H))  # band-limit
            return H.to(dtype)

        self.register_buffer("H_slab", asm_kernel(self.dz, n0))
        self.register_buffer("H_free", asm_kernel(z_recon_um, 1.0))
        self.k0 = k0

    def forward(self, dn_profile: torch.Tensor, shrinkage: float = 0.0,
                slant_deg: float = 20.0,
                incident: torch.Tensor | None = None) -> torch.Tensor:
        """Propagate a (plane-wave by default) readout beam; return far-field
        intensity at the reconstruction plane.

        Shrinkage model (v2): for fringes slanted at `slant_deg` to the surface
        normal, longitudinal shrinkage s rotates/compresses the grating vector,
        which at readout appears as a depth-dependent LATERAL shift of the
        recorded pattern: dx(z) = s * tan(slant) * z. Implemented as an exact
        per-slice FFT shift (differentiable). For unslanted transmission
        fringes (slant=0) shrinkage correctly has almost no effect -- the
        detuning is a slanted/reflection-geometry phenomenon (Kogelnik;
        photopolymer shrinkage literature).

        dn_profile may carry leading batch dimensions (..., n_x); a plane
        wave is then initialized per batch row via dn_profile.shape rather
        than the fixed self.n_x, so many gratings can be propagated in one
        vectorized call.
        """
        E = (torch.ones(dn_profile.shape, dtype=self.cdtype, device=dn_profile.device)
             if incident is None else incident.to(self.cdtype))
        dz_eff = self.dz * (1.0 - shrinkage)
        tan_phi = math.tan(math.radians(slant_deg))
        fx = torch.fft.fftfreq(self.n_x, d=self.dx).to(dn_profile.device)
        dn_hat = torch.fft.fft(dn_profile.to(torch.complex128))
        for iz in range(self.n_z):
            z = (iz + 0.5) * self.dz
            shift = shrinkage * tan_phi * z
            dn_z = torch.fft.ifft(
                dn_hat * torch.exp(-2j * math.pi * fx * shift)).real
            E = E * torch.exp(1j * self.k0 * dn_z.to(E.real.dtype) * dz_eff)
            E = torch.fft.ifft(torch.fft.fft(E) * self.H_slab)
        E = torch.fft.ifft(torch.fft.fft(E) * self.H_free)
        return (E.real ** 2 + E.imag ** 2)
