"""
3D split-step scalar BPM: readout of a (x, y) recorded index profile,
propagated through the slab in z, then to the observation plane -- showcase
extension of diffraction.py's SlabBPM. See npdd3d.py for the matching
recording-twin generalization.
"""
from __future__ import annotations
import math
import torch


class SlabBPM3D(torch.nn.Module):
    def __init__(self, n_x: int, n_y: int, dx: float, dy: float,
                wavelength_um: float, thickness_um: float,
                n_z: int = 24, n0: float = 1.5, z_recon_um: float = 5.0e4,
                dtype=torch.complex128):
        super().__init__()
        self.n_x, self.n_y, self.dx, self.dy = n_x, n_y, dx, dy
        self.lam, self.n0 = wavelength_um, n0
        self.T, self.n_z = thickness_um, n_z
        self.dz = thickness_um / n_z
        self.z_recon = z_recon_um
        self.cdtype = dtype

        fx = torch.fft.fftfreq(n_x, d=dx)
        fy = torch.fft.fftfreq(n_y, d=dy)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")
        k0 = 2 * math.pi / wavelength_um

        def asm_kernel(dist, n_medium):
            arg = (n_medium / wavelength_um) ** 2 - FX ** 2 - FY ** 2
            kz = 2 * math.pi * torch.sqrt(torch.clamp(arg, min=0.0))
            H = torch.exp(1j * kz * dist)
            H = torch.where(arg > 0, H, torch.zeros_like(H))
            return H.to(dtype)

        self.register_buffer("H_slab", asm_kernel(self.dz, n0))
        self.register_buffer("H_free", asm_kernel(z_recon_um, 1.0))
        self.k0 = k0

    def forward(self, dn_profile: torch.Tensor,
                incident: torch.Tensor | None = None) -> torch.Tensor:
        """dn_profile: (n_x, n_y) -> far-field intensity (n_x, n_y)."""
        E = (torch.ones(self.n_x, self.n_y, dtype=self.cdtype, device=dn_profile.device)
             if incident is None else incident.to(self.cdtype))
        dn_c = dn_profile.to(E.real.dtype)
        for _ in range(self.n_z):
            E = E * torch.exp(1j * self.k0 * dn_c * self.dz)
            E = torch.fft.ifft2(torch.fft.fft2(E) * self.H_slab)
        E = torch.fft.ifft2(torch.fft.fft2(E) * self.H_free)
        return (E.real ** 2 + E.imag ** 2)
