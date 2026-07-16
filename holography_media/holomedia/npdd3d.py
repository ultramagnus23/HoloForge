"""
3D (x, y, z) NPDD recording twin -- showcase extension of npdd.py.

Identical physics to `NPDDRecorder` (Sheridan NPDD, Eqs. 1-3 in the paper),
generalized from a 1D transverse coordinate x to a 2D transverse grid (x, y);
z (depth) still enters only through the readout stage (`diffraction3d.py`),
matching the paper's claim that main results are 2D(x,z) and this case
demonstrates no qualitative behavior is lost in the full 3D volume.

Kept as a separate module (rather than generalizing NPDDRecorder in place)
to avoid touching the tested, paper-scale 1D pipeline that all other
experiments depend on.
"""
from __future__ import annotations
import math
import torch

from .npdd import MediumParams


class NPDDRecorder3D(torch.nn.Module):
    """Exposure I(x,y) -> recorded index modulation profile dn(x,y).

    Same governing equations as NPDDRecorder, with the non-local kernel and
    diffusion operator generalized to a 2D Fourier transform.
    """

    def __init__(self, n_x: int, n_y: int, dx: float, dy: float,
                t_total: float = 10.0, n_steps: int = 300,
                params: MediumParams | None = None, dtype=torch.float64):
        super().__init__()
        self.n_x, self.n_y, self.dx, self.dy = n_x, n_y, dx, dy
        self.t_total, self.n_steps = t_total, n_steps
        self.dt = t_total / n_steps
        self.p = params or MediumParams()
        self.dtype = dtype

        kx = 2.0 * math.pi * torch.fft.fftfreq(n_x, d=dx)
        ky = 2.0 * math.pi * torch.fft.fftfreq(n_y, d=dy)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k2 = (KX ** 2 + KY ** 2).to(dtype)
        self.register_buffer("k2", k2)
        self.register_buffer("G_hat", torch.exp(-0.5 * k2 * (self.p.sigma ** 2)))

    def _nonlocal(self, u: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(u) * self.G_hat).real

    def _diffuse(self, u: torch.Tensor, D_eff: torch.Tensor) -> torch.Tensor:
        D_bar = D_eff.mean()
        decay = torch.exp(-D_bar * self.k2 * self.dt)
        return torch.fft.ifft2(torch.fft.fft2(u) * decay).real

    def forward(self, exposure: torch.Tensor) -> torch.Tensor:
        """exposure: (n_x, n_y) nonnegative intensity pattern -> dn (n_x, n_y)."""
        I = exposure.to(self.dtype)
        u = torch.ones_like(I)
        N = torch.zeros_like(I)
        d = torch.ones_like(I)

        for _ in range(self.n_steps):
            F_loc = self.p.kappa * torch.clamp(I * d, min=0.0) ** self.p.gamma
            poly_rate = self._nonlocal(F_loc * torch.clamp(u, min=0.0))
            u = u - self.dt * poly_rate
            N = N + self.dt * poly_rate
            d = d * torch.exp(-self.p.k_bleach * I * self.dt)
            u = torch.clamp(u, min=0.0)
            D_eff = self.p.D0 * torch.exp(-self.p.alpha_D * N)
            u = self._diffuse(u, D_eff)

        return self.p.dn_max * torch.tanh(1.5 * N)
