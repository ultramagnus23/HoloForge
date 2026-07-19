"""
Shared plumbing for the GPU-scale sweep scripts (gpu_npdd_mesh_convergence_sweep.py,
gpu_bpm_wavelength_sweep.py): device selection, incremental checkpointing, and
per-run runtime/memory logging.

Not used by any CPU-scale script (run_confirm.py, f2_f3_recovery.py, etc.) --
those are untouched and their results are left as prior-scale reference data.
"""
from __future__ import annotations
import json
import os
import time

import torch


def get_device(require_gpu: bool = False) -> torch.device:
    """Pick cuda if present; refuses to silently fall back to CPU when
    require_gpu=True, since a CPU run must never be reported as GPU-scale."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError(
            "No CUDA device visible (torch.cuda.is_available() == False). "
            "On Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU, "
            "then re-run. Refusing to silently run this as a CPU job and label it GPU-scale."
        )
    print("[gpu_common] WARNING: no CUDA device -- running on CPU. "
          "Do not report this run's results as GPU-scale.")
    return torch.device("cpu")


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)  # atomic-ish; avoids truncated file on interrupt


class RunTimer:
    """Wraps one sweep cell: wall-clock + peak GPU memory, written into the
    result dict so the methods section has real numbers, not estimates."""

    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.wall_s = time.time() - self.t0
        self.peak_mem_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1e6
            if self.device.type == "cuda" else None
        )

    def as_dict(self) -> dict:
        return dict(wall_s=self.wall_s, peak_mem_mb=self.peak_mem_mb,
                    device=str(self.device))
