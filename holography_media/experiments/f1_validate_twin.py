"""
F1 -- Twin validation (paper Figure 1).

(a) DE growth curves vs exposure time for sinusoidal exposure at several
    spatial frequencies: must reproduce the NPDD signature -- growth then
    saturation, with high-K rolloff governed by R = D0 K^2 / F0.
(b) Small-signal MTF H(K) from the linearized model overlaid on the
    contrast actually recorded by the full nonlinear simulator.
(c) Kogelnik angular selectivity for a recorded grating.

Digitized literature curves (WebPlotDigitizer CSVs) go in data/literature/
and are overlaid by plot_f1.py -- see paper Sec. 3.4 for sources to digitize
(Sheridan NPDD growth curves; PVA/AA DE-vs-dose curves).
"""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import NPDDRecorder, MediumParams, kogelnik_de

torch.set_default_dtype(torch.float64)
N_X, DX = 1024, 0.05  # 51.2 um window, 50 nm sampling


def sinusoidal_exposure(K_um, visibility=0.9):
    x = torch.arange(N_X) * DX
    return 1.0 + visibility * torch.cos(K_um * x)


def main():
    params = MediumParams()  # PVA/AA-like defaults; see configs/media/
    out = {}
    for K in [2.0, 6.0, 12.0, 20.0]:  # rad/um; K=2pi/Lambda
        des, ts = [], []
        for t_total in [1, 2, 4, 6, 8, 10, 14, 18]:
            rec = NPDDRecorder(N_X, DX, t_total=t_total, n_steps=200, params=params)
            dn = rec(sinusoidal_exposure(K))
            # recorded contrast at K -> first-harmonic amplitude
            x = torch.arange(N_X) * DX
            dn1 = 2.0 * torch.mean(dn * torch.cos(K * x))
            de = float(kogelnik_de(dn1.abs(), params.thickness, 0.405))
            des.append(de); ts.append(t_total)
        out[K] = (ts, des)
        print(f"K={K:5.1f} rad/um  DE growth: " +
              " ".join(f"{d:.3f}" for d in des))

    # small-signal MTF vs measured contrast at fixed dose
    rec = NPDDRecorder(N_X, DX, t_total=10, n_steps=300, params=params)
    Ks = torch.linspace(1.0, 30.0, 15)
    H_pred = rec.small_signal_mtf(Ks)
    print("\nPredicted H(K):", [f"{h:.3f}" for h in H_pred.tolist()])
    print(f"Predicted compensation cliff (4x budget): "
          f"K_c = {rec.predicted_cliff(budget=4.0):.2f} rad/um")
    torch.save(out, "results_f1.pt")


if __name__ == "__main__":
    main()
