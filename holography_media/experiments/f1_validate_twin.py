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
import csv, glob, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import NPDDRecorder, MediumParams, kogelnik_de

torch.set_default_dtype(torch.float64)
N_X, DX = 1024, 0.05  # 51.2 um window, 50 nm sampling
LITERATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "literature")


def sinusoidal_exposure(K_um, visibility=0.9):
    x = torch.arange(N_X) * DX
    return 1.0 + visibility * torch.cos(K_um * x)


CSV_SCHEMA_COLUMNS = ["x", "y", "source_doi", "figure_id", "digitized_by", "date"]


def load_literature_curves():
    """Load digitized (x, y) CSVs from data/literature/, if any exist.

    Phase 6 schema (header row required): x, y, source_doi, figure_id,
    digitized_by, date -- see data/literature/README.md. Only x, y are
    used for plotting/fitting here; the other 4 columns are provenance,
    read and kept alongside but not consumed by this function (the fitting
    script in fit_literature_curves.py surfaces them).

    Returns a dict {filename_stem: (xs, ys)}. Empty if the directory has no
    CSVs yet -- see data/literature/README.md for why and how to add them.
    """
    curves = {}
    for path in sorted(glob.glob(os.path.join(LITERATURE_DIR, "*.csv"))):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
                print(f"[f1] WARNING: {path} does not have the required header "
                      f"({CSV_SCHEMA_COLUMNS}) -- skipping.")
                continue
            xs, ys = [], []
            for row in reader:
                try:
                    xs.append(float(row["x"])); ys.append(float(row["y"]))
                except (ValueError, TypeError):
                    continue
        if xs:
            curves[os.path.splitext(os.path.basename(path))[0]] = (xs, ys)
    if not curves:
        print("[f1] no digitized literature curves found in data/literature/ "
              "-- twin-only validation (see data/literature/README.md).")
    return curves


def main():
    params = MediumParams()  # PVA/AA-like defaults; see configs/media/
    lit_curves = load_literature_curves()
    out = {"literature": lit_curves}
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
