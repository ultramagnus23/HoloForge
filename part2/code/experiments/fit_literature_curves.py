"""
Phase 6: twin-vs-literature calibration and fit-quality report.

Reads digitized CSVs from data/literature/ (Phase 6 schema: x,y,source_doi,
figure_id,digitized_by,date -- see data/literature/README.md), calibrates
the twin's free parameters against each curve, and reports RMSE + residual
structure honestly -- a poor fit is reported as a poor fit (ground rule 4),
not smoothed over.

Curve type is inferred from the filename (a documented convention, not a
guess): a filename containing "growth" is a DE-vs-exposure-time curve at
a fixed spatial frequency K (K given by a "_K<value>" token in the
filename, e.g. sheridan2011_growth_K6.csv -> K=6 rad/um); a filename
containing "angular" is a DE-vs-angular-detuning curve. Filenames matching
neither pattern are skipped with a printed warning rather than guessed at.

Free parameters: kappa (dose sensitivity) and D0 (monomer diffusivity) --
exactly 2, matching the master prompt's "kappa, plus at most one more."
Both are physically meaningful knobs on the SAME NPDD forward model
already used everywhere else in this codebase (holomedia.npdd), not a
separate fitting-only model. Every other MediumParams field is held at
its Table-1-sourced default while fitting.

Honest scope as of this pass: data/literature/ has no real digitized
CSVs yet (Phase 6 is explicitly your task -- WebPlotDigitizer on paywalled
figures). This script is verified via tests/test_fit_literature_curves.py
against a SYNTHETIC fixture (twin-generated "digitized" data with known
kappa/D0 plus noise, clearly not real literature) to prove the fitting
mechanics correctly recover known parameters -- that is a self-consistency
check of the CODE, not a validation claim about real literature agreement.
"""
from __future__ import annotations
import csv
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.optimize import least_squares

from holomedia import NPDDRecorder, MediumParams, kogelnik_de

torch.set_default_dtype(torch.float64)

LITERATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "literature")
CSV_SCHEMA_COLUMNS = ["x", "y", "source_doi", "figure_id", "digitized_by", "date"]

N_X, DX = 512, 0.1  # fit-time grid: lighter than F1's 1024/0.05, adequate for a scalar fit
WAVELENGTH_UM = 0.405


def load_curve_csv(path: str) -> dict:
    """Full Phase-6-schema read (x, y, plus provenance)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required x,y columns "
                             f"(schema: {CSV_SCHEMA_COLUMNS})")
        rows = list(reader)
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    prov = rows[0] if rows else {}
    return dict(x=xs, y=ys,
               source_doi=prov.get("source_doi", "unknown"),
               figure_id=prov.get("figure_id", "unknown"),
               digitized_by=prov.get("digitized_by", "unknown"),
               date=prov.get("date", "unknown"))


def infer_curve_type_and_K(filename: str) -> tuple[str, float | None]:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"_K([\d.]+)", stem)
    K = float(m.group(1)) if m else None
    if "growth" in stem:
        return "growth", K
    if "angular" in stem:
        return "angular", K
    return "unknown", K


# --------------------------------------------------------------- forward models
def simulate_growth_de(t_values, K, kappa, D0, base_params: MediumParams,
                       thickness_um: float, wavelength_um: float = WAVELENGTH_UM):
    """DE at spatial frequency K vs a list of exposure times t_values,
    for the given (kappa, D0), everything else from base_params."""
    p = MediumParams(**{**base_params.__dict__, "kappa": kappa, "D0": D0})
    x = torch.arange(N_X) * DX
    des = []
    for t_total in t_values:
        n_steps = max(20, int(20 * t_total))  # scale steps with duration, cheap but stable
        rec = NPDDRecorder(N_X, DX, t_total=float(t_total), n_steps=n_steps, params=p)
        exposure = 1.0 + 0.9 * torch.cos(K * x)
        dn = rec(exposure)
        dn1 = 2.0 * torch.mean(dn * torch.cos(K * x))
        de = float(kogelnik_de(dn1.abs(), thickness_um, wavelength_um))
        des.append(de)
    return np.array(des)


def simulate_angular_de(dtheta_deg_values, K, kappa, D0, base_params: MediumParams,
                        thickness_um: float, t_total: float = 10.0,
                        wavelength_um: float = WAVELENGTH_UM):
    """DE vs angular detuning (degrees) at spatial frequency K, for a
    grating recorded to convergence (fixed t_total) under (kappa, D0)."""
    p = MediumParams(**{**base_params.__dict__, "kappa": kappa, "D0": D0})
    x = torch.arange(N_X) * DX
    rec = NPDDRecorder(N_X, DX, t_total=t_total, n_steps=300, params=p)
    exposure = 1.0 + 0.9 * torch.cos(K * x)
    dn = rec(exposure)
    dn1 = 2.0 * torch.mean(dn * torch.cos(K * x)).abs()
    des = []
    for dtheta_deg in dtheta_deg_values:
        de = float(kogelnik_de(dn1, thickness_um, wavelength_um,
                              dtheta=torch.tensor(math.radians(dtheta_deg))))
        des.append(de)
    return np.array(des)


# --------------------------------------------------------------- fitting
def fit_curve(curve_type: str, K: float, xs: list, ys: list,
             base_params: MediumParams | None = None,
             thickness_um: float = 30.0) -> dict:
    """Least-squares fit of (kappa, D0) against a digitized curve. Returns
    fitted params, RMSE, and per-point residuals (for the "if the fit is
    poor, that gets reported" requirement)."""
    base_params = base_params or MediumParams()
    xs_arr, ys_arr = np.array(xs), np.array(ys)

    if curve_type == "growth":
        model_fn = lambda kappa, D0: simulate_growth_de(xs, K, kappa, D0, base_params, thickness_um)
    elif curve_type == "angular":
        model_fn = lambda kappa, D0: simulate_angular_de(xs, K, kappa, D0, base_params, thickness_um)
    else:
        raise ValueError(f"unsupported curve_type {curve_type!r} (expected 'growth' or 'angular')")

    def residuals(log_params):
        kappa, D0 = np.exp(log_params)  # fit in log-space: both params are positive, span decades
        return model_fn(kappa, D0) - ys_arr

    x0 = np.log([base_params.kappa, base_params.D0])
    result = least_squares(residuals, x0, method="lm", max_nfev=200)
    kappa_fit, D0_fit = np.exp(result.x)
    model_y = model_fn(kappa_fit, D0_fit)
    resid = model_y - ys_arr
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    return dict(curve_type=curve_type, K=K, kappa_fit=float(kappa_fit),
               D0_fit=float(D0_fit), rmse=rmse,
               x=xs, y_data=ys, y_model=model_y.tolist(), residuals=resid.tolist(),
               converged=bool(result.success), n_points=len(xs))


def main():
    csv_paths = sorted(glob.glob(os.path.join(LITERATURE_DIR, "*.csv")))
    if not csv_paths:
        print("[fit_literature_curves] no CSVs in data/literature/ -- nothing to fit. "
              "See data/literature/README.md for the digitization protocol.")
        return []

    reports = []
    for path in csv_paths:
        curve_type, K = infer_curve_type_and_K(path)
        if curve_type == "unknown" or K is None:
            print(f"[fit_literature_curves] SKIPPED {path}: could not infer curve "
                  f"type/K from filename (expected '..._growth_K<val>.csv' or "
                  f"'..._angular_K<val>.csv')")
            continue
        data = load_curve_csv(path)
        print(f"[fit_literature_curves] fitting {os.path.basename(path)} "
              f"(type={curve_type}, K={K}, n={len(data['x'])} points) ...")
        fit = fit_curve(curve_type, K, data["x"], data["y"])
        fit.update(source_doi=data["source_doi"], figure_id=data["figure_id"],
                  digitized_by=data["digitized_by"], date=data["date"],
                  file=os.path.basename(path))
        quality = ("GOOD" if fit["rmse"] < 0.05 else
                  "MARGINAL" if fit["rmse"] < 0.15 else "POOR")
        fit["fit_quality"] = quality
        print(f"  kappa={fit['kappa_fit']:.3f} D0={fit['D0_fit']:.4f} "
              f"RMSE={fit['rmse']:.4f} ({quality})")
        reports.append(fit)

    out_path = os.path.join(LITERATURE_DIR, "..", "..", "results_literature_fit.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(dict(fits=reports), f, indent=1)
    print(f"wrote {out_path}")
    return reports


if __name__ == "__main__":
    main()
