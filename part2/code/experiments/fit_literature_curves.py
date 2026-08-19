"""
Phase 6: twin-vs-literature calibration and fit-quality report.

Reads digitized CSVs from data/literature/ (Phase 6 schema: x,y,source_doi,
figure_id,digitized_by,date -- see data/literature/README.md), calibrates
the twin's free parameters against each curve, and reports RMSE + residual
structure honestly -- a poor fit is reported as a poor fit (ground rule 4),
not smoothed over.

Curve type is inferred from the filename (a documented convention, not a
guess): a filename containing "growth_dn" is a raw Delta-n1-vs-dose curve
at a fixed spatial frequency K (K given by a "_K<value>" token in the
filename, e.g. bruder2017_growth_dn_K8.98_exp.csv -> K=8.98 rad/um); a
filename containing "growth" (but not "growth_dn") is a DE-vs-exposure-
time curve at a fixed K; a filename containing "angular" is a DE-vs-
angular-detuning curve. Filenames matching none of these are skipped with
a printed warning rather than guessed at.

growth vs. growth_dn is a real distinction, not a naming nicety: some
published curves report diffraction efficiency (post-Kogelnik, needs a
thickness/wavelength to convert from Delta-n), others report the
recorded refractive-index modulation Delta-n1 directly against a
physical dose axis (mJ/cm^2) rather than the twin's internal exposure-
time units. growth_dn fits Delta-n1 directly (NPDDRecorder's raw output,
no Kogelnik conversion) and treats the dose axis as directly proportional
to the twin's t_total -- valid because MediumParams.kappa already "folds
in intensity scale" (see holomedia/npdd.py's docstring: F0 = kappa *
I_mean^gamma), so fitting kappa against a dose axis with the model's
exposure amplitude held at 1.0 is absorbing exactly the dose/time-unit
conversion into the same free parameter already used everywhere else.

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
import yaml
from scipy.optimize import least_squares

from holomedia import NPDDRecorder, MediumParams, kogelnik_de

torch.set_default_dtype(torch.float64)

LITERATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "literature")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "media")
CSV_SCHEMA_COLUMNS = ["x", "y", "source_doi", "figure_id", "digitized_by", "date"]

# Map a filename prefix (the part before the first "_") to the medium config
# it should be fit against, instead of the generic PVA/acrylamide-calibrated
# MediumParams() default. This matters: dn_max is held fixed during fitting
# (only kappa/D0 are free, per the master prompt's "kappa, plus at most one
# more"), and dn_max differs by a real physical factor between media
# families -- fitting a Bayfol HX curve (high-Dn material) against the
# generic default's dn_max=3.5e-3 caps the model well below the curve's own
# peak (~0.016) no matter what kappa/D0 are chosen, which looks like a
# fitting failure but is actually a medium-family mismatch. Discovered by
# fitting bruder2017's real digitized curve against the generic default and
# getting a flat, saturated-from-the-start model curve (NRMSE ~0.9) -- see
# git log for that finding before this mapping was added.
FILENAME_PREFIX_TO_CONFIG = {
    "bruder2017": "bayfol_hx_405nm.yaml",
    "hsieh2022": "pq_pmma_405nm.yaml",
}


def load_medium_config(yaml_name: str) -> MediumParams:
    path = os.path.join(CONFIGS_DIR, yaml_name)
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = MediumParams()
    known_fields = set(defaults.__dict__.keys())
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    ignored = set(raw.keys()) - known_fields
    if ignored:
        print(f"[fit_literature_curves] {yaml_name}: ignoring non-MediumParams "
              f"keys {sorted(ignored)}")
    return MediumParams(**{**defaults.__dict__, **filtered})


def base_params_for_file(filename: str) -> MediumParams:
    stem = os.path.splitext(os.path.basename(filename))[0]
    prefix = stem.split("_")[0]
    yaml_name = FILENAME_PREFIX_TO_CONFIG.get(prefix)
    if yaml_name is None:
        return MediumParams()
    return load_medium_config(yaml_name)

N_X, DX = 512, 0.02  # fit-time grid: 10.24um window. DX tightened from the
# original 0.1 (checked: 0.1 gives only ~2.5 samples/period at
# hsieh2022's K=24.94 rad/um, below what a spectral method needs for an
# accurate amplitude -- 0.02 gives ~12.6). Cost-free: NPDDRecorder's
# per-step cost scales with N_X (unchanged), not DX; N_X=512 at DX=0.02
# still covers >>1 non-local kernel width (sigma) for every medium config
# in configs/media/, so periodicity/boundary effects stay negligible.
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
    if "growth_dn" in stem:
        return "growth_dn", K
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


def simulate_growth_dn(dose_values, K, kappa, D0, base_params: MediumParams):
    """Raw |Delta-n1| at spatial frequency K vs a list of physical dose
    values (e.g. mJ/cm^2), for the given (kappa, D0). No Kogelnik
    conversion -- this is for curves that report the recorded index
    modulation directly, not diffraction efficiency. Dose is passed to
    NPDDRecorder as t_total directly (exposure amplitude held at 1.0);
    kappa absorbs the physical dose/time-unit conversion, same as every
    other use of kappa in this codebase."""
    p = MediumParams(**{**base_params.__dict__, "kappa": kappa, "D0": D0})
    x = torch.arange(N_X) * DX
    dns = []
    for dose in dose_values:
        # Capped at 500 (NPDDRecorder's own docstring: "200-500 adequate
        # for the parameter ranges above", IMEX scheme -- stability isn't
        # tied to raw t_total). The uncapped `20*dose` heuristic this was
        # copied from was written when dose/t_total values were always
        # small abstract units (1-18, see f1_validate_twin.py); it breaks
        # for curves reporting real physical exposure time in seconds
        # (checked: hsieh2022's t axis runs to 2000s, which would ask for
        # 40,000 steps uncapped).
        n_steps = min(max(20, int(20 * max(dose, 1e-3))), 500)
        rec = NPDDRecorder(N_X, DX, t_total=float(dose), n_steps=n_steps, params=p)
        exposure = 1.0 + 0.9 * torch.cos(K * x)
        dn = rec(exposure)
        dn1 = 2.0 * torch.mean(dn * torch.cos(K * x))
        dns.append(float(dn1.abs()))
    return np.array(dns)


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
# Second free parameter per curve, alongside kappa (always free). Default is
# D0 (monomer diffusivity), matching the master prompt's "kappa, plus at
# most one more." growth_dn curves use dn_max instead -- diagnostic work
# (see docs/parameter_provenance.md and the Gate-A report) showed the
# original {kappa, D0} choice was fitting the wrong knob: a 3-parameter
# diagnostic fit isolated dn_max, not D0, as what actually explained the
# residual on both real growth_dn sources (NRMSE dropped from 0.68-0.87 to
# 0.10-0.35 when dn_max replaced D0, at D0 held at its cited value). D0
# stays the default for "growth"/"angular" curve types since no real data
# has exercised that choice yet -- don't generalize past what was checked.
SECOND_PARAM_BY_CURVE_TYPE = {
    "growth": "D0",
    "growth_dn": "dn_max",
    "angular": "D0",
}

PARAM_BOUNDS = {
    "kappa": (1e-3, 1e3),
    # Widened past the generic docstring range (D0: 1e-3-1e0) -- checked:
    # configs/media/pq_pmma_405nm.yaml's real cited D0=1.24e-6 sits below
    # it, and an unwidened bound threw "Initial guess is outside of
    # provided bounds" on exactly that curve. Bounds must contain every
    # real config's own starting point, not just the generic default's.
    "D0": (1e-8, 1e2),
    "dn_max": (1e-4, 0.3),
}


def fit_curve(curve_type: str, K: float, xs: list, ys: list,
             base_params: MediumParams | None = None,
             thickness_um: float = 30.0, n_starts: int = 10,
             seed: int = 0, second_param: str | None = None) -> dict:
    """Bounded least-squares fit of (kappa, second_param) against a
    digitized curve, from n_starts random log-uniform initializations
    (default 10 -- a single-start result is not reported as a fit; a
    5-ish-parameter NPDD forward model has local minima even with only 2
    of those parameters free). Returns the best-of-n_starts fit plus the
    spread of final NRMSE across all starts, so a flat-looking "good" fit
    that only one of ten starts found can be told apart from a robust one.

    second_param defaults to SECOND_PARAM_BY_CURVE_TYPE[curve_type] if not
    given explicitly; callers (e.g. diagnostic scripts, tests) can override
    it to fit a specific parameter instead."""
    base_params = base_params or MediumParams()
    ys_arr = np.array(ys)
    if second_param is None:
        second_param = SECOND_PARAM_BY_CURVE_TYPE[curve_type]

    def make_model_fn(second_val):
        p = MediumParams(**{**base_params.__dict__, second_param: second_val})
        if curve_type == "growth":
            return lambda kappa: simulate_growth_de(xs, K, kappa, p.D0, p, thickness_um)
        elif curve_type == "growth_dn":
            return lambda kappa: simulate_growth_dn(xs, K, kappa, p.D0, p)
        elif curve_type == "angular":
            return lambda kappa: simulate_angular_de(xs, K, kappa, p.D0, p, thickness_um)
        raise ValueError(f"unsupported curve_type {curve_type!r} (expected 'growth', "
                         f"'growth_dn', or 'angular')")

    def model_fn(kappa, second_val):
        return make_model_fn(second_val)(kappa)

    def residuals(log_params):
        kappa, second_val = np.exp(log_params)
        return model_fn(kappa, second_val) - ys_arr

    kappa_lo, kappa_hi = PARAM_BOUNDS["kappa"]
    second_lo, second_hi = PARAM_BOUNDS[second_param]
    log_bounds = (np.log([kappa_lo, second_lo]), np.log([kappa_hi, second_hi]))

    base_second = getattr(base_params, second_param)
    rng = np.random.default_rng(seed)
    # First start is always the literature-cited value (matches the old
    # single-start behavior); the rest are log-uniform over the full
    # bounded range, so a fit that only works from the cited value isn't
    # silently reported as if it were reachable from anywhere reasonable.
    starts = [np.log([base_params.kappa, base_second])]
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(log_bounds[0], log_bounds[1]))

    attempts = []
    for x0 in starts:
        result = least_squares(residuals, x0, method="trf", bounds=log_bounds, max_nfev=200)
        kappa_fit, second_fit = np.exp(result.x)
        model_y = model_fn(kappa_fit, second_fit)
        resid = model_y - ys_arr
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        y_range = float(ys_arr.max() - ys_arr.min())
        nrmse = float(rmse / y_range) if y_range > 0 else float("nan")
        at_bound = (np.isclose(kappa_fit, kappa_lo, rtol=1e-2) or
                   np.isclose(kappa_fit, kappa_hi, rtol=1e-2) or
                   np.isclose(second_fit, second_lo, rtol=1e-2) or
                   np.isclose(second_fit, second_hi, rtol=1e-2))
        attempts.append(dict(kappa_fit=float(kappa_fit), second_fit=float(second_fit),
                             rmse=rmse, nrmse=nrmse, converged=bool(result.success),
                             at_bound=bool(at_bound), cost=float(result.cost),
                             model_y=model_y.tolist()))

    best = min(attempts, key=lambda a: a["cost"])
    nrmse_spread = [a["nrmse"] for a in attempts]

    return dict(curve_type=curve_type, K=K, second_param=second_param,
               kappa_fit=best["kappa_fit"], second_param_fit=best["second_fit"],
               # kept for backward-compat with code/macros keyed on D0_fit
               D0_fit=(best["second_fit"] if second_param == "D0" else base_params.D0),
               rmse=best["rmse"], nrmse=best["nrmse"],
               x=xs, y_data=ys, y_model=best["model_y"], residuals=(np.array(best["model_y"]) - ys_arr).tolist(),
               converged=best["converged"], at_bound=best["at_bound"], n_points=len(xs),
               n_starts=n_starts, nrmse_min=float(min(nrmse_spread)),
               nrmse_max=float(max(nrmse_spread)), nrmse_std=float(np.std(nrmse_spread)))


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
        base_params = base_params_for_file(path)
        print(f"[fit_literature_curves] fitting {os.path.basename(path)} "
              f"(type={curve_type}, K={K}, n={len(data['x'])} points, "
              f"base_params dn_max={base_params.dn_max:.4g}) ...")
        fit = fit_curve(curve_type, K, data["x"], data["y"], base_params=base_params)
        fit.update(source_doi=data["source_doi"], figure_id=data["figure_id"],
                  digitized_by=data["digitized_by"], date=data["date"],
                  file=os.path.basename(path))
        # NRMSE (RMSE / data range), not raw RMSE, drives the quality bucket --
        # raw RMSE is not comparable across curve types on very different
        # scales (DE is O(1), Delta-n1 is O(0.01)); a fixed RMSE threshold
        # would call a garbage Delta-n1 fit "GOOD" just because the numbers
        # are small.
        # Thresholds set to the Gate-A/B decision-tree values (NRMSE <0.3 =
        # good/straightforward validation, 0.3-0.5 = moderate/mechanism-
        # validity standard, >0.5 = poor), not the earlier 0.05/0.15 --
        # those were calibrated like a numerical self-consistency check,
        # not a realistic bar for fitting messy digitized literature data.
        # Keeping one threshold scheme (here, in figure labels, and in the
        # paper's prose) avoids a figure saying "POOR" next to a number the
        # text calls "good."
        quality = ("GOOD" if fit["nrmse"] < 0.3 else
                  "MODERATE" if fit["nrmse"] < 0.5 else "POOR")
        fit["fit_quality"] = quality
        bound_note = " [AT BOUND -- not a real optimum]" if fit["at_bound"] else ""
        print(f"  kappa={fit['kappa_fit']:.4g} {fit['second_param']}={fit['second_param_fit']:.4g} "
              f"RMSE={fit['rmse']:.4f} NRMSE={fit['nrmse']:.4f} ({quality}){bound_note}")
        print(f"  multi-start (n={fit['n_starts']}): NRMSE range "
              f"[{fit['nrmse_min']:.4f}, {fit['nrmse_max']:.4f}], std={fit['nrmse_std']:.4f}")
        reports.append(fit)

    out_path = os.path.join(LITERATURE_DIR, "..", "..", "results_literature_fit.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(dict(fits=reports), f, indent=1)
    print(f"wrote {out_path}")
    return reports


if __name__ == "__main__":
    main()
