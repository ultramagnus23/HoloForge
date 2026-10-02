"""
Twin refit with k_bleach free, plus a held-out cross-series prediction
(the part of the twin validation that does not need paywalled data).

Bruder et al. 2017, Fig. 3 reports two Delta-n1(dose) series for Bayfol HX at
the same grating (Lambda = 700 nm, K = 8.98 rad/um): the source's own kinetic
simulation (P_ave = 16.7 mW/cm^2) and its measurement (P_ave = 19.2 mW/cm^2).
Both are plotted against dose (mJ/cm^2), and the twin is dose-reciprocal at
gamma = 1, so parameters fitted to one series make a parameter-free
prediction of the other. That prediction is the held-out test reported here.

For each series and each fit variant:
  A  (kappa, dn_max) free, k_bleach = 0.2 held (the manuscript's earlier fit)
  B  (kappa, dn_max, k_bleach) free
we report the in-sample NRMSE and the held-out NRMSE on the other series
(NRMSE = RMSE / range of the scored series' own data).

Scope, stated here so it is not overstated elsewhere: the two series share a
material, a grating and nearly the same intensity, and one of them is itself
a model output. This tests whether the fitted parameters transfer between two
views of the same condition, not whether the twin extrapolates across K or
media -- that still needs the multi-K growth-curve families (paywalled).

Usage: python experiments/fit_twin_holdout.py   (CPU, ~1 h)
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.optimize import least_squares

from holomedia import MediumParams
from fit_literature_curves import (LITERATURE_DIR, base_params_for_file,
                                   load_curve_csv, simulate_growth_dn)

K = 8.98
SERIES = {"sim": "bruder2017_growth_dn_K8.98_sim.csv",
          "exp": "bruder2017_growth_dn_K8.98_exp.csv"}
BOUNDS = {"kappa": (1e-3, 1e3), "dn_max": (1e-4, 0.3), "k_bleach": (1e-4, 10.0)}
VARIANTS = {"A_kbleach_fixed": ["kappa", "dn_max"],
            "B_kbleach_free": ["kappa", "dn_max", "k_bleach"]}
N_STARTS = 8
OUT = os.path.join(os.path.dirname(__file__), "..", "results_twin_holdout.json")


def model(xs, base: MediumParams, free: dict) -> np.ndarray:
    p = MediumParams(**{**base.__dict__, **free})
    return simulate_growth_dn(xs, K, p.kappa, p.D0, p)


def nrmse(pred, ys) -> float:
    ys = np.asarray(ys)
    return float(np.sqrt(np.mean((np.asarray(pred) - ys) ** 2)) / (ys.max() - ys.min()))


def fit(xs, ys, base: MediumParams, names: list[str], seed: int = 0) -> dict:
    lo = np.log([BOUNDS[n][0] for n in names])
    hi = np.log([BOUNDS[n][1] for n in names])
    ys_arr = np.asarray(ys)

    def resid(logp):
        return model(xs, base, dict(zip(names, np.exp(logp)))) - ys_arr

    rng = np.random.default_rng(seed)
    starts = [np.log([getattr(base, n) for n in names])]
    starts += [rng.uniform(lo, hi) for _ in range(N_STARTS - 1)]
    attempts = []
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo, hi), method="trf", bounds=(lo, hi),
                          max_nfev=60)
        params = dict(zip(names, map(float, np.exp(r.x))))
        attempts.append(dict(params=params, cost=float(r.cost),
                             nrmse=nrmse(r.fun + ys_arr, ys_arr)))
        print(f"    start -> {params}  NRMSE={attempts[-1]['nrmse']:.3f}", flush=True)
    best = min(attempts, key=lambda a: a["cost"])
    at_bound = [n for n in names
                if np.isclose(best["params"][n], BOUNDS[n][0], rtol=1e-2)
                or np.isclose(best["params"][n], BOUNDS[n][1], rtol=1e-2)]
    return dict(best, at_bound=at_bound,
                nrmse_all_starts=[a["nrmse"] for a in attempts])


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    data = {k: load_curve_csv(os.path.join(LITERATURE_DIR, f)) for k, f in SERIES.items()}
    base = base_params_for_file(SERIES["sim"])
    out = dict(K=K, base_params=base.__dict__, n_starts=N_STARTS, fits={})
    for variant, names in VARIANTS.items():
        for train, test in (("sim", "exp"), ("exp", "sim")):
            print(f"[holdout] {variant}: fit on {train}", flush=True)
            f = fit(data[train]["x"], data[train]["y"], base, names)
            pred = model(data[test]["x"], base, f["params"])
            f.update(train=train, test=test,
                     heldout_nrmse=nrmse(pred, data[test]["y"]),
                     heldout_pred=pred.tolist(),
                     train_pred=model(data[train]["x"], base, f["params"]).tolist())
            print(f"  in-sample NRMSE {f['nrmse']:.3f}, held-out on {test} "
                  f"{f['heldout_nrmse']:.3f}", flush=True)
            out["fits"][f"{variant}/{train}"] = f
    out["data"] = {k: dict(x=v["x"], y=v["y"]) for k, v in data.items()}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
