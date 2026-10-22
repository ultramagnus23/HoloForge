"""
Which twin parameter does the media-in-the-loop gain depend on most?

Three views of the same question, all from result files already in
results/, so this script adds no new simulation:

  tornado          S3. Design the exposure once on the NOMINAL twin, record
                   it on a twin with ONE parameter scaled. One-at-a-time,
                   exact, and the only tier that can support a claim of the
                   form "the advantage survives getting this parameter
                   wrong", because the error is one-sided: BSGD's design
                   uses no recording parameter beyond the scale dn_max, so
                   the whole change in paired gain is MIL's exposure being
                   designed for the wrong medium.

  joint           S6. All four parameters wrong at once, drawn independently
  attribution     and uniformly within +/- S6_PCT_RANGE%. Regressing the
                  per-draw gain on the four standardized perturbations gives
                  a first-order linear attribution of the gain's variance.
                  This is NOT a Sobol decomposition: 30 draws cannot resolve
                  interaction or higher-order indices, and the reported R^2
                  states how much of the spread the linear part explains.

  reoptimized     S2. Perturb the medium and let BOTH arms re-optimize on
                  it. The perturbation is then common to both arms, so by
                  the paper's own pairing argument much of it cancels; this
                  answers "does the gain exist on a different medium",
                  not "what happens if the twin is wrong".

Writes results/applied_optics/sensitivity.json.

Usage: python analysis/sensitivity.py   (seconds, CPU, no simulation)
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np

from analysis.aggregate import (RESULTS_ROOT, group_by_config, load_all_results,
                                mean_std_median_ci95, paired_gain,
                                s2_sensitivity_summary, s3_mismatch_summary)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "applied_optics")
OUT_PATH = os.path.join(OUT_DIR, "sensitivity.json")

PARAM_ORDER = ["D0", "sigma", "kappa", "dn_max"]
TORNADO_PCTS = [-50, -25, 25, 50]
N_BOOTSTRAP = 10000


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def tornado_from_s3(grouped: dict) -> dict:
    """Gain at each one-at-a-time perturbation, and the width of the gain
    range each parameter spans within +/-50%. Ranked by that width, which is
    the operational answer to "which parameter matters most": how far the
    gain can move if you get this one wrong by a factor of two."""
    s3 = s3_mismatch_summary(grouped)
    if s3.get("status") != "ok":
        return dict(status="no_data")

    nominal = s3["nominal"]
    params = {}
    for param, rec in s3["by_param"].items():
        by_pct = {}
        for pct in TORNADO_PCTS:
            stat = rec["by_pct"].get(str(pct))
            if stat and stat.get("mean") is not None:
                by_pct[str(pct)] = stat
        if not by_pct:
            continue
        means = [v["mean"] for v in by_pct.values()]
        params[param] = dict(
            by_pct=by_pct,
            min_mean=min(means), max_mean=max(means),
            range_dB=max(means) - min(means),
            # signed drop from nominal: negative means the gain gets worse
            worst_drop_dB=min(means) - nominal["mean"],
            worst_pct=min(by_pct.items(), key=lambda kv: kv[1]["mean"])[0],
            sign_flip_pct=rec.get("sign_flip_pct"),
        )

    ranked = sorted(params, key=lambda p: -params[p]["range_dB"])
    extended = {p: sorted(int(k) for k in rec["by_pct"])
                for p, rec in s3["by_param"].items()}
    return dict(status="ok", nominal=nominal, params=params, ranked=ranked,
                tested_pcts=TORNADO_PCTS,
                all_tested_pcts_by_param=extended,
                note="S3: designed on the nominal twin, recorded on a twin "
                     "with one parameter scaled; K-averaged per seed, then "
                     "mean and 95% t-interval over seeds.")


def _s6_draws(grouped: dict) -> dict:
    """{draw_id: (gain_dB, {param: pct})} from the S6 joint draws."""
    out = {}
    for (exp_id, ch), by_method in grouped.items():
        if exp_id != "S6":
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        draw_id = cfg.get("draw_id")
        pcts = {}
        for token in (cfg.get("pct_by_param_str") or "").split(","):
            if "=" in token:
                name, val = token.split("=", 1)
                pcts[name] = float(val)
        for seed, g in paired_gain(by_method.get("MIL", []),
                                   by_method.get("BSGD", []), key="psnr"):
            rec = out.setdefault(draw_id, dict(pcts=pcts, by_seed={}))
            rec["by_seed"].setdefault(seed, []).append(g)

    draws = {}
    for draw_id, rec in out.items():
        seed_means = [sum(v) / len(v) for v in rec["by_seed"].values() if v]
        if seed_means and rec["pcts"]:
            draws[draw_id] = (sum(seed_means) / len(seed_means), rec["pcts"])
    return draws


def joint_attribution_from_s6(grouped: dict) -> dict:
    """Standardized first-order linear attribution of the per-draw gain to
    the four simultaneous parameter errors, with bootstrap intervals over
    draws. Reports R^2 so that the share of the spread the linear model
    does NOT explain is visible rather than implied."""
    draws = _s6_draws(grouped)
    if len(draws) < len(PARAM_ORDER) + 2:
        return dict(status="no_data", n_draws=len(draws))

    ids = sorted(draws)
    y = np.array([draws[i][0] for i in ids], dtype=float)
    params = [p for p in PARAM_ORDER if all(p in draws[i][1] for i in ids)]
    X_raw = np.array([[draws[i][1][p] for p in params] for i in ids], dtype=float)

    scale = X_raw.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    X = (X_raw - X_raw.mean(axis=0)) / scale
    y_scale = float(y.std(ddof=1))

    def fit(Xm, yv):
        A = np.column_stack([np.ones(len(yv)), Xm])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ beta
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        return beta, r2, resid

    beta, r2, resid = fit(X, y)

    rng = np.random.default_rng(0)
    boot = np.empty((N_BOOTSTRAP, len(params)))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        try:
            bb, _, _ = fit(X[idx], y[idx])
        except np.linalg.LinAlgError:
            boot[b] = np.nan
            continue
        boot[b] = bb[1:]
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    coeffs = {}
    for j, p in enumerate(params):
        coeffs[p] = dict(
            # dB of gain per one standard deviation of this parameter's error
            beta_dB_per_sd=float(beta[j + 1]),
            ci95_lo=float(lo[j]), ci95_hi=float(hi[j]),
            # share of the gain's own standard deviation this term moves
            beta_normalized=float(beta[j + 1] / y_scale) if y_scale > 0 else None,
            excludes_zero=bool(lo[j] > 0 or hi[j] < 0),
            sd_of_error_pct=float(scale[j]),
        )

    ranked = sorted(coeffs, key=lambda p: -abs(coeffs[p]["beta_dB_per_sd"]))
    return dict(status="ok", n_draws=len(y), params=params,
                intercept_dB=float(beta[0]),
                gain_mean_dB=float(y.mean()), gain_sd_dB=y_scale,
                coefficients=coeffs, ranked=ranked, r2=float(r2),
                residual_sd_dB=float(resid.std(ddof=len(params) + 1)),
                n_bootstrap=N_BOOTSTRAP,
                note="First-order linear attribution over the tested +/-25% "
                     "box, not a Sobol decomposition: with this many draws "
                     "interaction and higher-order indices are not "
                     "identifiable, and 1 - R^2 is the share of the spread "
                     "the linear terms do not explain.")


def main() -> int:
    grouped = group_by_config(load_all_results(RESULTS_ROOT))
    out = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        tornado=tornado_from_s3(grouped),
        joint_attribution=joint_attribution_from_s6(grouped),
        reoptimized=s2_sensitivity_summary(grouped),
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.normpath(OUT_PATH)}")

    t = out["tornado"]
    if t.get("status") == "ok":
        print(f"\n[tornado] nominal gain {t['nominal']['mean']:.3f} dB")
        print("  parameter   gain range over +/-50% (dB)   worst level   worst drop (dB)")
        for p in t["ranked"]:
            r = t["params"][p]
            print(f"  {p:<10}  {r['range_dB']:>10.3f}                  "
                  f"{r['worst_pct']:>5}%      {r['worst_drop_dB']:>7.3f}")
    j = out["joint_attribution"]
    if j.get("status") == "ok":
        print(f"\n[joint] {j['n_draws']} draws, gain {j['gain_mean_dB']:.3f} "
              f"+/- {j['gain_sd_dB']:.3f} dB (sd), linear R^2 = {j['r2']:.2f}")
        print("  parameter   dB per sd of error        95% CI            excludes 0")
        for p in j["ranked"]:
            c = j["coefficients"][p]
            print(f"  {p:<10}  {c['beta_dB_per_sd']:>+8.4f}            "
                  f"[{c['ci95_lo']:>+7.4f}, {c['ci95_hi']:>+7.4f}]      "
                  f"{'yes' if c['excludes_zero'] else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
