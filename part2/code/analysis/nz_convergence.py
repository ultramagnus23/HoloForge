"""
n_z (slice-count) convergence of the headline paired gain -- reads the NZ
manifest's results (experiments/manifest.py::build_NZ_jobs) and reports,
per (K, n_z), the paired gain MIL - BSGD in dB with a two-sided 95%
t-interval over seeds (same n=3 convention as the paper's other CIs).

Decision rule this exists to apply (peer-review item B4):
  * gain(n_z=256) still > 0 with a CI that excludes zero at every K, AND
    |gain(256) - gain(128)| small vs. the CI half-width  -> the central
    claim survives at a converged slice count; a full M1/M2 rerun at
    n_z>=128 is then worth its cost.
  * gain(256) CI includes zero, or gain keeps drifting between 128 and
    256 -> the claim as written is not supported; do not spend the M1/M2
    rerun until the physics/claim is reconsidered.

Usage:  python -m analysis.nz_convergence [--results-dir DIR]
Writes results/summary/nz_convergence.json and prints a table.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")


def load(results_dir: str) -> list[dict]:
    out = []
    for p in glob.glob(os.path.join(results_dir, "NZ*", "*", "*_seed*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def t_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi); degenerate (nan CI) for n<2."""
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * se)
    return m, m - h, m + h


def paired_gains(rows: list[dict]) -> dict:
    """{(n_x, K, n_z): [gain_seed0, gain_seed1, ...]} using only seeds where
    BOTH arms finished."""
    by = defaultdict(dict)
    for r in rows:
        c = r["config"]
        by[(c["n_x"], round(c["K_nominal"], 4), c["n_z"])][(r["method_id"], r["seed"])] = r["psnr"]
    gains = {}
    for key, d in by.items():
        seeds = sorted({s for (_, s) in d})
        g = [d[("MIL", s)] - d[("BSGD", s)] for s in seeds
             if ("MIL", s) in d and ("BSGD", s) in d]
        if g:
            gains[key] = g
    return gains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()

    rows = load(args.results_dir)
    if not rows:
        raise SystemExit("no NZ results found yet")
    gains = paired_gains(rows)
    summary = {}
    for n_x in sorted({k[0] for k in gains}):
        g_nx = {(K, nz): v for (nx, K, nz), v in gains.items() if nx == n_x}
        Ks = sorted({k for k, _ in g_nx})
        nzs = sorted({n for _, n in g_nx})
        s_nx = summary[str(n_x)] = {"per_K": {}, "pooled_over_K": {}}
        print(f"\n=== n_x = {n_x} ===")
        print(f"{'K':>7} {'n_z':>5} {'n':>2} {'gain dB':>9} {'95% CI':>22}")
        for K in Ks:
            s_nx["per_K"][str(K)] = {}
            for nz in nzs:
                g = g_nx.get((K, nz))
                if not g:
                    continue
                m, lo, hi = t_ci(np.array(g))
                s_nx["per_K"][str(K)][str(nz)] = dict(n=len(g), gain_db=m, ci_lo=lo, ci_hi=hi, per_seed=g)
                print(f"{K:7.3f} {nz:5d} {len(g):2d} {m:9.4f} [{lo:9.4f},{hi:9.4f}]")
        print("pooled over K (per-seed mean across K points, seeds with all K):")
        for nz in nzs:
            per_seed = defaultdict(list)
            for K in Ks:
                for s_i, gv in enumerate(g_nx.get((K, nz), [])):
                    per_seed[s_i].append(gv)
            full = [np.mean(v) for v in per_seed.values() if len(v) == len(Ks)]
            if full:
                m, lo, hi = t_ci(np.array(full))
                s_nx["pooled_over_K"][str(nz)] = dict(n=len(full), gain_db=m, ci_lo=lo, ci_hi=hi)
                print(f"  n_z={nz:4d}  n={len(full)}  gain={m:8.4f}  CI=[{lo:8.4f},{hi:8.4f}]")

    out = os.path.join(args.results_dir, "summary", "nz_convergence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
