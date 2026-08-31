"""
Aggregates experiments/run_2d.py's results/M1_2D/*.json (the bounded 2D
study -- 3 targets x 3 budgets x 3 seeds x {BSGD, MIL}) into a handful of
\\newcommand macros, appended to paper/numbers.tex in the same style as
scripts/make_numbers_tex.py's 1D macros (every quantitative claim in the
manuscript traces to a macro, never a hand-typed number).

Kept as a separate, standalone script rather than folded into
analysis/aggregate.py: the 2D study's job schema (target_kind x
contrast_cap, no K_nominal/method-registry structure) doesn't fit the 1D
M1/M2/S1/S2 schema aggregate.py already parses, and forcing it in would
risk that existing, tested pipeline more than it's worth for one bounded
supplementary study.

Usage: python analysis/aggregate_2d.py
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
RESULTS_GLOB = os.path.join(ROOT, "results", "M1_2D", "**", "*.json")
NUMBERS_TEX = os.path.join(ROOT, "..", "paper", "numbers.tex")

MARKER_START = "% --- bounded 2D study (experiments/run_2d.py) macros ---"


def main():
    rows = [json.load(open(f)) for f in glob.glob(RESULTS_GLOB, recursive=True)]
    if not rows:
        raise SystemExit("no results/M1_2D/*.json found -- run experiments/run_2d.py first")

    by_key = {}
    for r in rows:
        c = r["config"]
        key = (c["target_kind"], c["contrast_cap"])
        by_key.setdefault(key, {}).setdefault(r["method_id"], {})[r["seed"]] = r["psnr_si"]

    all_gains = []
    for (t, b), methods in by_key.items():
        bsgd, mil = methods["BSGD"], methods["MIL"]
        for seed in bsgd:
            all_gains.append(mil[seed] - bsgd[seed])

    n_targets = len({k[0] for k in by_key})
    n_budgets = len({k[1] for k in by_key})
    n_seeds = len({r["seed"] for r in rows if r["method_id"] == "MIL"})

    macros = {
        "TwoDNTargets": str(n_targets),
        "TwoDNBudgets": str(n_budgets),
        "TwoDNSeeds": str(n_seeds),
        "TwoDGridN": str(rows[0]["config"]["n"]),
        "TwoDNIters": str(rows[0]["config"]["n_iters"]),
        "TwoDMinGain": f"{min(all_gains):.2f}",
        "TwoDMeanGain": f"{st.mean(all_gains):.2f}",
        "TwoDMaxGain": f"{max(all_gains):.2f}",
        # matches experiments/make_2d_reconstructions.py's fixed seed=0
        "TwoDReconSeed": "0",
    }

    lines = [MARKER_START]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    block = "\n".join(lines) + "\n"

    existing = open(NUMBERS_TEX).read() if os.path.exists(NUMBERS_TEX) else ""
    if MARKER_START in existing:
        pre, _, post = existing.partition(MARKER_START)
        rest_after_block = post.split("\n", len(macros))[-1]
        existing = pre + block + "\n".join(post.split("\n")[len(macros) + 1:])
    else:
        existing = existing.rstrip("\n") + "\n\n" + block

    with open(NUMBERS_TEX, "w") as f:
        f.write(existing)

    print(f"wrote {len(macros)} macros to {NUMBERS_TEX}")
    for k, v in macros.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
