"""
Diffraction-efficiency confirmation check (Phase 3 Tier-1 item, near-zero
marginal cost): the M1 result JSONs already carry a `diffraction_efficiency`
field per job (computed alongside psnr, never previously aggregated into
paper_numbers.json/numbers.tex). This script checks whether the paired-gain
story PSNR tells (positive at nearly every K/budget, magnitude rising with
budget) is corroborated by an independent metric, or diverges from it.

Uses analysis/aggregate.py's existing paired_gain(..., key=...) directly --
no new aggregation machinery, just a different key on data already loaded.

Usage: python analysis/de_confirmation.py
"""
import os
import statistics as st
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from analysis.aggregate import load_all_results, group_by_config, paired_gain

NUMBERS_TEX = os.path.join(HERE, "..", "..", "paper", "numbers.tex")
MARKER = "% --- diffraction-efficiency confirmation (Sec. 5.1) macros ---"


def main():
    grouped = group_by_config(load_all_results())
    per_budget = {}
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != "M1":
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        budget = cfg.get("contrast_cap")
        rows, bsgd = by_method.get("MIL", []), by_method.get("BSGD", [])
        if not rows or not bsgd:
            continue
        gains = [g for _, g in paired_gain(rows, bsgd, key="diffraction_efficiency")]
        per_budget.setdefault(budget, []).extend(gains)

    all_gains = [g for gs in per_budget.values() for g in gs]
    n_negative = sum(1 for g in all_gains if g < 0)

    macros = {
        "DEMeanGainTwoX": f"{st.mean(per_budget[2.0]) * 100:.2f}",
        "DEMeanGainFourX": f"{st.mean(per_budget[4.0]) * 100:.2f}",
        "DEMeanGainEightX": f"{st.mean(per_budget[8.0]) * 100:.2f}",
        "DEMinGainAllBudgets": f"{min(all_gains) * 100:.3f}",
        "DENNegative": str(n_negative),
        "DENPairs": str(len(all_gains)),
    }

    lines = [MARKER] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    block = "\n".join(lines) + "\n"
    existing = open(NUMBERS_TEX).read() if os.path.exists(NUMBERS_TEX) else ""
    if MARKER in existing:
        pre, _, _post = existing.partition(MARKER)
        rest = existing.split("\n")
        # find and drop old block lines (marker + len(macros) lines), keep the rest
        idx = rest.index(MARKER)
        rest = rest[:idx] + rest[idx + 1 + len(macros):]
        existing = "\n".join(rest).rstrip("\n") + "\n\n" + block
    else:
        existing = existing.rstrip("\n") + "\n\n" + block
    with open(NUMBERS_TEX, "w") as f:
        f.write(existing)

    print(f"wrote {len(macros)} macros to {NUMBERS_TEX}")
    for k, v in macros.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
