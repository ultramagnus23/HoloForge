"""
Turns Sec. 5.4's "wasted media" argument from rhetoric into a measurement
(Phase 3 Tier-1 item 2). Threshold choice, stated before the result (not
fit to it): an exposure "fails" if its PSNR falls more than 2 dB below
the constrained oracle's PSNR ceiling at the same (K, budget) -- a round
number tied to the oracle framework Sec. 4.1 already defines for exactly
this purpose (isolating what's achievable from what a given method
reaches), not chosen post-hoc to flatter either arm. 1 dB and 3 dB are
also reported alongside 2 dB to show the finding isn't fragile to the
exact round number picked.

WP3 (Applied Optics revision) extends this from BSGD/MIL-only to every
non-oracle method (GS, LPC, SAT, RSGD, GPC too), so Sec. 5.4's table can
show the full baseline set, not just the two methods it originally
compared. BSGD/MIL keep their original macro names (WastedMedia{BSGD,
MIL}Frac...) for compatibility with the existing prose in
Sec.~\ref{sec:failure-rate}; the newly added methods get parallel
macros (WastedMedia{GS,LPC,SAT,RSGD,GPC}Frac...). GS/LPC/GPC are
closed-form/seed-0-only (Sec. 4.1) so their per-config "gap" is a single
value rather than a seed mean, same distinction gain_vs_bsgd_seed_mean
already draws elsewhere in this codebase.

Usage: python analysis/wasted_media.py
"""
import os
import statistics as st
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from analysis.aggregate import load_all_results, group_by_config

NUMBERS_TEX = os.path.join(HERE, "..", "..", "paper", "numbers.tex")
MARKER = "% --- wasted-media measurement (Sec. 5.4) macros ---"
THRESH_DB = 2.0
EXTRA_METHODS = ["GS", "LPC", "SAT", "RSGD", "GPC"]


def main():
    grouped = group_by_config(load_all_results())
    gaps_bsgd, gaps_mil = [], []
    gaps_extra = {m: [] for m in EXTRA_METHODS}
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != "M1":
            continue
        orc, bsgd, mil = by_method.get("ORC", []), by_method.get("BSGD", []), by_method.get("MIL", [])
        if not orc or not bsgd or not mil:
            continue
        orc_mean = st.mean(r["psnr"] for r in orc)
        gaps_bsgd.append(orc_mean - st.mean(r["psnr"] for r in bsgd))
        gaps_mil.append(orc_mean - st.mean(r["psnr"] for r in mil))
        for m in EXTRA_METHODS:
            rows = by_method.get(m, [])
            if rows:
                gaps_extra[m].append(orc_mean - st.mean(r["psnr"] for r in rows))

    def fail_frac(gaps, thresh):
        return sum(1 for g in gaps if g > thresh) / len(gaps)

    macros = {
        "WastedMediaThreshDB": f"{THRESH_DB:.0f}",
        "WastedMediaBSGDFrac": f"{fail_frac(gaps_bsgd, THRESH_DB) * 100:.0f}",
        "WastedMediaMILFrac": f"{fail_frac(gaps_mil, THRESH_DB) * 100:.0f}",
        # No digits in LaTeX control-word names -- \Foo1dB parses as \Foo
        # followed by literal "1dB", not one macro. Spelled out instead.
        "WastedMediaBSGDFracOneDB": f"{fail_frac(gaps_bsgd, 1.0) * 100:.0f}",
        "WastedMediaMILFracOneDB": f"{fail_frac(gaps_mil, 1.0) * 100:.0f}",
        "WastedMediaBSGDFracThreeDB": f"{fail_frac(gaps_bsgd, 3.0) * 100:.0f}",
        "WastedMediaMILFracThreeDB": f"{fail_frac(gaps_mil, 3.0) * 100:.0f}",
        "WastedMediaNConfigs": str(len(gaps_bsgd)),
    }
    for m in EXTRA_METHODS:
        gaps = gaps_extra[m]
        n = f"WastedMedia{m}"
        if gaps:
            macros[f"{n}Frac"] = f"{fail_frac(gaps, THRESH_DB) * 100:.0f}"
            macros[f"{n}FracOneDB"] = f"{fail_frac(gaps, 1.0) * 100:.0f}"
            macros[f"{n}FracThreeDB"] = f"{fail_frac(gaps, 3.0) * 100:.0f}"
        else:
            macros[f"{n}Frac"] = macros[f"{n}FracOneDB"] = macros[f"{n}FracThreeDB"] = "N/A"

    lines = [MARKER] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    block = "\n".join(lines) + "\n"
    existing = open(NUMBERS_TEX).read() if os.path.exists(NUMBERS_TEX) else ""
    if MARKER in existing:
        rest = existing.split("\n")
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
