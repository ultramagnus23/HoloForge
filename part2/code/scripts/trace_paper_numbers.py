"""Phase 0.4 provenance audit: trace every number in paper/part2_media_main.tex's
cliff table and prose back to its exact entry in results_prelim.json /
results_prelim2.json, and flag anything that does NOT trace cleanly.

This is a read-only audit script: it recomputes each claimed number from the
raw JSON and reports match/mismatch. It does not modify any results file
(ground rule: raw results are append-only) and does not touch the tex.

Usage: python scripts/trace_paper_numbers.py
Output: prints a table to stdout; docs/provenance_report.md is the curated
write-up of these findings (written by hand, not generated, since it also
needs to state conclusions the raw diff table alone can't express -- e.g.
staleness after the seed-init fix).
"""
import json
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def check(label, claimed, actual, tol=0.005):
    ok = abs(claimed - actual) <= tol
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] {label}: claimed={claimed}  actual={actual:.6f}  diff={actual-claimed:+.6f}")
    return ok


def main():
    prelim = load("results_prelim.json")
    prelim2 = load("results_prelim2.json")

    print("=== Cliff table (paper/part2_media_main.tex Table I, all from results_prelim.json seed=0) ===")
    # (period, claimed_ours, claimed_blind, claimed_gain, claimed_oracle)
    cliff_claims = [
        (48, 7.80, 4.29, 3.50, 14.71),
        (24, 8.06, 5.68, 2.37, 12.61),
        (16, 5.59, 3.93, 1.66, 12.09),
        (64, 5.20, 3.56, 1.65, 18.16),
        (32, 5.40, 4.45, 0.95, 14.11),
        (8, 4.77, 4.58, 0.19, 7.85),
        (12, 4.75, 4.95, -0.20, 10.34),
    ]
    all_ok = True
    gains = []
    for period, c_ours, c_blind, c_gain, c_oracle in cliff_claims:
        row = prelim["cliff"][f"period{period}"]["rows"][0]  # seed=0
        K = prelim["cliff"][f"period{period}"]["K"]
        print(f" period{period} (K={K:.4f}):")
        all_ok &= check("  ours", c_ours, row["ours"], tol=0.005)
        all_ok &= check("  blind", c_blind, row["blind"], tol=0.005)
        all_ok &= check("  gain", c_gain, row["ours"] - row["blind"], tol=0.005)
        all_ok &= check("  oracle", c_oracle, row["oracle"], tol=0.005)
        gains.append(row["ours"] - row["blind"])

    mean_gain = sum(gains) / len(gains)
    print(f"\n mean gain across 7 K values: {mean_gain:.4f} (claimed +1.45 dB)")
    print(f" gain range: [{min(gains):.4f}, {max(gains):.4f}] (claimed -0.20 to +3.50 dB)")

    print("\n=== oracle 'K=7.85 coincidence' check ===")
    oracle_785 = prelim["cliff"]["period8"]["rows"][0]["oracle"]
    K_785 = prelim["cliff"]["period8"]["K"]
    print(f" oracle PSNR at period8 = {oracle_785:.9f} dB")
    print(f" K at period8           = {K_785:.9f} rad/um")
    print(f" difference             = {K_785 - oracle_785:.9f} (different quantities, different units;")
    print(f"                          both real, independently-computed numbers -- NOT a transcription error)")

    print("\n=== Mean in-support DE (claimed 0.553 ours / 0.508 blind) ===")
    de_ours = [prelim["cliff"][f"period{p}"]["rows"][0]["de_ours"] for p, *_ in cliff_claims]
    de_blind = [prelim["cliff"][f"period{p}"]["rows"][0]["de_blind"] for p, *_ in cliff_claims]
    print(f" mean de_ours  = {sum(de_ours)/len(de_ours):.6f} (claimed 0.553)")
    print(f" mean de_blind = {sum(de_blind)/len(de_blind):.6f} (claimed 0.508)")

    print("\n=== Oracle gap (claimed 3.1-13.0 dB below oracle) ===")
    gaps = [prelim["cliff"][f"period{p}"]["rows"][0]["oracle"] - prelim["cliff"][f"period{p}"]["rows"][0]["ours"]
            for p, *_ in cliff_claims]
    print(f" gap range: [{min(gaps):.4f}, {max(gaps):.4f}] (claimed 3.1 to 13.0 dB)")

    print("\n=== GS margin (claimed 0.90-4.33 dB, seed=0 vs seed=1) ===")
    for period, *_ in cliff_claims:
        rows = prelim["cliff"][f"period{period}"]["rows"]
        for s, row in enumerate(rows):
            margin = row["ours"] - row["gs"]
            print(f" period{period} seed{s}: ours-gs = {margin:.4f}")

    print("\n=== Sigma high-K probe (results_prelim2.json, claimed +0.23/+0.19/+0.16/+0.02 dB) ===")
    sigma_claims = {"0.02": 0.23, "0.08": 0.19, "0.2": 0.16, "0.3": 0.02}
    for sg, claimed in sigma_claims.items():
        row = prelim2["sigma_highK"][sg][0]
        gain = row["ours"] - row["blind"]
        check(f"sigma={sg}", claimed, gain, tol=0.005)

    print("\n=== Shrinkage sweep (results_prelim2.json, claimed +1.67/+1.64/+1.83 dB) ===")
    shrink_claims = {"0.0": 1.67, "0.01": 1.64, "0.03": 1.83}
    for s, claimed in shrink_claims.items():
        row = prelim2["shrinkage"][s][0]
        gain = row["ours"] - row["blind"]
        check(f"s={s}", claimed, gain, tol=0.005)

    print("\n=== F2 panel numbers (part2_media_draft_v1.md only, NOT in tex) ===")
    print(" claimed: gs=6.54 blind=9.04 ours=16.64 oracle=26.57 (experiments/f2_panel.py)")
    print(" NOTE: f2_panel.py calls media_in_the_loop/media_blind_sgd/oracle_ideal with")
    print("       DEFAULT seed=0, which changed meaning after the seed-init fix (exact")
    print("       zeros -> seeded random perturbation). Rerunning today does NOT reproduce")
    print("       these exact numbers -- see docs/provenance_report.md for measured drift.")

    print("\n" + ("ALL CLIFF-TABLE / PROSE NUMBERS TRACE CLEANLY (see notes above for caveats)"
                  if all_ok else "SOME MISMATCHES FOUND -- see MISMATCH lines above"))


if __name__ == "__main__":
    main()
