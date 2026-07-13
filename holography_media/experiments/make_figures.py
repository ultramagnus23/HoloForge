"""Generate paper figures from results_prelim.json (or GPU-scale results).

Fig A: compensation cliff -- PSNR vs target spatial frequency K, four methods,
       vertical lines at analytically predicted K_c for several budgets.
Fig B: recovery vs non-locality sigma.
Fig C: recovery vs index budget dn_max.
Fig D: qualitative panel -- target vs reconstructions at defaults.
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [("gs", "media-blind GS", "#c44"), ("blind", "media-blind SGD", "#e90"),
           ("ours", "media-in-the-loop (ours)", "#16a"), ("oracle", "oracle (ideal medium)", "#888")]


def agg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def main(path="results_prelim.json"):
    R = json.load(open(path))
    os.makedirs("figures", exist_ok=True)

    # ---- Fig A: cliff
    fig, ax = plt.subplots(figsize=(6, 4))
    Ks = []; series = {m: ([], []) for m, _, _ in METHODS}
    for k, cell in sorted(R["cliff"].items(), key=lambda kv: kv[1]["K"]):
        Ks.append(cell["K"])
        for m, _, _ in METHODS:
            mu, sd = agg(cell["rows"], m)
            series[m][0].append(mu); series[m][1].append(sd)
    for m, label, c in METHODS:
        mu, sd = np.array(series[m][0]), np.array(series[m][1])
        ax.plot(Ks, mu, "-o", color=c, label=label, ms=4)
        ax.fill_between(Ks, mu - sd, mu + sd, color=c, alpha=0.15)
    for b, Kc in R["predicted_cliff_K"].items():
        if math.isfinite(Kc) and Kc < max(Ks) * 1.2:
            ax.axvline(Kc, ls="--", lw=1, color="k", alpha=0.5)
            ax.text(Kc, ax.get_ylim()[0], f" {b}", rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel("target spatial frequency K (rad/µm)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Compensation cliff: quality vs target frequency\n(dashed: analytic K$_c$ predictions)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("figures/figA_cliff.png", dpi=200)

    # ---- Fig B: sigma
    fig, ax = plt.subplots(figsize=(6, 4))
    sigmas = sorted(float(s) for s in R["sigma"])
    for m, label, c in METHODS:
        mu = [agg(R["sigma"][str(s)]["bars"], m)[0] for s in sigmas]
        sd = [agg(R["sigma"][str(s)]["bars"], m)[1] for s in sigmas]
        ax.errorbar(sigmas, mu, yerr=sd, fmt="-o", color=c, label=label, ms=4, capsize=2)
    ax.set_xlabel("non-locality length σ (µm)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Recovery vs non-locality (bars, Λ=1.6 µm)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("figures/figB_sigma.png", dpi=200)

    # ---- Fig C: dn_max
    fig, ax = plt.subplots(figsize=(6, 4))
    dns = sorted(float(s) for s in R["dn_max"])
    for m, label, c in METHODS:
        mu = [agg(R["dn_max"][str(d)]["bars"], m)[0] for d in dns]
        ax.plot(dns, mu, "-o", color=c, label=label, ms=4)
    ax.set_xlabel("index budget Δn$_{max}$"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Recovery vs dynamic range")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("figures/figC_dnmax.png", dpi=200)

    print("wrote figures/figA_cliff.png figB_sigma.png figC_dnmax.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_prelim.json")
