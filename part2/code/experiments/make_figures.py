"""Generate paper figures from results_confirm.json (confirmation-scale;
preferred if present) or results_prelim.json (CPU-scale v0.1 fallback), plus
the ablation/RCWA/3D-showcase side results.

Fig A: compensation cliff -- PSNR vs target spatial frequency K, four methods,
       vertical lines at analytically predicted K_c for several budgets.
Fig B: recovery vs non-locality sigma.
Fig C: recovery vs index budget dn_max.
Fig D: qualitative panel -- target vs reconstructions at defaults (see f2_panel.py).
Fig E: gradient-pathway ablation -- wall-clock and fidelity, unrolled vs
       adjoint(checkpointed) vs neural-surrogate.
Fig F: RCWA cross-check -- scalar Kogelnik prediction vs full-vector RCWA.

Usage: python make_figures.py [results.json]
       defaults to results_confirm.json if it exists, else results_prelim.json.
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def default_results_path():
    confirm = os.path.join(ROOT, "results_confirm.json")
    if os.path.exists(confirm):
        return confirm
    return os.path.join(ROOT, "results_prelim.json")

METHODS = [("gs", "media-blind GS", "#c44"), ("blind", "media-blind SGD", "#e90"),
           ("ours", "media-in-the-loop (ours)", "#16a"), ("oracle", "oracle (ideal medium)", "#888")]


def agg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def main(path=None):
    path = path or default_results_path()
    print(f"reading {path}")
    R = json.load(open(path))
    scale_note = R.get("meta", {}).get("scale", "CPU prelim scale (v0.1) -- not paper scale")
    print(f"scale: {scale_note}")
    os.makedirs("figures", exist_ok=True)

    # ---- Fig A: cliff
    if "cliff" in R and "predicted_cliff_K" in R:
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
        print("wrote figures/figA_cliff.png")
    else:
        print(f"[skip Fig A] 'cliff'/'predicted_cliff_K' not in {path} -- run the A sweep first")

    # ---- Fig B: sigma
    if "sigma" in R:
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
        print("wrote figures/figB_sigma.png")
    else:
        print(f"[skip Fig B] 'sigma' not in {path} -- run the B sweep first")

    # ---- Fig C: dn_max
    if "dn_max" in R:
        fig, ax = plt.subplots(figsize=(6, 4))
        dns = sorted(float(s) for s in R["dn_max"])
        for m, label, c in METHODS:
            mu = [agg(R["dn_max"][str(d)]["bars"], m)[0] for d in dns]
            ax.plot(dns, mu, "-o", color=c, label=label, ms=4)
        ax.set_xlabel("index budget Δn$_{max}$"); ax.set_ylabel("PSNR (dB)")
        ax.set_title("Recovery vs dynamic range")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig("figures/figC_dnmax.png", dpi=200)
        print("wrote figures/figC_dnmax.png")
    else:
        print(f"[skip Fig C] 'dn_max' not in {path} -- run the C sweep first")

    # ---- Fig E: gradient-pathway ablation
    abl_path = os.path.join(ROOT, "results_ablation_gradients.json")
    if os.path.exists(abl_path):
        A = json.load(open(abl_path))
        methods = ["unrolled", "checkpoint", "surrogate"]
        labels = ["unrolled\nautodiff", "adjoint\n(checkpointed)", "neural\nsurrogate"]
        walls = [A["optimization"][m]["wall"] for m in methods]
        psnrs = [A["optimization"][m]["psnr"] for m in methods]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        ax1.bar(labels, walls, color=["#16a", "#e90", "#2a6"])
        ax1.set_ylabel("optimization wall-clock (s)")
        ax1.set_title("Cost")
        ax2.bar(labels, psnrs, color=["#16a", "#e90", "#2a6"])
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title("Quality (same iter budget)")
        fig.suptitle("Fig E: gradient-pathway ablation")
        fig.tight_layout()
        fig.savefig("figures/figE_gradient_ablation.png", dpi=200)
        print("wrote figures/figE_gradient_ablation.png")
    else:
        print(f"[skip Fig E] {abl_path} not found -- run experiments/ablation_gradients.py")

    # ---- Fig F: RCWA cross-check
    rcwa_path = os.path.join(ROOT, "results_rcwa.json")
    if os.path.exists(rcwa_path):
        Rc = json.load(open(rcwa_path))
        cases = [c for c in Rc["cases"] if c.get("rcwa_t1") is not None]
        if cases:
            Ks = [c["K"] for c in cases]
            kog = [c["kogelnik"] for c in cases]
            rc1 = [c["rcwa_t1"] for c in cases]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(Ks, kog, "-o", label="Kogelnik (scalar)", color="#16a")
            ax.plot(Ks, rc1, "-s", label="RCWA (full-vector)", color="#c44")
            ax.set_xlabel("K (rad/µm)"); ax.set_ylabel("diffraction efficiency")
            ax.set_title(f"Fig F: RCWA cross-check\n(max |Δ| = {Rc['max_abs_deviation']:.3f})")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig("figures/figF_rcwa.png", dpi=200)
            print("wrote figures/figF_rcwa.png")
    else:
        print(f"[skip Fig F] {rcwa_path} not found -- run experiments/rcwa_crosscheck.py")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
