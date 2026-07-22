"""
Phase 5 shared figure style: colorblind-safe palette (Okabe-Ito), Optica
sizing (single column ~8.6cm, double ~17.8cm), fonts >=7pt at print size,
vector PDF output.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
SINGLE_COL_IN = 8.6 * CM
DOUBLE_COL_IN = 17.8 * CM

# Okabe & Ito (2008) colorblind-safe palette.
COLORS = dict(
    black="#000000", orange="#E69F00", sky_blue="#56B4E9",
    bluish_green="#009E73", yellow="#F0E442", blue="#0072B2",
    vermillion="#D55E00", reddish_purple="#CC79A7",
)
METHOD_COLORS = {
    "M1": COLORS["reddish_purple"], "M2": COLORS["vermillion"],
    "M3": COLORS["yellow"], "M4": COLORS["blue"],
    "M5a": COLORS["bluish_green"], "M5b": COLORS["sky_blue"],
}
METHOD_LABELS = {
    "M1": "media-blind GS", "M2": "media-blind SGD", "M3": "linear pre-comp",
    "M4": "ours (media-in-the-loop)", "M5a": "oracle (constrained)",
    "M5b": "oracle (unconstrained)",
}

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed fonts as text, not curves
})


def new_fig(width="single", height_in=None, ncols=1, nrows=1, **kwargs):
    w = SINGLE_COL_IN if width == "single" else DOUBLE_COL_IN
    h = height_in if height_in is not None else w * 0.75
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)


def savefig(fig, path, dpi=600):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")


def no_data_placeholder(path, title, reason, width="single"):
    """A real PDF file (not silently skipped) stating exactly what's
    missing and why -- so 'figure absent' is never ambiguous with 'figure
    forgotten,' and no fabricated data ever substitutes for it."""
    fig, ax = new_fig(width=width, height_in=SINGLE_COL_IN * 0.5)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=8, weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.35, f"NOT YET AVAILABLE:\n{reason}", ha="center", va="center",
            fontsize=6.5, color=COLORS["vermillion"], transform=ax.transAxes, wrap=True)
    savefig(fig, path)
