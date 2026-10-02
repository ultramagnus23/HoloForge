"""
Shared figure style: colorblind-safe palette (Okabe-Ito), Optica sizing
(single column 8.6 cm, full width 17.8 cm), fonts >= 6.5 pt at print size,
vector PDF output with embedded fonts. Every series differs in line style
or marker as well as hue, because Optica asks that colour not be the only
way to tell figure elements apart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
SINGLE_COL_IN = 8.6 * CM
DOUBLE_COL_IN = 17.8 * CM
# Text block of the Optica universal manuscript template (8.5 in page,
# 1.625 in side margins): figures drawn at this width print at true font size
# in the submitted PDF.
TEXT_IN = 5.25

# Okabe & Ito (2008) colorblind-safe palette (yellow omitted: unreadable on white).
COLORS = dict(
    black="#000000", orange="#E69F00", sky_blue="#56B4E9",
    bluish_green="#009E73", blue="#0072B2",
    vermillion="#D55E00", reddish_purple="#CC79A7",
)
METHOD_COLORS = {
    "GS": COLORS["reddish_purple"], "BSGD": COLORS["vermillion"],
    "LPC": COLORS["sky_blue"], "GPC": COLORS["bluish_green"],
    "RSGD": COLORS["black"], "SAT": COLORS["orange"], "MIL": COLORS["blue"],
}
METHOD_LABELS = {
    "GS": "GS", "BSGD": "BSGD", "LPC": "LPC", "GPC": "GPC",
    "RSGD": "RSGD", "SAT": "SAT", "MIL": "MIL",
}
METHOD_LINESTYLES = {
    "GS": (0, (1, 1)), "BSGD": "--", "LPC": ":", "GPC": (0, (4, 1, 1, 1)),
    "RSGD": (0, (3, 1)), "SAT": "-.", "MIL": "-",
}
METHOD_MARKERS = {
    "GS": "o", "BSGD": "s", "LPC": "^", "GPC": "p",
    "RSGD": "*", "SAT": "X", "MIL": "D",
}
# Contrast budgets B_c = 2, 4, 8.
BUDGET_COLORS = {2.0: COLORS["blue"], 4.0: COLORS["vermillion"], 8.0: COLORS["bluish_green"]}
BUDGET_LINESTYLES = {2.0: "-", 4.0: "--", 8.0: ":"}
BUDGET_MARKERS = {2.0: "o", 4.0: "s", 8.0: "^"}

K_LABEL = r"spatial frequency $K$ (rad/$\mu$m)"
GAIN_LABEL = r"paired gain, MIL $-$ BSGD (dB)"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed fonts as text, not curves
})


def new_fig(width="single", height_in=None, ncols=1, nrows=1, **kwargs):
    w = {"single": SINGLE_COL_IN, "double": DOUBLE_COL_IN, "text": TEXT_IN}[width]
    h = height_in if height_in is not None else w * 0.75
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)


def panel_label(ax, letter):
    """Bold '(a)'-style label in the upper-left corner outside the axes."""
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, fontweight="bold")


def savefig(fig, path, dpi=600):
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")


def no_data_placeholder(path, title, reason, width="single"):
    """A real PDF stating exactly what is missing, so that 'figure absent'
    is never ambiguous with 'figure forgotten' and no fabricated data ever
    stands in for it."""
    fig, ax = new_fig(width=width, height_in=SINGLE_COL_IN * 0.5)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=8, weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.35, f"NOT AVAILABLE:\n{reason}", ha="center", va="center",
            fontsize=6.5, color=COLORS["vermillion"], transform=ax.transAxes, wrap=True)
    savefig(fig, path)
