"""The README's results figure: the logP error of each arm, as a bar chart. Run it to rebuild results_logp.png:

    python docs/assets/results_logp.py

The numbers are the ones in the README's "Results at a glance" table, which come from docs/results.md (ZINC250k,
logP, the FreeGress protocol: 100 targets drawn from real molecules, 10 generations each, validation). Edit them
there first; this script only draws them.

Form: one arm is the point (Feynman-Kac, the largest single step), so it is drawn in the accent and the rest in
the de-emphasis gray, with the value at every bar tip and no legend or axis ticks (every bar is labelled; the
README table is the table view). Colours and ink are the dataviz reference palette's light surface.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

# (label, logP error, note, is_branch, is_headline). Rows 5 and 6 are alternatives, not additions to the row above.
ARMS = [
    ("the first adapter (FiLM only)", 0.5420, "", False, False),
    ("cross-attention + Fourier adapter", 0.3250, "", False, False),
    ("+ learned node counts", 0.3132, "", False, False),
    ("+ Feynman-Kac (β = 20)", 0.2299, "", False, True),
    ("instead of FK: + RL on the adapter", 0.2807, "", True, False),
    ("RL and FK together", 0.2157, "not significantly better than FK alone", True, False),
]
NOISE_FLOOR = 0.011

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
ACCENT, CONTEXT, BASELINE, HAIRLINE = "#2a78d6", "#c3c2b7", "#c3c2b7", "#e1e0d9"
FONT = "Roboto Condensed"

W, H = 760, 330                      # canvas in "1x" pixels; the file is written at 2x
LABEL_X, BAR_X0, SCALE = 214, 228, 740   # labels end at LABEL_X; the bars start at BAR_X0; SCALE px per unit of error
ROW0, PITCH, GAP, THICK, RADIUS = 92, 38, 14, 18, 4


def rounded_bar(ax, x0, y_mid, length, color):
    """A bar that grows from a square baseline end to a 4 px rounded data end."""
    y0, y1, x1, r = y_mid - THICK / 2, y_mid + THICK / 2, x0 + length, RADIUS
    verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", lw=0))


def main(out: Path) -> None:
    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    px = 72 / 100                    # one "1x" pixel in points
    ax.text(24, 30, "Error to the logP target, by arm (lower is better)", fontsize=15 * px, fontweight="medium",
            color=INK, va="baseline")
    ax.text(24, 52, f"ZINC250k, 100 targets × 10 generations, validation. Run-to-run noise is about {NOISE_FLOOR}.",
            fontsize=12 * px, color=INK2, va="baseline")

    y = ROW0
    for k, (label, value, note, branch, headline) in enumerate(ARMS):
        if k == 4:                                       # the two alternatives sit below a hairline
            y += GAP
            ax.plot([24, W - 24], [y - PITCH / 2 + 1, y - PITCH / 2 + 1], color=HAIRLINE, lw=0.8, solid_capstyle="butt")
        length = value * SCALE
        rounded_bar(ax, BAR_X0, y, length, ACCENT if headline else CONTEXT)
        ax.text(LABEL_X, y + 4.2, label, fontsize=13 * px, ha="right", va="baseline",
                color=INK if headline else INK2, fontweight="medium" if headline else "regular",
                fontstyle="italic" if branch else "normal")
        ax.text(BAR_X0 + length + 8, y + 4.6, f"{value:.4f}", fontsize=13 * px, ha="left", va="baseline", color=INK,
                fontweight="medium" if headline else "regular")
        if note:
            ax.text(BAR_X0 + length + 58, y + 4.2, note, fontsize=12 * px, ha="left", va="baseline", color=MUTED)
        y += PITCH
    ax.plot([BAR_X0, BAR_X0], [ROW0 - PITCH / 2 + 4, y - PITCH / 2 + GAP - 4], color=BASELINE, lw=0.8,
            solid_capstyle="butt")                       # the one baseline the bars grow from

    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main(Path(__file__).with_name("results_logp.png"))
