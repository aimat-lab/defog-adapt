"""The README's results figure: the logP error of each way of conditioning a ZINC model, as a bar chart.

    python docs/assets/results_logp.py            # reads results_logp.json, writes results_logp.png

``results_logp.json`` is the summary that ``examples/compare_arms__zinc.py summarize`` writes for the held-out
test pass (ZINC250k, 100 targets drawn from real molecules, 10 generations each, three seeds); this script only
draws it. The numbers live there, with their protocol, so a chart cannot drift from the data it shows. They are
discussed, with what they do not support, in docs/results.md and docs/caveats.md.

Form: one arm is the point (Feynman-Kac on the cross-attention adapter, the end of the progression), so it is
drawn in the accent and the rest in the de-emphasis gray. The two references (nothing steering the base, and a
model trained end to end on the condition) sit above a hairline, the three adapter arms below it. Every bar is
labelled with its mean and sd over the sampling seeds, and the whisker is the same sd, so there is no legend and no axis
ticks; the table view is docs/results.md. Colours and ink are the dataviz reference palette's light surface.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

#: (arm, label) in drawing order; the first ``N_REFERENCES`` are the references above the hairline.
ROWS = [
    ("base", "unconditional base"),
    ("conditional", "trained end to end on logP"),
    ("film", "FiLM adapter"),
    ("xattn", "cross-attention adapter"),
    ("xattn_fk", "cross-attention adapter + Feynman-Kac"),
]
N_REFERENCES = 2
HEADLINE = "xattn_fk"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
ACCENT, CONTEXT, BASELINE, HAIRLINE = "#2a78d6", "#c3c2b7", "#c3c2b7", "#e1e0d9"
FONT = "Roboto Condensed"

W, H = 760, 330                      # canvas in "1x" pixels; the file is written at 2x
LABEL_X, BAR_X0, MAX_BAR = 256, 270, 330     # labels end at LABEL_X; bars start at BAR_X0; the longest is MAX_BAR px
ROW0, PITCH, GAP, THICK, RADIUS = 96, 38, 14, 18, 4


def rounded_bar(ax, x0, y_mid, length, color):
    """A bar that grows from a square baseline end to a 4 px rounded data end."""
    y0, y1, x1, r = y_mid - THICK / 2, y_mid + THICK / 2, x0 + length, RADIUS
    verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", lw=0))


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [arm for arm, _ in ROWS if arm not in data["arms"]]
    if missing:
        raise SystemExit(f"{path.name} has no result for {missing}; the figure needs all five arms")
    return data


def main(data_path: Path, out: Path) -> None:
    data = load(data_path)
    protocol, arms = data["protocol"], data["arms"]
    n_seeds = len(arms[HEADLINE]["seeds"])
    top = max(arms[arm]["mae_mean"] + (arms[arm]["mae_sd"] or 0.0) for arm, _ in ROWS)
    scale = MAX_BAR / top

    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    px = 72 / 100                    # one "1x" pixel in points
    ax.text(24, 30, "Error to the logP target, by approach (lower is better)", fontsize=15 * px,
            fontweight="medium", color=INK, va="baseline")
    ax.text(24, 52, f"ZINC250k {protocol['split']} split, {protocol['n_targets']} targets × "
                    f"{protocol['n_per_target']} generations. Bars: mean over {n_seeds} sampling seeds, one trained "
                    f"model per approach; whiskers: ±1 sd.",
            fontsize=12 * px, color=INK2, va="baseline")

    y = ROW0
    for k, (arm, label) in enumerate(ROWS):
        if k == N_REFERENCES:                            # the adapter arms sit below a hairline
            y += GAP
            ax.plot([24, W - 24], [y - PITCH / 2 + 1, y - PITCH / 2 + 1], color=HAIRLINE, lw=0.8,
                    solid_capstyle="butt")
        mean, sd = arms[arm]["mae_mean"], arms[arm]["mae_sd"]
        headline, reference = arm == HEADLINE, k < N_REFERENCES
        length = mean * scale
        rounded_bar(ax, BAR_X0, y, length, ACCENT if headline else CONTEXT)
        end = BAR_X0 + length
        if sd:                                           # the whisker: ±1 sd, with end caps
            lo, hi = BAR_X0 + max(mean - sd, 0.0) * scale, BAR_X0 + (mean + sd) * scale
            ax.plot([lo, hi], [y, y], color=INK, lw=1.0, solid_capstyle="butt")
            for x in (lo, hi):
                ax.plot([x, x], [y - 4, y + 4], color=INK, lw=1.0, solid_capstyle="butt")
            end = hi
        ax.text(LABEL_X, y + 4.2, label, fontsize=13 * px, ha="right", va="baseline",
                color=INK if headline else INK2, fontweight="medium" if headline else "regular",
                fontstyle="italic" if reference else "normal")
        value = ax.text(end + 10, y + 4.6, f"{mean:.3f}", fontsize=13 * px, ha="left", va="baseline", color=INK,
                        fontweight="medium" if headline else "regular")
        if sd:
            fig.canvas.draw()
            right = value.get_window_extent().transformed(ax.transData.inverted()).x1
            ax.text(right + 4, y + 4.6, f"± {sd:.3f}", fontsize=12 * px, ha="left", va="baseline", color=MUTED)
        y += PITCH
    ax.plot([BAR_X0, BAR_X0], [ROW0 - PITCH / 2 + 4, y - PITCH / 2 + GAP - 4], color=BASELINE, lw=0.8,
            solid_capstyle="butt")                       # the one baseline the bars grow from

    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    here = Path(__file__).parent
    data_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results_logp.json"
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "results_logp.png"
    main(data_arg, out_arg)
