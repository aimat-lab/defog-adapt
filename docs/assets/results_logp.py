"""The README's results figure: the logP error of each way of conditioning a ZINC model, as a bar chart.

    python docs/assets/results_logp.py            # reads results_logp.json, writes results_logp.png and results_logp_dark.png

``results_logp.json`` is the summary that ``examples/compare_arms__zinc.py summarize`` writes for the held-out
test pass (ZINC250k, 100 targets drawn from real molecules, 10 generations each, three seeds); this script only
draws it. The numbers live there, with their protocol, so a chart cannot drift from the data it shows. They are
discussed, with what they do not support, in docs/results.md and docs/caveats.md.

Form: one arm is the point (Feynman-Kac on the cross-attention adapter, the end of the progression), so it is
drawn in the accent and the rest in the de-emphasis gray. The two approaches without an adapter (the unconditional
base, and a model trained end to end on the condition) sit above a hairline, the three adapter arms below it, and a
small header names each group. Every bar is labelled with its mean and sd over the sampling seeds, and the whisker is
the same sd, so there is no legend and no axis ticks; the table view is docs/results.md. The protocol, and that each
approach was trained once, are in the footer, so the figure does not claim more certainty than it has when it is seen
on its own.

Colours and ink are the dataviz reference palette, on its light surface and, for readers on a dark page, its dark
surface (``results_logp_dark.png``, chosen by the README's <picture> element). The one colour that is not in the
palette is the context gray of the bars: the palette's baseline hairline is only 1.75:1 against the light surface,
too pale for a mark, so the bars use a gray that clears 3:1 on their surface.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

#: (arm, label) in drawing order; the first ``N_REFERENCES`` are the approaches without an adapter, above the hairline.
ROWS = [
    ("base", "unconditional base"),
    ("conditional", "trained end to end on logP"),
    ("film", "FiLM adapter"),
    ("xattn", "cross-attention adapter"),
    ("xattn_fk", "cross-attention adapter + Feynman-Kac"),
]
N_REFERENCES = 2
HEADLINE = "xattn_fk"
GROUPS = ("Without an adapter", "Adapter on the frozen base")

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "accent": "#2a78d6", "context": "#908f87",
              "hairline": "#e1e0d9", "baseline": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "accent": "#3987e5", "context": "#6c6b65",
             "hairline": "#2c2c2a", "baseline": "#383835"},
}
FONT = "Roboto Condensed"

W = 760                                              # canvas width in "1x" pixels; the file is written at 2x
LEFT = 24
LABEL_X, BAR_X0, MAX_BAR = 256, 270, 340             # labels end at LABEL_X; bars start at BAR_X0; the longest is MAX_BAR px
PITCH, THICK, RADIUS = 40, 20, 4
HEADER_TO_ROW = 26                                   # a group header's baseline to the centre of its first row


def layout():
    """Vertical positions in "1x" pixels: the two group-header baselines, every row's centre, the hairline between
    the groups, the footer's baseline and the canvas height."""
    first_header = 96
    centres = [first_header + HEADER_TO_ROW + PITCH * i for i in range(N_REFERENCES)]
    hairline = centres[-1] + THICK / 2 + 18
    second_header = hairline + 24
    centres += [second_header + HEADER_TO_ROW + PITCH * i for i in range(len(ROWS) - N_REFERENCES)]
    footer = centres[-1] + THICK / 2 + 36
    return (first_header, second_header), centres, hairline, footer, int(footer + 20)


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


def main(data_path: Path, out: Path, theme: str = "light") -> None:
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; choose one of {sorted(THEMES)}")
    c = THEMES[theme]
    data = load(data_path)
    protocol, arms = data["protocol"], data["arms"]
    n_seeds = len(arms[HEADLINE]["seeds"])
    top = max(arms[arm]["mae_mean"] + (arms[arm]["mae_sd"] or 0.0) for arm, _ in ROWS)
    scale = MAX_BAR / top
    headers, centres, hairline, footer, H = layout()

    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=200, facecolor=c["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    px = 72 / 100                                    # one "1x" pixel in points
    ax.text(LEFT, 34, "Error to the logP target, by approach (lower is better)", fontsize=16.5 * px,
            fontweight="medium", color=c["ink"], va="baseline")
    ax.text(LEFT, 56, "How far the generated molecules land from the requested logP: mean absolute error of the decoded "
                      "logP.", fontsize=12.5 * px, color=c["ink2"], va="baseline")
    for header, text in zip(headers, GROUPS):
        ax.text(LEFT, header, text, fontsize=12 * px, fontweight="medium", color=c["ink2"], va="baseline")
    ax.plot([LEFT, W - LEFT], [hairline, hairline], color=c["hairline"], lw=0.8, solid_capstyle="butt")

    for group in (centres[:N_REFERENCES], centres[N_REFERENCES:]):   # the one baseline each group's bars grow from
        ax.plot([BAR_X0, BAR_X0], [group[0] - THICK / 2 - 6, group[-1] + THICK / 2 + 6], color=c["baseline"], lw=0.8,
                solid_capstyle="butt")

    for (arm, label), y in zip(ROWS, centres):
        mean, sd = arms[arm]["mae_mean"], arms[arm]["mae_sd"]
        headline = arm == HEADLINE
        length = mean * scale
        rounded_bar(ax, BAR_X0, y, length, c["accent"] if headline else c["context"])
        end = BAR_X0 + length
        if sd:                                       # the whisker: ±1 sd, with end caps and a surface halo so it reads on the bar
            lo, hi = BAR_X0 + max(mean - sd, 0.0) * scale, BAR_X0 + (mean + sd) * scale
            for width, color in ((2.6, c["surface"]), (0.9, c["ink"])):
                ax.plot([lo, hi], [y, y], color=color, lw=width, solid_capstyle="butt")
                for x in (lo, hi):
                    ax.plot([x, x], [y - 5.5, y + 5.5], color=color, lw=width, solid_capstyle="butt")
            end = hi
        ax.text(LABEL_X, y + 4.6, label, fontsize=13.5 * px, ha="right", va="baseline",
                color=c["ink"] if headline else c["ink2"], fontweight="medium" if headline else "regular")
        value = ax.text(end + 10, y + 4.8, f"{mean:.3f}", fontsize=13.5 * px, ha="left", va="baseline", color=c["ink"],
                        fontweight="medium" if headline else "regular")
        if sd:
            fig.canvas.draw()
            right = value.get_window_extent().transformed(ax.transData.inverted()).x1
            ax.text(right + 4, y + 4.8, f"± {sd:.3f}", fontsize=12 * px, ha="left", va="baseline", color=c["ink2"])

    ax.text(LEFT, footer, f"ZINC250k {protocol['split']} split · {protocol['n_targets']} targets × "
                          f"{protocol['n_per_target']} generations · bars: mean of {n_seeds} sampling seeds · "
                          f"whiskers: ±1 sd · one trained model per approach",
            fontsize=11.5 * px, color=c["ink2"], va="baseline")

    fig.savefig(out, dpi=200, facecolor=c["surface"])
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    here = Path(__file__).parent
    data_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results_logp.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here
    main(data_arg, out_dir / "results_logp.png", "light")
    main(data_arg, out_dir / "results_logp_dark.png", "dark")
