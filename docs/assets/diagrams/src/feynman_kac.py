"""Feynman-Kac particles (Steering, "Feynman-Kac steering").

K candidates denoise side by side, from noise (left) to molecule (right). During the warm-up nothing is resampled. Later, at
each checkpoint, every candidate gets a weight from how far its predicted clean graph is from the target: the heavy ones are
cloned and the light ones are dropped, so the population concentrates on the candidates that head for the target. The
weights, the parents and the molecules are drawn for illustration; they are not data.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/feynman_kac.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'feynman-kac'
SIZE = (440, 214)

# -- layout (pt)
K = 6
ROWS = [64 + 20 * i for i in range(K)]            # one row per candidate
X0, X_END = 46, 394                               # a lineage starts and ends here
X_WARM = 256                                      # the end of the warm-up zone
CHECKS = (282, 328, 374)                          # the checkpoints
JUMP = 14                                         # horizontal length of a re-routed lineage
ZONE_Y0, ZONE_Y1 = 36, 178
AXIS_Y = 190
X_MOL = 418

# -- what happens (illustrative). Weights of the K candidates at each checkpoint, and which candidate each of the K that
# continue descends from (monotone, so that lineages do not cross).
WEIGHTS = [[0.20, 0.90, 0.50, 0.15, 0.80, 0.35],
           [0.45, 0.55, 0.20, 0.90, 0.60, 0.25],
           [0.85, 0.30, 0.60, 0.70, 0.20, 0.90]]
PARENTS = [[1, 1, 2, 4, 4, 5],
           [0, 1, 3, 3, 4, 4],
           [0, 0, 2, 3, 5, 5]]
KINDS = ['benzene', 'phenol', 'toluene', 'pyridine', 'furan', 'ether']
NOTE = dict(size=6.5, color=GREY)


def cross(fig, x, y, s=2.6):
    for d in (-1, 1):
        fig.line(x - s, y - s * d, x + s, y + s * d, stroke=DGREY, sw=1.1, cap='round', name='dropped')


def build(fig):
    background(fig)
    # --- the zones and the checkpoints
    fig.box(X0 - 8, ZONE_Y0, X_WARM - X0 + 8, ZONE_Y1 - ZONE_Y0, fill=GROUP_GREY, radius=0, name='warm-up')
    fig.text((X0 - 8 + X_WARM) / 2, ZONE_Y0 + 10, 'warm-up: no resampling', justify='center', **NOTE)
    for x in CHECKS:
        fig.line(x, ZONE_Y0 + 14, x, ZONE_Y1, stroke=LABEL_GREY, sw=0.6, dash=[2, 2], name='checkpoint')
    fig.text(CHECKS[1], ZONE_Y0 + 10, 'checkpoints', justify='center', **NOTE)
    fig.rich((CHECKS[0] + CHECKS[-1]) / 2, 16, [('w', 'mi'), SP, ('∝', 'mm'), SP, ('exp', 'mr'), ('(', 'mr'), ('−', 'mm'),
                                              ('β', 'mm'), SP, ('E', 'mi'), (')', 'mr')], size=9, justify='center')
    fig.text((CHECKS[0] + CHECKS[-1]) / 2, 27, 'E: how far the predicted clean graph is from the target', justify='center',
             size=5.8, color=GREY)

    # --- the lineages, interval by interval
    starts = list(range(K))                       # which row each lineage occupies at the start of the interval
    xs = [X0] + [x + JUMP for x in CHECKS]        # where a horizontal run begins
    ends = list(CHECKS) + [X_END]                 # where it ends
    for step in range(len(CHECKS) + 1):
        for row in range(K):
            fig.line(xs[step], ROWS[row], ends[step], ROWS[row], sw=1.0, name='lineage')
        if step == len(CHECKS):
            break
        x = CHECKS[step]
        parents = PARENTS[step]
        for row in range(K):                      # a candidate nobody descends from is dropped
            if row not in parents:
                cross(fig, x + 4, ROWS[row])
        for row, parent in enumerate(parents):    # the K that continue: a child in its parent's row carries on, any other is a clone
            if parent != row:
                fig.path([(x, ROWS[parent]), (x + JUMP, ROWS[row])], stroke=FK_LINE, sw=1.3, name='clone')
            else:
                fig.line(x, ROWS[row], x + JUMP, ROWS[row], sw=1.0, name='lineage')
        for row in range(K):                      # the weights, as small bars on the checkpoint
            w = WEIGHTS[step][row]
            h = 3 + 8 * w
            fig.rect(x - 7, ROWS[row] - h / 2, 4, h, radius=1, fill=DGREY, stroke='none', opacity=70, name='weight')
    # direct labels where the eye lands: the weights, a dropped candidate, a cloned one (first checkpoint)
    fig.text(CHECKS[0] - 5, ZONE_Y1 + 9, 'weights', justify='center', **NOTE)
    fig.text(CHECKS[0] + 12, ROWS[0] - 7, 'dropped', **NOTE)
    fig.text(CHECKS[0] + JUMP + 4, (ROWS[0] + ROWS[1]) / 2 + 2.3, 'cloned', **NOTE)
    # candidates at the start: a little noise each, and a dot where the lineage begins
    for row in range(K):
        pos, bonds, fills = noise(X0 - 20, ROWS[row], 15, 11, n=4, seed=11 + row, b=4.2)
        fig.graph(pos, bonds, fills, 1.6, bond_w=0.8, node_w=0.6, opacity=55, name='noise')
        fig.ellipse(X0, ROWS[row], 1.7, fill=BLACK, stroke='none', name='start')
    # the finished candidates
    for row, kind in enumerate(KINDS):
        draw_molecule(fig, kind, X_MOL, ROWS[row], b=5.0, r=1.9, name='sample')
    fig.vtext(9, (ROWS[0] + ROWS[-1]) / 2, 'K candidates', size=6.5, font='rc', color=GREY)

    # --- the time axis
    fig.arrow([(X0 - 8, AXIS_Y), (X_END + 6, AXIS_Y)], head=3.2, sw=1.0)
    fig.text(X0 - 8, AXIS_Y + 12, 'noise', **NOTE)
    fig.text((X0 + X_END) / 2, AXIS_Y + 12, 'denoising time', justify='center', **NOTE)
    fig.text(X_MOL + 10, AXIS_Y + 12, 'molecules', justify='right', **NOTE)
