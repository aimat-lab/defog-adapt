"""Who does what when steering toward a target (Guides, "Steer toward a target").

The size model draws the number of atoms from P(n | target). Inside the Feynman-Kac sampler the proposal (the frozen base with
the adapter) generates candidates, the ruler (the property head) scores each candidate's predicted clean graph, and the
candidates are resampled; that loop is repeated at every checkpoint. Afterwards RDKit decodes the graphs and the judge
measures the true property of each molecule. The ruler steers the run; it is not a measurement.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/steering_roles.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'steering-roles'
SIZE = (440, 148)

# -- layout (pt)
Y = 90                                        # the centre line of the stages
X_SIZE, SIZE_W, SIZE_H = 8, 64, 32            # the size model
GROUP = (104, 14, 222, 128)                   # the sampler: x, y, w, h
X_PROP, MW, MH = 118, 40, 32                  # the proposal (frozen base with the adapter docked)
X_RULER, RW, RH = 248, 48, 36                 # the ruler (the property head)
X_JUDGE, JUDGE_W, JUDGE_H = 356, 78, 54
NOTE = dict(size=6.5, color=GREY)


def build(fig):
    # --- the size model, fed by the target
    fig.emoji('💧', X_SIZE + SIZE_W / 2, 42, size=11)
    fig.text(X_SIZE + SIZE_W / 2, 60, 'target', justify='center', **NOTE)
    fig.line(X_SIZE + SIZE_W / 2, 65, X_SIZE + SIZE_W / 2, Y - SIZE_H / 2, sw=STYLE['arrow_sw'], name='into size model')
    fig.box(X_SIZE, Y - SIZE_H / 2, SIZE_W, SIZE_H, fill=PILL_GREY, radius=5, name='size model')
    fig.text(X_SIZE + SIZE_W / 2, Y + 2.6, 'Size model', justify='center', size=7, font='rcm')
    fig.rich(X_SIZE + SIZE_W / 2, Y + 29, [('P', 'mi'), ('(', 'mr'), ('n', 'mi'), SP, ('|', 'mm'), SP, ('target', 'rc'), (')', 'mr')],
             size=7, color=GREY, justify='center')
    fig.arrow([(X_SIZE + SIZE_W + 3, Y), (GROUP[0] - 2, Y)], head=3.2)
    fig.text((X_SIZE + SIZE_W + GROUP[0]) / 2, Y - 6, 'n atoms', justify='center', **NOTE)

    # --- the sampler: a proposal and a ruler in a loop
    gx, gy, gw, gh = GROUP
    fig.group(gx, gy, gw, gh, label='Feynman-Kac sampler')
    ym = Y - MH / 2
    frozen_model(fig, X_PROP, ym, MW, MH, '')
    ax = X_PROP + 0.36 * MW
    top = ym + MH * 0.38 / 2 * (ax - X_PROP) / MW
    a_top = top - 4 - 15
    adapter_block(fig, ax, a_top)
    fig.line(ax, a_top + 15, ax, top, sw=1.1, name='adapter into base')
    fig.emoji('💧', ax, a_top - 7, size=8)
    fig.pill(X_PROP + MW / 2, 38, 'Proposal', fill=ADAPTER_FILL, anchor='center')
    fig.text(X_PROP + MW / 2, Y + 30, 'base + adapter', justify='center', **NOTE)

    fig.model(X_RULER, Y - RH / 2, RW, RH, 'Property\nhead', fill=FK_FILL, font='rcm', size=7, name='ruler')
    fig.emoji('💧', X_RULER + 12, Y - RH / 2 - 8, size=8)
    fig.pill(X_RULER + RW / 2, 38, 'Ruler', fill=FK_FILL, anchor='center')
    fig.text(X_RULER + RW / 2, Y + 30, 'steers the run', justify='center', size=6.5, font='rcm', color=BLACK)

    # the loop: candidates go to the ruler (upper arrow), the resampled ones come back (lower arrow)
    up, dn = Y - 6, Y + 7
    x0, x1 = X_PROP + MW + 3, X_RULER - 3
    mid = (x0 + x1) / 2
    fig.arrow([(x0, up), (x1, up)], head=3.0)
    fig.text(mid, up - 5, 'predicted clean graphs', justify='center', **NOTE)
    fig.arrow([(x1, dn), (x0, dn)], head=3.0)
    fig.badge(mid, dn, 5.0, [('↻', 'mm')], size=6.5)
    fig.text(mid, dn + 15, 'resample at checkpoints', justify='center', **NOTE)

    # --- the judge: decode, then measure the true property
    fig.arrow([(gx + gw + 3, Y), (X_JUDGE - 2, Y)], head=3.2)
    fig.text((gx + gw + X_JUDGE) / 2, Y - 6, 'decode', justify='center', **NOTE)
    fig.card(X_JUDGE, Y - JUDGE_H / 2, JUDGE_W, JUDGE_H, fill=WHITE, name='judge')
    fig.text(X_JUDGE + JUDGE_W / 2, Y - 5, 'RDKit', justify='center', size=7.5, font='rcm')
    fig.text(X_JUDGE + JUDGE_W / 2, Y + 6, 'true property', justify='center', size=6.3)
    fig.text(X_JUDGE + JUDGE_W / 2, Y + 14.5, 'validity, uniqueness', justify='center', size=6.3)
    fig.pill(X_JUDGE + JUDGE_W / 2, 38, 'Judge', anchor='center')
    fig.text(X_JUDGE + JUDGE_W / 2, Y + 40, 'measures the run', justify='center', size=6.5, font='rcm', color=BLACK)
