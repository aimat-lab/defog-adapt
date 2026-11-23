"""The three tools and when each one acts (Concepts, overview).

Rows are the tools, columns are Training and Sampling. A snowflake badge marks weights that stay frozen, a delta-W badge
weights that change. The adapter is trained once beside a frozen base and then acts at every sampling step; Feynman-Kac
steering trains nothing and acts only while sampling; RL fine-tuning changes the weights and needs no reward afterwards.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/tools_overview.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'tools-overview'
SIZE = (440, 336)

# -- layout (pt). Every position derives from these numbers.
X_LABEL = 46                      # centre of the row pills
BAND_X0 = 98                      # left edge of the row bands
COL_T, COL_S = 106, 284           # left edges of the Training and Sampling cells
X_SEP = 276                       # dashed separator between the two columns
HEAD_Y = 13                       # centre line of the column header pills
ROW_Y = (72, 162, 252)            # centre of each row band
ROW_H = 82
CY = 5                            # the model's centre line, below the band centre (room above for the adapter)
LEGEND_Y = 318
MW, MH, TAPER = 34, 27, 0.38      # the model trapezoid
B, AR = 6.0, 2.3                  # bond length and node radius of the small molecules
IN_X, MODEL_DX, OUT_X = 16, 42, 128   # inside a Sampling cell: where the noise, the model and the result sit
GREY_NOTE = dict(size=6.5, color=GREY)


def cell_model_xy(x0, cy):
    return x0 + MODEL_DX, cy - MH / 2


def adapter_on(fig, x_model, y_model, delta=False):
    """Dock the adapter on the model from above, as the banner does; returns the adapter's centre x and top y."""
    ax = x_model + 0.36 * MW
    top = y_model + MH * TAPER / 2 * (ax - x_model) / MW          # y of the model's slanted top edge at ax
    a_top = top - 4 - 15
    adapter_block(fig, ax, a_top)
    fig.line(ax, a_top + 15, ax, top, sw=1.1, name='adapter into base')
    if delta:
        fig.badge(ax + 15.5, a_top + 5, 5.4, [('Δ', 'mm'), ('W', 'mi')], size=5.0)
    return ax, a_top


def sample_flow(fig, x0, cy, model_fill=None, adapter=False, out_kind='phenol', seed=3, label_below=None):
    """noise -> model -> molecule. The model is frozen (stripes and a snowflake) unless a fill says it was fine-tuned."""
    draw_noise(fig, x0 + IN_X, cy, seed=seed)
    xm, ym = cell_model_xy(x0, cy)
    fig.line(x0 + IN_X + 15, cy, xm, cy, sw=STYLE['arrow_sw'], name='into model')
    if model_fill is None:
        frozen_model(fig, xm, ym, MW, MH, '')
    else:
        live_model(fig, xm, ym, MW, MH, model_fill, delta=False)
    fig.arrow([(xm + MW + 3, cy), (x0 + OUT_X - 15, cy)], head=3.0)
    draw_molecule(fig, out_kind, x0 + OUT_X, cy, b=B, r=AR)
    if label_below:
        fig.text(x0 + 82, cy + 26, label_below, justify='center', **GREY_NOTE)
    return xm, ym


def build(fig):
    # --- the grid: row bands, the dashed separator, column headers
    for y in ROW_Y:
        fig.box(BAND_X0, y - ROW_H / 2, SIZE[0] - BAND_X0 - 4, ROW_H, fill=GROUP_GREY, radius=6, name='row band')
        fig.line(X_SEP, y - ROW_H / 2 + 9, X_SEP, y + ROW_H / 2 - 9, stroke=LABEL_GREY, sw=0.6, dash=[2, 2], name='separator')
    x0, x1 = fig.pill(COL_T + 78, HEAD_Y, 'Training', anchor='center')
    y0, y1 = fig.pill(COL_S + 78, HEAD_Y, 'Sampling', anchor='center')
    fig.arrow([(x1 + 9, HEAD_Y), (y0 - 9, HEAD_Y)], head=3.2, sw=STYLE['arrow_sw'])
    fig.pill(X_LABEL, ROW_Y[0], 'Adapter', fill=ADAPTER_FILL, anchor='center')
    fig.pill(X_LABEL, ROW_Y[1], 'Feynman-Kac', fill=FK_FILL, anchor='center')
    fig.pill(X_LABEL, ROW_Y[2], 'RL fine-tuning', fill=RL_FILL, anchor='center')

    # --- row 1, Adapter. Training: a labelled molecule, a frozen base, the adapter beside it learning.
    cy = ROW_Y[0] + CY
    draw_molecule(fig, 'pyridine', COL_T + 14, cy - 3, b=B, r=AR)
    fig.emoji('💧', COL_T + 14, cy + 14, size=8)
    xm, ym = cell_model_xy(COL_T, cy)
    fig.line(COL_T + 31, cy, xm, cy, sw=STYLE['arrow_sw'], name='into model')
    frozen_model(fig, xm, ym, MW, MH, '')
    adapter_on(fig, xm, ym, delta=True)
    fig.text(xm + MW + 9, cy - 1, 'trained once', **GREY_NOTE)
    fig.text(xm + MW + 9, cy + 8.5, 'per property', **GREY_NOTE)
    # Sampling: noise -> frozen base with the adapter docked -> molecule
    cy = ROW_Y[0] + CY
    xm, ym = sample_flow(fig, COL_S, cy, adapter=True, out_kind='phenol', seed=3)
    ax, a_top = adapter_on(fig, xm, ym)
    fig.text(ax + 14, a_top + 10, 'acts at every step', **GREY_NOTE)
    fig.emoji('💧', ax, a_top - 7, size=8)

    # --- row 2, Feynman-Kac. Training: nothing. Sampling: candidates are resampled while the base denoises.
    cy = ROW_Y[1] + CY
    fig.rect(COL_T + 12, cy - 17, 140, 34, radius=6, fill='none', stroke=LABEL_GREY, sw=0.8, dash=[3, 2.2], name='nothing trained')
    fig.text(COL_T + 82, cy + 2.6, 'nothing is trained', justify='center', size=7, color=GREY)
    draw_noise(fig, COL_S + 13, cy, w=22, seed=5)
    xm, ym = COL_S + 34, cy - MH / 2
    fig.line(COL_S + 26, cy, xm, cy, sw=STYLE['arrow_sw'], name='into model')
    frozen_model(fig, xm, ym, MW, MH, '')
    bx = COL_S + 86                                               # the resample badge sits on the arrow to the candidates
    fig.line(xm + MW + 3, cy, bx - 5.4, cy, sw=STYLE['arrow_sw'], name='out of model')
    fig.arrow([(bx + 5.4, cy), (COL_S + 100 - 3, cy)], head=3.0)
    fig.badge(bx, cy, 5.4, [('↻', 'mm')], size=7)
    fig.text(bx, cy - 10, 'resample', justify='center', **GREY_NOTE)
    fig.box(COL_S + 100, cy - 30, 46, 60, fill=FK_FILL, radius=6, name='candidates')
    for k, (kind, op) in enumerate((('toluene', None), ('furan', 38), ('amide', None))):
        y = cy - 19 + 19 * k
        draw_molecule(fig, kind, COL_S + 128, y, b=5.2, r=2.0, opacity=op)
        if op:                                                    # a dropped candidate: crossed out beside it
            for s in (-1, 1):
                fig.line(COL_S + 108 - 3, y - 3 * s, COL_S + 108 + 3, y + 3 * s, stroke=DGREY, sw=1.0, cap='round', name='dropped')

    # --- row 3, RL. Training: sample, reward, update (a loop). Sampling: the fine-tuned model, no reward.
    cy = ROW_Y[2] + CY
    xm, ym = COL_T + 8, cy - MH / 2 - 4
    live_model(fig, xm, ym, MW, MH, RL_FILL, delta=True)
    cx_mol, cx_r = COL_T + 78, COL_T + 120
    fig.arrow([(xm + MW + 3, ym + MH / 2), (cx_mol - 14, ym + MH / 2)], head=3.0)
    draw_molecule(fig, 'ether', cx_mol, ym + MH / 2, b=B, r=AR)
    fig.arrow([(cx_mol + 15, ym + MH / 2), (cx_r - 8, ym + MH / 2)], head=3.0)
    fig.badge(cx_r, ym + MH / 2, 6, [('r', 'mi')], size=7)
    fig.text(cx_mol, ym + MH / 2 - 15, 'sample', justify='center', **GREY_NOTE)
    fig.text(cx_r, ym + MH / 2 - 15, 'reward', justify='center', **GREY_NOTE)
    loop_y = ym + MH + 12
    fig.arrow([(cx_r, ym + MH / 2 + 8), (cx_r, loop_y), (xm + MW * 0.42, loop_y), (xm + MW * 0.42, ym + MH - 3)], head=3.0)
    fig.text((cx_r + xm + MW * 0.42) / 2, loop_y - 3, 'update', justify='center', **GREY_NOTE)
    sample_flow(fig, COL_S, cy, model_fill=RL_FILL, out_kind='thiophene', seed=7, label_below='no reward needed')

    # --- legend
    fig.legend(LEGEND_Y, [(*frozen_symbol(), 'Weights stay frozen'), (*delta_symbol(), 'Weights change')], x0=130, x1=310)
