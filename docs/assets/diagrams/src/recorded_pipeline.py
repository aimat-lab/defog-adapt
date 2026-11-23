"""The recorded pipeline (Concepts, overview).

Training, left to right: train the base; optionally clean it up with two rounds of RL; train one adapter per property while
the base stays frozen; optionally tighten the adapter with RL. Sampling: the adapter steers the frozen base at every step, and
Feynman-Kac steering may resample the candidates on top. A solid card is a required step and a dashed frame an optional one;
a snowflake marks frozen weights and delta-W weights that change. The figure shows an order, not a measurement.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/recorded_pipeline.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'recorded-pipeline'
SIZE = (440, 176)

# -- layout (pt). Every position derives from these numbers.
CW, PITCH, X0 = 62, 70, 8                 # a training card: width, distance from one to the next, left edge of the first
XS = [X0 + PITCH * i for i in range(4)]   # left edges of the four training cards
CARD_Y, CARD_H = 34, 116                  # top edge (where the name pill sits) and height of a training card
MY, MW, MH = 72, 40, 32                   # the model trapezoid in a training card: top, width, height
CY = MY + MH / 2                          # the line the pictograms and the arrows between cards sit on
CAP_Y = (120, 128)                        # baselines of the two caption lines
TAG_Y = 139                               # baseline of the 'optional' tag
GROUP_Y, GROUP_H = 6, 150
X_SAMPLE, W_SAMPLE = 300, 130             # the sampling stage: left edge and width of its two cards
LEGEND_Y = 166
NOTE = dict(size=6.3, color=GREY)


def step(fig, x, y, w, h, label, fill, optional=False):
    """A step: a solid card (required) or a dashed frame that lets the grey show through (optional), named by a pill on its
    top edge."""
    if optional:
        fig.rect(x, y, w, h, radius=4.5, fill='none', stroke=BLACK, sw=STYLE['outline'], dash=[6, 1.6],
                 name=f'optional {label}')
    else:
        fig.card(x, y, w, h, name=f'card {label}')
    fig.pill(x + w / 2, y, label, fill=fill, anchor='center')


def caption(fig, cx, lines, optional=False):
    for y, s in zip(CAP_Y, lines):
        fig.text(cx, y, s, justify='center', **NOTE)
    if optional:
        word = text_width('optional', 6.0) + 6
        tag(fig, cx - word / 2, TAG_Y, required=False)


def build(fig):
    background(fig)
    # --- the two stages
    xt1 = XS[3] + CW + 4
    fig.group(4, GROUP_Y, xt1 - 4, GROUP_H, label='Training')
    fig.group(X_SAMPLE - 6, GROUP_Y, SIZE[0] - 4 - (X_SAMPLE - 6), GROUP_H, label='Sampling')

    # --- the arrows come first, so that the cards cover where they start
    for a, b in zip(XS, XS[1:]):
        fig.arrow([(a + CW + 1.5, CY), (b - 1.5, CY)], head=2.6, sw=1.0)
    fig.arrow([(XS[3] + CW + 1.5, CY), (X_SAMPLE - 1.5, CY)], head=2.8, sw=1.0)

    # --- 1. train the base: it is unconditional
    cx = XS[0] + CW / 2
    step(fig, XS[0], CARD_Y, CW, CARD_H, 'Train base', PILL_GREY)
    fig.model(cx - MW / 2, MY, MW, MH, '', taper=0.38, name='base model')
    caption(fig, cx, ('knows molecules,', 'not properties'))

    # --- 2. RL cleanup (optional): the weights of the base change
    cx = XS[1] + CW / 2
    step(fig, XS[1], CARD_Y, CW, CARD_H, 'RL cleanup ×2', RL_FILL, optional=True)
    live_model(fig, cx - MW / 2, MY, MW, MH, RL_FILL, delta=True, name='base model')
    caption(fig, cx, ('fewer fragments', 'and odd rings'), optional=True)

    # --- 3. train an adapter: the base is frozen from here on
    cx = XS[2] + CW / 2
    step(fig, XS[2], CARD_Y, CW, CARD_H, 'Train adapter', ADAPTER_FILL)
    frozen_model(fig, cx - MW / 2, MY, MW, MH, '')
    dock_adapter(fig, cx - MW / 2, MY, MW, MH)
    caption(fig, cx, ('one per property', 'base stays frozen'))

    # --- 4. RL on the adapter (optional): the adapter's weights change
    cx = XS[3] + CW / 2
    step(fig, XS[3], CARD_Y, CW, CARD_H, 'RL on adapter', RL_FILL, optional=True)
    frozen_model(fig, cx - MW / 2, MY, MW, MH, '')
    dock_adapter(fig, cx - MW / 2, MY, MW, MH, delta=True)
    caption(fig, cx, ('tighter', 'conditioning'), optional=True)

    # --- 5. sample: noise -> the frozen base steered by the adapter -> a molecule
    sy, sh = CARD_Y, 64
    step(fig, X_SAMPLE, sy, W_SAMPLE, sh, 'Sample', PILL_GREY)
    my, mw, mh = sy + 27, 30, 24
    cy = my + mh / 2
    draw_noise(fig, X_SAMPLE + 17, cy, w=20, h=20, n=6, seed=5, r=2.0)
    fig.line(X_SAMPLE + 29, cy, X_SAMPLE + 38, cy, sw=STYLE['arrow_sw'], name='into model')
    frozen_model(fig, X_SAMPLE + 38, my, mw, mh, '', badge_r=5.6, snow_r=3.6)
    dock_adapter(fig, X_SAMPLE + 38, my, mw, mh)
    fig.arrow([(X_SAMPLE + 38 + mw + 3, cy), (X_SAMPLE + W_SAMPLE - 36, cy)], head=3.0)
    draw_molecule(fig, 'phenol', X_SAMPLE + W_SAMPLE - 20, cy, b=5.0, r=1.9)
    fig.text(X_SAMPLE + W_SAMPLE / 2, sy + sh - 4, 'the adapter steers every step', justify='center', **NOTE)

    # --- 6. Feynman-Kac (optional): resamples the candidates while sampling
    fy, fh = 108, CARD_Y + CARD_H - 108       # the stage ends on the same line as the training cards
    step(fig, X_SAMPLE, fy, W_SAMPLE, fh, 'Feynman-Kac', FK_FILL, optional=True)
    fk_glyph(fig, X_SAMPLE + 12, fy + 22, w=46, gap=9)
    fig.text(X_SAMPLE + 94, fy + 19, 'resamples the', justify='center', **NOTE)
    fig.text(X_SAMPLE + 94, fy + 27, 'candidates', justify='center', **NOTE)
    word = text_width('optional', 6.0) + 6
    tag(fig, X_SAMPLE + 94 - word / 2, fy + 36, required=False)

    # --- legend
    fig.legend(LEGEND_Y, [(*frozen_symbol(), 'weights stay frozen'), (*delta_symbol(), 'weights change'),
                          (*optional_symbol(), 'optional step')], x0=70, x1=370)
