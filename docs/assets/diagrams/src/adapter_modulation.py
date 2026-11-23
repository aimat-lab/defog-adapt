"""How an adapter plugs in (Concepts, Adapters).

The target (say logP = 4) and the flow time go into a small adapter. It gives every layer of the frozen base a scale, a shift
and a gate, and the noisy graph passes through those layers to come out a little cleaner. The gates start at zero, so an
untrained adapter changes nothing. Cross-attention, the second mechanism, is left to the text.

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/adapter_modulation.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'adapter-modulation'
SIZE = (440, 162)

# -- layout (pt). Every position derives from these numbers.
X_IN, W_IN, H_IN = 8, 88, 24              # the two inputs: left edge, width, height
IN_Y = (21, 51)                           # top edges of the target and the flow-time card
AD_X, AD_Y, AD_W, AD_H = 110, 22, 62, 52  # the adapter
BUS_Y = AD_Y + AD_H / 2                   # the line that carries the adapter's output to the layers
PANEL = (152, 88, 238, 56)                # the frozen base: x, y, w, h
LAYERS = ('layer 1', 'layer 2', '…', 'layer L')
LW, LH, LPITCH = 50, 34, 58               # a layer card: width, height, distance to the next
LX0, LY = PANEL[0] + 6, PANEL[1] + 11     # left edge of the first layer, top edge of all
FLOW_Y = LY + LH / 2                      # the graph flows through the base along this line
NOTE = dict(size=6.3, color=GREY)


def build(fig):
    background(fig)
    # --- the frozen base and its layers (drawn first: the arrows from the adapter land on top of them)
    px, py, pw, ph = PANEL
    bx, by = frozen_panel(fig, px, py, pw, ph)
    centres = []
    for k, name in enumerate(LAYERS):
        x = LX0 + LPITCH * k
        fig.card(x, LY, LW, LH, name=f'layer {k + 1}')
        fig.text(x + LW / 2, FLOW_Y + 2.4, name, justify='center', size=6.8)
        centres.append(x + LW / 2)
    for a, b in zip(centres, centres[1:]):
        fig.arrow([(a + LW / 2 + 1.5, FLOW_Y), (b - LW / 2 - 1.5, FLOW_Y)], head=2.4, sw=1.0)
    frozen_badge(fig, bx, by)
    fig.text(px + pw / 2, py + ph + 11, 'frozen base model', justify='center', size=7, font='rcm')

    # --- the graph flows through the base: noisy in, a little cleaner out
    draw_noise(fig, 34, FLOW_Y, w=30, h=26, n=7, seed=3, r=2.3)
    fig.text(34, FLOW_Y + 23, 'noisy', justify='center', **NOTE)
    fig.text(34, FLOW_Y + 31, 'graph', justify='center', **NOTE)
    fig.line(54, FLOW_Y, px + 1, FLOW_Y, sw=STYLE['arrow_sw'], name='into the base')
    fig.arrow([(px + pw + 3, FLOW_Y), (px + pw + 17, FLOW_Y)], head=3.0)
    draw_molecule(fig, 'toluene', 422, FLOW_Y, b=5.2, r=2.0)
    fig.text(422, FLOW_Y + 23, 'cleaner', justify='center', **NOTE)
    fig.text(422, FLOW_Y + 31, 'graph', justify='center', **NOTE)

    # --- the adapter and what it reads
    for y, title, sub in ((IN_Y[0], 'Target', 'e.g. logP = 4'), (IN_Y[1], 'Flow time', 'how noisy the graph is')):
        fig.card(X_IN, y, W_IN, H_IN, name=title)
        fig.text(X_IN + 24, y + 10.5, title, size=7, font='rcm')
        fig.text(X_IN + 24, y + 18.5, sub, **NOTE)
    fig.emoji('💧', X_IN + 12, IN_Y[0] + H_IN / 2, size=11)
    cy = IN_Y[1] + H_IN / 2                                     # flow time: a dot moving along a line
    fig.line(X_IN + 5, cy, X_IN + 19, cy, stroke=DGREY, sw=1.0, name='time')
    fig.ellipse(X_IN + 14, cy, 1.9, fill=BLACK, stroke='none', name='now')
    for y in (IN_Y[0] + H_IN / 2, IN_Y[1] + H_IN / 2):
        fig.line(X_IN + W_IN, y, AD_X, y, sw=STYLE['arrow_sw'], name='into the adapter')
    fig.model(AD_X, AD_Y, AD_W, AD_H, 'Adapter', fill=ADAPTER_FILL, font='rcm', size=7.5, name='adapter')
    fig.text(AD_X + AD_W / 2 - 2, AD_Y + AD_H + 10, 'small, trained', justify='center', **NOTE)

    # --- one signal per layer: a scale, a shift and a gate
    fig.line(AD_X + AD_W, BUS_Y, centres[-1], BUS_Y, sw=STYLE['arrow_sw'], name='to every layer')
    for x in centres:
        fig.arrow([(x, BUS_Y), (x, LY - 1)], head=3.0)
    mid = (centres[0] + centres[-1]) / 2
    fig.text(mid, BUS_Y - 15, 'scale · shift · gate for every layer', justify='center', size=7, font='rcm')
    fig.text(mid, BUS_Y - 6, 'the gates start at 0: an untrained adapter changes nothing', justify='center', **NOTE)
