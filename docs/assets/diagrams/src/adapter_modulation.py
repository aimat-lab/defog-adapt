"""What the adapter modulates (Adapters, "What it is").

(a) The condition (through Fourier bands) and the flow time (through its embedding) go into the adapter. For every layer of
the frozen base it returns a scale, a shift and a gate, which modulate that layer's node (X), edge (E) and global (y)
streams. The gates start at zero, so an untrained adapter leaves the base unchanged.
(b) In addition, each atom reads the condition by cross-attention (nodes only).

Build:  python3 $FIGURE_SKILL/scripts/build.py docs/assets/diagrams/src/adapter_modulation.py [--final]
"""
from figlib import *  # noqa: F401,F403
from vocab import *  # noqa: F401,F403

NAME = 'adapter-modulation'
SIZE = (440, 214)

# -- layout (pt). Panel a spans x 0..298, panel b x 316..440.
X_IN = 22                                   # centre of the two inputs
Y_C, Y_T = 72, 140                          # centre lines of the condition and the flow-time input
X_PRE, PRE_W, PRE_H = 44, 50, 15            # the two small pre-processing boxes
X_AD, AD_W, AD_H, Y_MID = 114, 32, 80, 106  # the adapter trapezoid and its centre line
X_BUS = 160                                 # the vertical line that distributes the adapter's output
LAYERS = (57, 106, 155)                     # centre lines of the three layers (the vectors share them)
X_VEC, CW, CH = 172, 11, 12                 # the scale / shift / gate vector beside each layer
X_STACK, STACK_W, STACK_Y, STACK_H = 226, 68, 26, 160   # the frozen base
CARD_W, CARD_H = 56, 38
X_B = 318                                   # panel b
GATE_NOTE_Y = 178

SCALE_C, SHIFT_C, GATE_C = tint(ORANGE, 0.5), tint(ORANGE, 0.72), WHITE     # the gate cell is empty: it starts at zero
NOTE = dict(size=6.5, color=GREY)


def build(fig):
    fig.panel_title(4, 12, 'a', 'Scale, shift and gate for every layer')

    # --- inputs: the condition (a target property) and the flow time, each through a small pre-processing box
    fig.emoji('💧', X_IN, Y_C - 3, size=14)
    fig.rich(X_IN, Y_C + 19, [('c', 'mi')], size=8, justify='center')
    fig.text(X_IN, Y_C + 28, 'Condition', justify='center', **NOTE)
    fig.line(X_IN - 11, Y_T, X_IN + 11, Y_T, stroke=DGREY, sw=0.9, name='time axis')
    fig.ellipse(X_IN + 3, Y_T, 2.5, fill=BLACK, stroke='none', name='time knob')
    fig.rich(X_IN, Y_T + 19, [('t', 'mi')], size=8, justify='center')
    fig.text(X_IN, Y_T + 28, 'Flow time', justify='center', **NOTE)
    for y, label in ((Y_C, 'Fourier bands'), (Y_T, 'time embedding')):
        fig.line(X_IN + 13, y, X_PRE, y, sw=STYLE['arrow_sw'], name='into box')
        fig.box(X_PRE, y - PRE_H / 2, PRE_W, PRE_H, fill=PILL_GREY, radius=4, name='pre-processing')
        fig.text(X_PRE + PRE_W / 2, y + 2.4, label, justify='center', size=6.3)
        fig.line(X_PRE + PRE_W, y, X_AD, y, sw=STYLE['arrow_sw'], name='into adapter')

    # --- the adapter
    fig.model(X_AD, Y_MID - AD_H / 2, AD_W, AD_H, '', taper=0.4, fill=ADAPTER_FILL, name='adapter')
    fig.vtext(X_AD + AD_W / 2 - 2, Y_MID, 'Adapter', size=8, font='rcm')
    fig.emoji('🔥', X_AD + AD_W / 2, Y_MID - AD_H / 2 - 9, size=12)

    # --- the frozen base first (a striped container with three layers), so that the arrows can end on the layers
    CARD_X = X_STACK + (STACK_W - CARD_W) / 2
    bx, by = frozen_panel(fig, X_STACK, STACK_Y, STACK_W, STACK_H)
    names = ('Layer 1', 'Layer 2', None)
    for k, y in enumerate(LAYERS):
        fig.card(CARD_X, y - CARD_H / 2, CARD_W, CARD_H, fill=WHITE, name=f'layer {k + 1}')
        if names[k]:
            fig.text(CARD_X + 5, y - CARD_H / 2 + 9, names[k], size=6.5, font='rcm')
        else:
            fig.rich(CARD_X + 5, y - CARD_H / 2 + 9, [('Layer ', 'rc'), ('L', 'mi')], size=6.5)
        for j, (glyph, label) in enumerate((('X', 'nodes'), ('E', 'edges'), ('y', 'global'))):
            sx = CARD_X + 6 + 16.5 * j
            fig.rect(sx, y - 3, 14, 11, radius=2.5, fill=PILL_GREY, stroke=BLACK, sw=0.6, name=f'stream {glyph}')
            fig.rich(sx + 7, y + 5.3, [(glyph, 'mi')], size=7, justify='center')
            if k == 0:
                fig.text(sx + 7, y + 16, label, justify='center', size=5.6, color=GREY)
    frozen_badge(fig, bx, by)
    fig.text(X_STACK + STACK_W / 2, STACK_Y + STACK_H + 11, 'Frozen base', justify='center', size=6.5, font='rcm')

    # --- the output bus and one {scale, shift, gate} vector per layer, which modulates that layer
    fig.line(X_AD + AD_W + 3, Y_MID, X_BUS, Y_MID, sw=STYLE['arrow_sw'], name='out of adapter')
    fig.line(X_BUS, LAYERS[0], X_BUS, LAYERS[-1], sw=STYLE['arrow_sw'], name='bus')
    for k, y in enumerate(LAYERS):
        fig.arrow([(X_BUS, y), (X_VEC - 3, y)], head=2.8, sw=0.9)
        fig.cells(X_VEC, y - CH / 2, [SCALE_C, SHIFT_C, GATE_C], CW, CH, name=f'scale shift gate {k + 1}')
        for j, glyph in enumerate((('γ', 'mm'), ('β', 'mm'), ('g', 'mi'))):
            fig.rich(X_VEC + CW * (j + 0.5), y + 2.6, [glyph], size=7, justify='center')
        fig.arrow([(X_VEC + 3 * CW + 2, y), (CARD_X - 1.2, y)], head=2.8, sw=0.9)
    fig.text(X_VEC + 1.5 * CW, LAYERS[0] - 12, 'scale · shift · gate', justify='center', **NOTE)
    fig.text(X_VEC + 1.5 * CW, GATE_NOTE_Y, 'the gate starts at 0', justify='center', **NOTE)
    fig.line(X_VEC + 2.5 * CW, LAYERS[-1] + CH / 2 + 1, X_VEC + 2.5 * CW, GATE_NOTE_Y - 8, stroke=LABEL_GREY, sw=0.6,
             dash=[1.5, 1.5], name='gate leader')

    # --- panel b: each atom queries the condition (nodes only)
    fig.panel_title(X_B - 2, 12, 'b', 'Atoms read the condition')
    cx = X_B + 58
    pos, bonds, fills = molecule('toluene', cx, 76, 12)
    fig.graph(pos, bonds, fills, 3.8, name='atoms')
    fig.text(cx, 50, 'Nodes (X)', justify='center', **NOTE)
    tok_y, tok_w = 136, 20
    toks = [cx - tok_w + tok_w * k for k in range(3)]
    fig.cells(toks[0] - tok_w / 2, tok_y, [SCALE_C, SHIFT_C, tint(ORANGE, 0.35)], tok_w, 13, name='condition tokens')
    fig.text(cx, tok_y + 25, 'Condition tokens (from the adapter)', justify='center', **NOTE)
    low = sorted(range(len(pos)), key=lambda i: -pos[i][1])[:3]
    low.sort(key=lambda i: pos[i][0])
    for i, tx in zip(low, toks):
        fig.curved_arrow((pos[i][0], pos[i][1] + 5.5), ((pos[i][0] + tx) / 2 + (5 if tx > pos[i][0] else -5), 112), (tx, tok_y - 3),
                         color=BLACK, sw=0.9, head=2.4)
    fig.text(cx, tok_y + 38, 'nodes only: edges and global', justify='center', size=5.8, color=GREY)
    fig.text(cx, tok_y + 46, 'keep the scale, shift and gate', justify='center', size=5.8, color=GREY)
