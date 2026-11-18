"""README banner, one frame: wordmark on the left, and on the right a REAL recorded run of the frozen DeFoG base.

The animation has two beats on the SAME noise. Beat 1: the frozen base alone turns the noise into a molecule.
Beat 2: a flame-marked logP adapter docks on top of the frozen model and pulses at every step, and the same noise
ends in a molecule at the requested logP. Every atom and bond flip is a flip the model made (trajectory.json is
the recorded sampler state, see record_trajectories.py); only the node POSITIONS are a drawing choice, because a
graph has no coordinates: the noise is scattered and the final molecule uses an RDKit 2D layout, and the drawing
glides from one to the other while the categories flip.

This script draws ONE frame, chosen by $BANNER_FRAME; render_banner.py renders them all and assembles the GIF.

Build one frame:  BANNER_FRAME=120 python3 ~/.claude/skills/figure/scripts/build.py banner.py --scale 4
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
from figlib import *  # noqa: F401,F403
from figlib import _svg_file
from molecules import (assign_slots, draw_state, fit_layout, keep_apart, lerp_points, molecule_layout, pct,
                       scatter_layout)

NAME = 'banner'
SIZE = (468, 120)                  # about 4:1; the type sizes are those of the thesis figures (7 pt labels)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('BANNER_DATA', os.path.join(HERE, 'trajectory.json'))
FRAME = int(os.environ.get('BANNER_FRAME', '120'))

# ------------------------------------------------------------------------------ layout (pt)
MID = 58                           # centre line of the stations
LEFT = 14                          # left margin of the wordmark
WM_BASE, TAG_BASE = 62, 80         # baselines: wordmark (30 pt) and tagline
X_NOISE, NOISE_W, NOISE_H = 212, 52, 48          # noise graph: centre and extent of its scatter
X_MODEL, MW, MH, TAPER = 266, 60, 46, 0.38       # frozen DeFoG: left edge, width, height
X_SAMPLE, SAMPLE_W, SAMPLE_H, BOND_MAX = 407, 76, 40, 11.0
X_OUT = 352                        # where the arrow from the model ends
LABEL = 100                        # baseline of the station labels and of the step counter
CAPTION = 111                      # baseline of the logP caption under "Sample"
LOOP_Y = MID + MH / 2 + 8          # the feedback loop under the model
X_LOOP_OUT, X_LOOP_IN = X_MODEL + MW + 12, X_MODEL - 12
ADAPTER_X = X_MODEL + 0.36 * MW    # where the adapter docks, along the model's top edge
ADAPTER_W, ADAPTER_H, DOCK_GAP = 26, 17, 5

FROST, FROST_SW, FROST_GAP = '#b9dcff', 2.0, 5.0       # light blue stripes: frozen weights (as in the visual abstract)
ADAPTER_FILL = tint(ORANGE, 0.78)                      # the trained adapter's colour (flame)

# ------------------------------------------------------------------------------ timeline (15 fps)
PERIOD = 10                                            # frames per round of the feedback loop (one model application)
N_DEN = 50                                             # frames of visible denoising per beat
T_INTRO, T_HOLD1, T_SWAP, T_HOLD2, T_RESET = 8, 22, 22, 28, 14
B_BEAT1 = T_INTRO
B_HOLD1 = B_BEAT1 + N_DEN
B_SWAP = B_HOLD1 + T_HOLD1
B_BEAT2 = B_SWAP + T_SWAP
B_HOLD2 = B_BEAT2 + N_DEN
B_RESET = B_HOLD2 + T_HOLD2
FRAMES = B_RESET + T_RESET


def smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3 - 2 * k)


def ease_out(k):
    return 1 - (1 - min(max(k, 0.0), 1.0)) ** 3


def ease_in(k):
    return min(max(k, 0.0), 1.0) ** 3


def timeline(f):
    """What is on screen at frame f. The last frame equals the first, so the loop is seamless."""
    s = SimpleNamespace(run=None, p=0.0, fade_noise=1.0, fade_result=0.0, a=0.0, caption=0.0, glow=0.0,
                        pulse=False, running=False, show_result=False)
    if f < B_BEAT1:
        pass
    elif f < B_HOLD1:
        s.run, s.p, s.running = 'base', (f - B_BEAT1) / (N_DEN - 1), True
    elif f < B_SWAP:
        k = f - B_HOLD1
        s.run, s.p, s.show_result = 'base', 1.0, True
        s.fade_noise, s.fade_result, s.caption = 0.0, 1.0, smooth(k / 6)
    elif f < B_BEAT2:
        k = f - B_SWAP                           # the result fades out, then the noise fades in (little overlap)
        s.run, s.p, s.show_result = 'base', 1.0, k < 4
        s.fade_result, s.fade_noise = 1 - smooth(k / 4), smooth((k - 3) / 3)
        s.caption, s.a = 1 - smooth(k / 4), ease_out(k / 13)
    elif f < B_HOLD2:
        s.run, s.p, s.running, s.a, s.pulse = 'adapter', (f - B_BEAT2) / (N_DEN - 1), True, 1.0, True
    elif f < B_RESET:
        k = f - B_HOLD2
        s.run, s.p, s.show_result, s.a = 'adapter', 1.0, True, 1.0
        s.fade_noise, s.fade_result, s.caption, s.glow = 0.0, 1.0, smooth(k / 6), smooth(k / 8)
    else:
        k = f - B_RESET
        s.run, s.p, s.show_result = 'adapter', 1.0, k < 4
        s.fade_result, s.fade_noise = 1 - smooth(k / 4), smooth((k - 3) / 3)
        s.caption = s.glow = 1 - smooth(k / 4)
        s.a = 1 - ease_in(k / 13)
    if s.running:
        s.fade_noise, s.fade_result = 0.0, 1.0
    return s


# ------------------------------------------------------------------------------ data
def step_clock(run, steps):
    """Which recorded step a beat shows at fraction p of its frames. Frames are spent where the model is changing
    things: the pace follows the number of flips per step (smoothed), with a floor so that quiet stretches are
    fast-forwarded rather than frozen. The step counter shows the true time of the step on screen."""
    X, B = run['X'], run['bonds']
    act = np.zeros(steps)
    for s in range(steps):
        a, b = {(i, j): c for i, j, c in B[s]}, {(i, j): c for i, j, c in B[s + 1]}
        act[s] = sum(x != y for x, y in zip(X[s], X[s + 1])) + sum(a.get(q, 0) != b.get(q, 0) for q in set(a) | set(b))
    smooth_act = np.convolve(np.pad(act, 6, mode='edge'), np.ones(13) / 13, mode='valid')
    w = 0.3 + smooth_act / max(smooth_act.max(), 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    cum /= cum[-1]
    return lambda p: int(round(float(np.interp(p, cum, np.arange(steps + 1)))))


def load():
    with open(DATA) as fh:
        d = json.load(fh)
    atom_types = d['meta']['atom_types']
    n = d['meta']['n']
    slots = scatter_layout(n, NOISE_W, NOISE_H, seed=7, min_dist=9.5, cx=X_NOISE, cy=MID)
    finals = {}
    for kind, run in d['runs'].items():
        Xf, bf = run['X'][-1], run['bonds'][-1]
        unit = molecule_layout(Xf, bf, atom_types)
        finals[kind], _ = fit_layout(unit, SAMPLE_W, SAMPLE_H, BOND_MAX, X_SAMPLE, MID)
    shift = X_SAMPLE - X_NOISE
    noise_pos = assign_slots(slots, [[(x - shift, y) for x, y in f] for f in finals.values()])
    d['clock'] = {kind: step_clock(run, d['meta']['steps']) for kind, run in d['runs'].items()}
    return d, atom_types, noise_pos, finals


# ------------------------------------------------------------------------------ parts
def frozen_model(fig, x, y, w, h, label, taper=TAPER, size=7.5):
    """Frozen model: white trapezoid (narrow side right) with broad light blue diagonal stripes."""
    t = h * taper / 2
    pts = [(x, y), (x + w, y + t), (x + w, y + h - t), (x, y + h)]
    fig.path(pts, closed=True, fill=WHITE, stroke='none', effects=[shadow()], name='frozen body')
    poly = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
    lines = ''.join(f'<line x1="{x + k:.2f}" y1="{y + h:.2f}" x2="{x + k + h:.2f}" y2="{y:.2f}" '
                    f'stroke="{FROST}" stroke-width="{FROST_SW}"/>'
                    for k in [i * FROST_GAP - h for i in range(int((w + h) / FROST_GAP) + 2)])
    content = f'<defs><clipPath id="fc"><polygon points="{poly}"/></clipPath></defs><g clip-path="url(#fc)">{lines}</g>'
    fig.place(_svg_file('frost', x, y, w, h, content), x, y, w, h, name='frozen stripes')
    fig.path(pts, closed=True, fill='none', stroke=BLACK, sw=STYLE['outline'], name='frozen outline')
    fig.text(x + w / 2 - 2, y + h / 2 + size * 0.36, label, size=size, font='rcm', justify='center')


def model_top_y(x):
    """y of the frozen model's slanted top edge at x."""
    t = MH * TAPER / 2
    return (MID - MH / 2) + t * (x - X_MODEL) / MW


def loop_point(u):
    """Point at fraction u (0..1) along the feedback loop: down from the output arrow, left under the model, up into
    the input line."""
    legs = [((X_LOOP_OUT, MID), (X_LOOP_OUT, LOOP_Y)), ((X_LOOP_OUT, LOOP_Y), (X_LOOP_IN, LOOP_Y)),
            ((X_LOOP_IN, LOOP_Y), (X_LOOP_IN, MID + 2))]
    lens = [math.dist(a, b) for a, b in legs]
    d = u * sum(lens)
    for (a, b), L in zip(legs, lens):
        if d <= L:
            k = d / L
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        d -= L
    return legs[-1][1]


def adapter(fig, a):
    """The flame adapter docking on the frozen model: slides down from above while fading in. Its label (a water
    drop for the property, logP* = 4.5 for the target) appears once it has docked."""
    if a < 0.02:
        return
    x = ADAPTER_X
    dock = model_top_y(x) - DOCK_GAP - ADAPTER_H           # top of the adapter when docked
    y = dock - 15 * (1 - a)
    fig.model(x - ADAPTER_W / 2, y, ADAPTER_W, ADAPTER_H, '', taper=0.38, direction='down', fill=ADAPTER_FILL,
              name='adapter')
    fig.emoji('🔥', x, y + 0.45 * ADAPTER_H, size=10)
    fig.line(x, y + ADAPTER_H, x, model_top_y(x) + 1.5 * (1 - a), sw=1.1, name='adapter into base')
    if a > 0.85:
        k = smooth((a - 0.85) / 0.15)
        parts = [('logP', 'mr'), ('∗', 'mm', 'sup'), SP, ('=', 'mm'), SP, ('4.5', 'mr')]
        _, w = fig._measure(parts, 8)
        drop, gap = 9.0, 2.5
        x0 = x - (drop + gap + w) / 2
        base = dock - 4.5
        fig.rich(x0 + drop + gap, base, parts, size=8, color=tint(BLACK, 1 - k))
        if k > 0.5:
            fig.emoji('💧', x0 + drop / 2, base - 3.1, size=drop)


def pulses(fig, frame, active):
    """Orange ripples that leave the docking point into the model: the condition is injected at every step. One
    ripple per round of the feedback loop, starting when the token re-enters the model, with a soft flash at the
    docking point."""
    if not active:
        return
    cx, cy = ADAPTER_X, model_top_y(ADAPTER_X)
    phase = ((frame % PERIOD) / PERIOD - 0.85) % 1.0
    flash = 1 - phase / 0.4
    if flash > 0.02:
        fig.ellipse(cx, cy + 1, 6.5, fill=ORANGE, stroke='none', opacity=pct(70 * flash), effects=[blur(3.2)],
                    name='pulse flash')
    for lag in (0.0, 0.2):
        k = phase - lag
        if k < 0 or k > 0.78:
            continue
        k /= 0.78
        fig.ellipse(cx, cy, 3.0 + 12.0 * k, fill='none', stroke=ORANGE, sw=2.0 - 1.1 * k,
                    opacity=pct(100 * (1 - k) ** 1.2), name='pulse')


def sample_slot(fig, tl, d, atom_types, noise_pos, finals):
    """The evolving sample: noise (copied from the left), the half-denoised state, or the finished molecule."""
    dx = X_SAMPLE - X_NOISE
    sample_noise = [(x + dx, y) for x, y in noise_pos]
    X0, b0 = d['noise']['X'], d['noise']['bonds']
    run = d['runs'][tl.run] if tl.run else None
    if tl.running:
        s = d['clock'][tl.run](tl.p)
        q = min(tl.p / 0.92, 1.0)
        morph = 0.5 * q + 0.5 * smooth(q)
        pos = keep_apart(lerp_points(sample_noise, finals[tl.run], morph), 7.8)
        draw_state(fig, pos, run['X'][s], run['bonds'][s], atom_types, name='sample')
        return
    if tl.fade_noise > 0.01:
        draw_state(fig, sample_noise, X0, b0, atom_types, fade=tl.fade_noise, name='sample noise')
    if tl.fade_result > 0.01 and run is not None:
        draw_state(fig, finals[tl.run], run['X'][-1], run['bonds'][-1], atom_types, fade=tl.fade_result,
                   glow=0.8 * tl.glow, name='sample result')


# ------------------------------------------------------------------------------ the frame
def build(fig):
    d, atom_types, noise_pos, finals = load()
    tl = timeline(FRAME)

    # left: wordmark and tagline
    fig.text(LEFT, WM_BASE, 'defog-adapt', size=30, font='rcm')
    fig.text(LEFT + 1, TAG_BASE, 'Steer a frozen graph generator', size=9.5, color=GREY)

    # the noise, the same in both beats
    draw_state(fig, noise_pos, d['noise']['X'], d['noise']['bonds'], atom_types, name='noise')
    fig.text(X_NOISE, LABEL, 'Noise', size=7, justify='center')

    # the model is applied again and again: a feedback loop under it, a token running round it while it works
    fig.arrow([(X_LOOP_OUT, MID), (X_LOOP_OUT, LOOP_Y), (X_LOOP_IN, LOOP_Y), (X_LOOP_IN, MID + 2.5)], sw=0.9,
              head=3.0, color=GREY, name='feedback loop')
    if tl.running:
        u = (FRAME % PERIOD) / PERIOD
        for k in range(4):
            px, py = loop_point((u - k * 0.045) % 1.0)
            fig.ellipse(px, py, 2.1 - 0.4 * k, fill=BLACK if k == 0 else GREY, stroke='none',
                        opacity=pct(100 - 22 * k), name='token')
    shown = d['clock'][tl.run](tl.p) if tl.run else 0
    t_val = d['meta']['flow_t'][shown] if (tl.running or (tl.show_result and tl.fade_noise < 0.5)) else 0.0
    fig.rich((X_LOOP_OUT + X_LOOP_IN) / 2, LABEL, [('t', 'mi'), SP, ('=', 'mm'), SP, (f'{t_val:.2f}', 'mr')], size=7.5,
             color=DGREY, justify='center')

    fig.line(X_NOISE + NOISE_W / 2 + 4, MID, X_MODEL, MID, sw=STYLE['arrow_sw'], name='noise into model')
    frozen_model(fig, X_MODEL, MID - MH / 2, MW, MH, 'DeFoG')
    fig.emoji('❄️', X_MODEL + MW - 3.5, MID - MH / 2 + MH * TAPER / 2 + 3, size=11)
    pulses(fig, FRAME, tl.pulse)
    adapter(fig, tl.a)
    fig.arrow([(X_MODEL + MW + 3, MID), (X_OUT, MID)], name='model to sample')

    # the evolving sample and its captions
    sample_slot(fig, tl, d, atom_types, noise_pos, finals)
    fig.text(X_SAMPLE, LABEL, 'Sample', size=7, justify='center')
    if tl.caption > 0.02 and tl.run:
        fig.rich(X_SAMPLE, CAPTION, [('logP', 'mr'), SP, ('=', 'mm'), SP, (f'{d["runs"][tl.run]["logp"]:.1f}'.replace('-', '\u2212'), 'mr')],
                 size=7.5, color=tint(DGREY, 1 - tl.caption), justify='center')
