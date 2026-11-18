"""README banner, one frame: wordmark on the left, and on the right a REAL recorded run of the frozen DeFoG base.

The animation has two beats on the SAME noise. Beat 1: the frozen base alone turns the noise into a molecule.
Beat 2: a flame-marked logP adapter docks on top of the frozen model, and the same noise ends in a molecule at the
requested logP. Every atom and bond flip is a flip the model made (trajectory.json is the recorded sampler state, see
record_trajectories.py); only the node POSITIONS are a drawing choice, because a graph has no coordinates: the noise is
scattered and the final molecule uses an RDKit 2D layout, and the drawing glides from one to the other while the
categories flip.

While a beat runs, small pulses travel from the noise, disappear behind the model and come out of it along the arrow to
the sample; one pulse stands for one application of the model. Where the adapter is docked, each pulse passes under it,
the adapter's ring fires and the pulse comes out of the model orange. A faint time axis under the stations runs from the noise (t = 0) to the
sample (t = 1); its knob carries the flow time the model was given at the step on screen, and it rewinds when the
noise comes back.

This script draws ONE frame, chosen by $BANNER_FRAME; render_banner.py renders them all and assembles the GIF.

Build one frame:  BANNER_FRAME=120 python3 ~/.claude/skills/figure/scripts/build.py banner.py --scale 4
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
from figlib import *  # noqa: F401,F403
from figlib import _svg_file, fonts
from molecules import (assign_slots, draw_state, fit_layout, keep_apart, lerp_points, molecule_layout, pct,
                       scatter_layout)

NAME = 'banner'
SIZE = (468, 126)                  # about 3.7:1; the type sizes are those of the thesis figures (7 pt labels)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('BANNER_DATA', os.path.join(HERE, 'trajectory.json'))
FRAME = int(os.environ.get('BANNER_FRAME', '120'))

# ------------------------------------------------------------------------------ layout (pt)
MID = 58                           # centre line of the stations
LEFT = 14                          # left margin of the wordmark
WM_SIZE, WM_BASE, TAG_BASE = 30, 61, 77          # wordmark size and baseline, tagline baseline
X_NOISE, NOISE_W, NOISE_H = 222, 52, 48          # noise graph: centre and extent of its scatter
X_MODEL, MW, MH, TAPER = 270, 60, 46, 0.38       # frozen DeFoG: left edge, width, height
X_SAMPLE, SAMPLE_W, SAMPLE_H, BOND_MAX = 405, 74, 40, 11.0
X_OUT = 353                        # where the arrow from the model ends
LABEL = 96                         # baseline of the station labels
AXIS_Y, TICKS = 110, 10            # the time axis under the stations, and its ticks (every 0.1)
T_BASE = 120.5                     # baseline of the t counter that rides under the knob
ADAPTER_X = X_MODEL + 0.36 * MW    # where the adapter docks, along the model's top edge
ADAPTER_W, ADAPTER_H, DOCK_GAP = 26, 17, 5
FLOW_X0, FLOW_X1 = X_NOISE + NOISE_W / 2 + 4, X_OUT      # the path of the pulses: line into the model, model, arrow
FLOW_TAU = (0.0, 0.30, 0.55, 1.0)                        # time fractions at which a pulse reaches the model, leaves it

FROST, FROST_SW, FROST_GAP = '#b9dcff', 2.0, 5.0       # light blue stripes: frozen weights (as in the visual abstract)
ICE_D = '#2275c9'                  # the frozen model's blue, deeper than the pale stripes so that a snowflake can be seen
ADAPTER_FILL = tint(ORANGE, 0.78)                      # the trained adapter's colour (flame)
PULSE = '#3d3d3d'                                      # a pulse of the model working (orange once the adapter acted)
AX_TRACK, AX_TICK_OFF, AX_PASSED = '#ececec', '#dedede', '#a3a3a3'     # the faint time axis
AX_KNOB, AX_TEXT = '#8c8c8c', '#808080'

# ------------------------------------------------------------------------------ timeline (15 fps)
SLOW = 1.3                         # the first draft's pace divided by this: 30 % slower


def fr(n):
    """A duration of the first draft (in frames) at the current pace."""
    return round(n * SLOW)


N_DEN = fr(50)                                         # frames of visible denoising per beat
T_INTRO, T_HOLD1, T_SWAP, T_HOLD2, T_RESET = fr(8), fr(22), fr(22), fr(28), fr(14)
B_BEAT1 = T_INTRO
B_HOLD1 = B_BEAT1 + N_DEN
B_SWAP = B_HOLD1 + T_HOLD1
B_BEAT2 = B_SWAP + T_SWAP
B_HOLD2 = B_BEAT2 + N_DEN
B_RESET = B_HOLD2 + T_HOLD2
FRAMES = B_RESET + T_RESET
REWIND = (3, 12)                                       # first-draft frames (after a result is hidden) of the knob's rewind
PULSE_PERIOD, PULSE_TRAVEL, PULSE_LIFE = fr(8), fr(10), fr(8)    # a pulse starts, takes, and the adapter's ring lives


def smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3 - 2 * k)


def ease_out(k):
    return 1 - (1 - min(max(k, 0.0), 1.0)) ** 3


def ease_in(k):
    return min(max(k, 0.0), 1.0) ** 3


def mix(c0, c1, k):
    """Colour between two '#rrggbb' colours."""
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * k):02x}' for x, y in zip(a, b))


def timeline(f):
    """What is on screen at frame f. The last frame equals the first, so the loop is seamless. Times inside a phase
    (k) are in frames of the first draft, so that the pace is set in one place (SLOW)."""
    s = SimpleNamespace(run=None, p=0.0, fade_noise=1.0, fade_result=0.0, a=0.0, caption=0.0, glow=0.0,
                        running=False, show_result=False)
    if f < B_BEAT1:
        pass
    elif f < B_HOLD1:
        s.run, s.p, s.running = 'base', (f - B_BEAT1) / (N_DEN - 1), True
    elif f < B_SWAP:
        k = (f - B_HOLD1) / SLOW
        s.run, s.p, s.show_result = 'base', 1.0, True
        s.fade_noise, s.fade_result, s.caption = 0.0, 1.0, smooth(k / 6)
    elif f < B_BEAT2:
        k = (f - B_SWAP) / SLOW                  # the result fades out, then the noise fades in (little overlap)
        s.run, s.p, s.show_result = 'base', 1.0, k < 4
        s.fade_result, s.fade_noise = 1 - smooth(k / 4), smooth((k - 3) / 3)
        s.caption, s.a = 1 - smooth(k / 4), ease_out(k / 13)
    elif f < B_HOLD2:
        s.run, s.p, s.running, s.a = 'adapter', (f - B_BEAT2) / (N_DEN - 1), True, 1.0
    elif f < B_RESET:
        k = (f - B_HOLD2) / SLOW
        s.run, s.p, s.show_result, s.a = 'adapter', 1.0, True, 1.0
        s.fade_noise, s.fade_result, s.caption, s.glow = 0.0, 1.0, smooth(k / 6), smooth(k / 8)
    else:
        k = (f - B_RESET) / SLOW
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


def counter_time(f, tl, d):
    """The flow time t the counter and the knob show. While a beat runs it is the time of the step on screen; in a
    hold it is 1; after a result is hidden the knob rewinds to 0 (so that the noise starts again from t = 0)."""
    if tl.running:
        return d['meta']['flow_t'][d['clock'][tl.run](tl.p)]
    if f < B_BEAT1:
        return 0.0
    if f < B_SWAP or B_HOLD2 <= f < B_RESET:
        return 1.0
    k = (f - (B_SWAP if f < B_BEAT2 else B_RESET)) / SLOW
    return 1.0 - smooth((k - REWIND[0]) / (REWIND[1] - REWIND[0]))


def pulse_starts():
    """First frames of the pulses of a beat: one every PULSE_PERIOD, none so late that the end of the beat would cut
    it off while it is still running."""
    return range(0, N_DEN - PULSE_TRAVEL, PULSE_PERIOD)


def flow_x(tau):
    """x of a pulse at time fraction tau (0..1) of its run. It lingers where it can be seen (the line into the model,
    the arrow) and zips through the stretch behind the model, whose body hides it."""
    return float(np.interp(tau, FLOW_TAU, (FLOW_X0, X_MODEL, X_MODEL + MW, FLOW_X1)))


def flow_pulses(i):
    """Time fraction (0..1 of its run) of each pulse in flight at frame i of a beat."""
    return [(i - s) / PULSE_TRAVEL for s in pulse_starts() if 0 <= i - s <= PULSE_TRAVEL]


def ring_ages(i):
    """Ages (frames) of the adapter's rings at frame i of beat 2: a ring fires when a pulse passes under the dock."""
    tau_dock = float(np.interp(ADAPTER_X, (FLOW_X0, X_MODEL, X_MODEL + MW, FLOW_X1), FLOW_TAU))
    ages = [i - (s + tau_dock * PULSE_TRAVEL) for s in pulse_starts()]
    return [a for a in ages if 0 <= a < PULSE_LIFE]


# ------------------------------------------------------------------------------ parts
def tracked(fig, x, y, s, size, font, color, tracking):
    """Text with letter-spacing (figlib has none): glyph by glyph. Returns the end x."""
    for ch in s:
        fig.text(x, y, ch, size=size, font=font, color=color)
        x += fonts.text_width(ch, size, font) + tracking
    return x - tracking


def wordmark(fig):
    """'defog-adapt' with the hyphen drawn as a short orange bar (the adapter's colour), and the tagline in small
    letter-spaced capitals."""
    x = LEFT
    fig.text(x, WM_BASE, 'defog', size=WM_SIZE, font='rcm')
    x += fonts.text_width('defog', WM_SIZE, 'rcm') + 2.8
    fig.rect(x, WM_BASE - 0.31 * WM_SIZE - 1.5, 9, 3, radius=1.5, fill=ORANGE, stroke='none', name='hyphen')
    x += 9 + 2.8
    fig.text(x, WM_BASE, 'adapt', size=WM_SIZE, font='rcm')
    tracked(fig, LEFT + 1, TAG_BASE, 'STEER A FROZEN GRAPH GENERATOR', 6.6, 'rcm', GREY, 1.0)


def snowflake(fig, cx, cy, r, color, sw, name='snowflake'):
    """A six-armed snowflake with a V branch on each arm, drawn as round-capped lines."""
    for k in range(6):
        a = math.radians(90 + 60 * k)
        ux, uy = math.cos(a), -math.sin(a)
        fig.line(cx, cy, cx + r * ux, cy + r * uy, stroke=color, sw=sw, cap='round', name=name)
        bx, by = cx + 0.55 * r * ux, cy + 0.55 * r * uy
        for s in (-1, 1):
            b = a + s * math.radians(52)
            fig.line(bx, by, bx + 0.36 * r * math.cos(b), by - 0.36 * r * math.sin(b), stroke=color, sw=sw,
                     cap='round', name=name)


def frozen_model(fig, x, y, w, h, label, taper=TAPER, size=7.5):
    """Frozen model: white trapezoid (narrow side right) with broad light blue diagonal stripes, and a badge with a
    snowflake on its top corner (the pale stripes alone would hide a pale snowflake)."""
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
    bx, by = x + w - 6, y + t + 1
    fig.ellipse(bx, by, 8, fill=WHITE, stroke=ICE_D, sw=0.9, effects=[shadow(0.7, opacity=30)], name='frozen badge')
    snowflake(fig, bx, by, 5.4, ICE_D, 1.05)


def model_top_y(x):
    """y of the frozen model's slanted top edge at x."""
    t = MH * TAPER / 2
    return (MID - MH / 2) + t * (x - X_MODEL) / MW


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


def pulses(fig, ages):
    """Orange ripples at the docking point: the condition acts on each pulse that passes under the adapter. A soft
    flash first, then a ring that widens and fades."""
    cx, cy = ADAPTER_X, model_top_y(ADAPTER_X)
    for age in ages:
        k = age / PULSE_LIFE
        flash = 1 - age / (3 * SLOW)
        if flash > 0.02:
            fig.ellipse(cx, cy + 1, 6.5, fill=ORANGE, stroke='none', opacity=pct(70 * flash), effects=[blur(3.2)],
                        name='pulse flash')
        fig.ellipse(cx, cy, 3.0 + 12.0 * k, fill='none', stroke=ORANGE, sw=2.0 - 1.1 * k,
                    opacity=pct(100 * (1 - k) ** 1.2), name='pulse ring')


def draw_flow_pulses(fig, i, adapted, x_lo, x_hi):
    """Small pulses that run from the noise behind the model and out along the arrow to the sample: a dot with a short
    fading trail. Only the dots between x_lo and x_hi are drawn: the ones left of the model's right edge BEHIND the
    model (its body hides them while they are inside), the rest in front of the arrow. With the adapter docked a
    pulse turns orange as it passes under it."""
    for u in flow_pulses(i):
        head = flow_x(u)
        vis = smooth(u / 0.1) * (1 - smooth((u - 0.88) / 0.12))       # fades in at the start, out on arrival
        if vis < 0.03:
            continue
        for dist, r, op in ((0, 2.3, 100), (4.5, 1.8, 62), (9, 1.3, 36), (13.5, 0.9, 18)):
            x = head - dist
            if x < FLOW_X0 or not (x_lo <= x < x_hi):
                continue
            col = mix(PULSE, ORANGE, smooth((x - (ADAPTER_X - 2)) / 12)) if adapted else PULSE
            if dist == 0:
                fig.ellipse(x, MID, 5.0, fill=col, stroke='none', opacity=pct(22 * vis), effects=[blur(2.6)],
                            name='pulse halo')
            fig.ellipse(x, MID, r, fill=col, stroke='none', opacity=pct(op * vis), name='pulse')


def time_axis(fig, t):
    """The flow time as a faint scrubber under the stations: from the noise (t = 0) to the sample (t = 1). The knob
    carries the counter; the part the knob has passed is a shade darker."""
    x0, x1, y = X_NOISE, X_SAMPLE, AXIS_Y
    xt = x0 + (x1 - x0) * t
    fig.line(x0, y, x1, y, stroke=AX_TRACK, sw=1.1, cap='round', name='time track')
    if t > 0.004:
        fig.line(x0, y, xt, y, stroke=AX_PASSED, sw=1.1, cap='round', name='time passed')
    for k in range(1, TICKS):
        xk = x0 + (x1 - x0) * k / TICKS
        fig.line(xk, y - 1.4, xk, y + 1.4, stroke=AX_PASSED if t >= k / TICKS - 1e-9 else AX_TICK_OFF, sw=0.5,
                 name='tick')
    fig.ellipse(x0, y, 1.6, fill=AX_PASSED, stroke='none', name='t = 0')
    fig.ellipse(x1, y, 1.6, fill=AX_PASSED if t >= 0.999 else AX_TICK_OFF, stroke='none', name='t = 1')
    fig.ellipse(xt, y, 2.7, fill=WHITE, stroke=AX_KNOB, sw=0.9, effects=[shadow(0.5, opacity=22)], name='knob')
    fig.rich(xt, T_BASE, [('t', 'mi'), SP, ('=', 'mm'), SP, (f'{t:.2f}', 'mr')], size=7, color=AX_TEXT,
             justify='center')


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
    t = counter_time(FRAME, tl, d)
    i = FRAME - (B_BEAT1 if tl.run == 'base' else B_BEAT2)           # frame within the beat that is running
    adapted = tl.run == 'adapter'

    wordmark(fig)

    # the noise, the same in both beats
    draw_state(fig, noise_pos, d['noise']['X'], d['noise']['bonds'], atom_types, name='noise')
    fig.text(X_NOISE, LABEL, 'Noise', size=7, justify='center')

    time_axis(fig, t)

    fig.line(FLOW_X0, MID, X_MODEL, MID, sw=STYLE['arrow_sw'], name='noise into model')
    if tl.running:
        draw_flow_pulses(fig, i, adapted, -1e9, X_MODEL + MW)            # on the line, and BEHIND the model
    frozen_model(fig, X_MODEL, MID - MH / 2, MW, MH, 'DeFoG')
    if tl.running and adapted:
        pulses(fig, ring_ages(i))
    adapter(fig, tl.a)
    fig.arrow([(X_MODEL + MW + 3, MID), (X_OUT, MID)], name='model to sample')
    if tl.running:
        draw_flow_pulses(fig, i, adapted, X_MODEL + MW, 1e9)             # out of the model, on the arrow

    # the evolving sample and its captions
    sample_slot(fig, tl, d, atom_types, noise_pos, finals)
    fig.text(X_SAMPLE, LABEL, 'Sample', size=7, justify='center')
    if tl.caption > 0.02 and tl.run:
        top = min(y for _, y in finals[tl.run]) - 4.1                    # the caption sits just above its molecule
        value = f'{d["runs"][tl.run]["logp"]:.1f}'.replace('-', '−')
        fig.rich(X_SAMPLE, max(12.0, top - 7.5), [('logP', 'mr'), SP, ('=', 'mm'), SP, (value, 'mr')], size=7.5,
                 color=tint(DGREY, 1 - tl.caption), justify='center')
