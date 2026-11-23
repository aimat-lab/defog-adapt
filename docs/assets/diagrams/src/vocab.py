"""The shared drawing vocabulary of the documentation diagrams.

It is the README banner's (docs/assets/banner/banner.py), so that the two look alike. A frozen model is a white trapezoid
with pale blue stripes and a snowflake badge; the adapter is an orange block with a flame; a model whose weights change
carries a delta-W badge. Feynman-Kac steering is cyan (a process acting on the candidates) and RL fine-tuning purple.
Molecules are small hand-made graphs: they illustrate, they are not data.

The drawing calls are figlib's, the author's figure toolchain (found through $FIGURE_SKILL, not published); see
README.md in the folder above.
"""
import itertools
import math
import random

from figlib import *  # noqa: F401,F403
from figlib import _svg_file

# colours that carry a meaning in these figures
FROST, FROST_SW, FROST_GAP = '#b9dcff', 2.0, 5.0       # light blue stripes: frozen weights (as in the banner)
ICE_D = '#2275c9'                                      # the frozen model's blue, deeper than the stripes
ADAPTER_FILL = tint(ORANGE, 0.78)                      # the adapter (flame)
FK_FILL = tint(CYAN, 0.78)                             # Feynman-Kac steering: a process acting on the candidates
RL_FILL = tint(PURPLE, 0.78)                           # RL fine-tuning
FK_LINE = CYAN                                         # a cloned lineage
# element colours of the small molecules (as in the thesis figures): carbon grey, nitrogen blue, oxygen red, sulphur yellow
ELEMENT = {'C': C_FILL, 'N': '#3a5ee0', 'O': '#e5303c', 'S': '#d6c100'}

_frost_ids = itertools.count()


def background(fig, radius=6):
    """A flat white card behind the whole figure. The tool's SVG is transparent, and the figures are drawn in black, so on the
    dark theme of the site they would vanish; on a card they read on either."""
    fig.rect(0, 0, fig.w, fig.h, radius=radius, fill=WHITE, stroke='none', name='background')


def snowflake(fig, cx, cy, r, color=ICE_D, sw=1.05, name='snowflake'):
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


def model_edges(x, y, w, h, taper=0.38):
    """The four corners of a model trapezoid (narrow side right) and the height the slanted ends lose."""
    t = h * taper / 2
    return [(x, y), (x + w, y + t), (x + w, y + h - t), (x, y + h)], t


def model_top_y(x_at, x, y, w, h, taper=0.38):
    """y of a model's slanted top edge at x_at."""
    return y + h * taper / 2 * (x_at - x) / w


def frozen_model(fig, x, y, w, h, label='', taper=0.38, size=7.5, badge_r=6.6, snow_r=4.3, name='frozen model'):
    """Frozen model: white trapezoid (narrow side right) with broad light blue diagonal stripes, and a badge with a
    snowflake on its top corner."""
    pts, t = model_edges(x, y, w, h, taper)
    fig.path(pts, closed=True, fill=WHITE, stroke='none', effects=[shadow()], name=f'{name} body')
    poly = ' '.join(f'{px:.2f},{py:.2f}' for px, py in pts)
    cid = f'fc{next(_frost_ids)}'
    lines = ''.join(f'<line x1="{x + k:.2f}" y1="{y + h:.2f}" x2="{x + k + h:.2f}" y2="{y:.2f}" '
                    f'stroke="{FROST}" stroke-width="{FROST_SW}"/>'
                    for k in [i * FROST_GAP - h for i in range(int((w + h) / FROST_GAP) + 2)])
    content = f'<defs><clipPath id="{cid}"><polygon points="{poly}"/></clipPath></defs><g clip-path="url(#{cid})">{lines}</g>'
    fig.place(_svg_file(f'frost{cid}', x, y, w, h, content), x, y, w, h, name=f'{name} stripes')
    fig.path(pts, closed=True, fill='none', stroke=BLACK, sw=STYLE['outline'], name=f'{name} outline')
    if label:
        fig.text(x + w / 2 - 2, y + h / 2 + size * 0.36, label, size=size, font='rcm', justify='center')
    bx, by = x + w - badge_r + 1.2, y + t + 0.6
    fig.ellipse(bx, by, badge_r, fill=WHITE, stroke=ICE_D, sw=0.9, effects=[shadow(0.6, opacity=30)],
                name=f'{name} badge')
    snowflake(fig, bx, by, snow_r, ICE_D, 0.95)


def frozen_panel(fig, x, y, w, h, badge_r=6.6, snow_r=4.3, name='frozen base'):
    """A frozen container (a rectangle with the same pale blue stripes and a snowflake badge on its top corner), for a
    stack of layers that are drawn on top of it."""
    fig.rect(x, y, w, h, radius=0, fill=WHITE, stroke='none', effects=[shadow()], name=f'{name} body')
    poly = f'{x:.2f},{y:.2f} {x + w:.2f},{y:.2f} {x + w:.2f},{y + h:.2f} {x:.2f},{y + h:.2f}'
    cid = f'fc{next(_frost_ids)}'
    lines = ''.join(f'<line x1="{x + k:.2f}" y1="{y + h:.2f}" x2="{x + k + h:.2f}" y2="{y:.2f}" '
                    f'stroke="{FROST}" stroke-width="{FROST_SW}"/>'
                    for k in [i * FROST_GAP - h for i in range(int((w + h) / FROST_GAP) + 2)])
    content = f'<defs><clipPath id="{cid}"><polygon points="{poly}"/></clipPath></defs><g clip-path="url(#{cid})">{lines}</g>'
    fig.place(_svg_file(f'frost{cid}', x, y, w, h, content), x, y, w, h, name=f'{name} stripes')
    fig.rect(x, y, w, h, radius=0, fill='none', stroke=BLACK, sw=STYLE['outline'], name=f'{name} outline')
    return x + w - badge_r + 1.2, y + 1.0                           # where the snowflake badge goes (drawn last by the caller)


def frozen_badge(fig, bx, by, badge_r=6.6, snow_r=4.3, name='frozen badge'):
    fig.ellipse(bx, by, badge_r, fill=WHITE, stroke=ICE_D, sw=0.9, effects=[shadow(0.6, opacity=30)], name=name)
    snowflake(fig, bx, by, snow_r, ICE_D, 0.95)


def live_model(fig, x, y, w, h, fill, label='', taper=0.38, size=7.5, delta=True, badge_r=6.0, name='model'):
    """A model whose weights change: a tinted trapezoid with a delta-W badge on its top corner (or none, once trained)."""
    fig.model(x, y, w, h, label, taper=taper, fill=fill, size=size, font='rcm', name=name)
    if delta:
        t = h * taper / 2
        bx, by = x + w - badge_r + 1.2, y + t + 0.6
        fig.badge(bx, by, badge_r, [('Δ', 'mm'), ('W', 'mi')], size=5.4)


def adapter_block(fig, cx, top, w=24, h=15, flame=9.5, name='adapter'):
    """The flame adapter: an orange block that narrows downwards, docking on a model from above."""
    fig.model(cx - w / 2, top, w, h, '', taper=0.38, direction='down', fill=ADAPTER_FILL, name=name)
    fig.emoji('🔥', cx, top + 0.45 * h, size=flame)


# ----------------------------------------------------------------------------------------------- small molecules
def _chain(start, direction, elements, b, zig=1):
    """Atoms leaving `start` along `direction` (radians), one bond apart, zigzagging a little."""
    out, (x, y), a = [], start, direction
    for k, el in enumerate(elements):
        a2 = a + zig * math.radians(30) * (1 if k % 2 == 0 else -1)
        x, y = x + b * math.cos(a2), y - b * math.sin(a2)
        out.append(((x, y), el))
    return out


def molecule(kind, cx, cy, b=6.5):
    """A small hand-made molecule centred on (cx, cy): (pos, bonds, fills). `kind` picks one of a few shapes, so that
    no two slots in a figure need show the same drawing."""
    pos, els, bonds = [], [], []

    def add_ring(c, n, start=0, hetero=None):
        corners = ring(c[0], c[1], b, n, start)
        base = len(pos)
        for k, p in enumerate(corners):
            pos.append(p)
            els.append((hetero or {}).get(k, 'C'))
        bonds.extend((base + k, base + (k + 1) % n) for k in range(n))
        return base, corners

    def add_chain(from_idx, elements, direction, zig=1):
        prev = from_idx
        for (p, el) in _chain(pos[from_idx], direction, elements, b, zig):
            pos.append(p)
            els.append(el)
            bonds.append((prev, len(pos) - 1))
            prev = len(pos) - 1

    shapes = {
        'benzene': lambda: add_ring((0, 0), 6),
        'phenol': lambda: (add_ring((0, 0), 6, 0), add_chain(0, ['O'], 0)),
        'toluene': lambda: (add_ring((0, 0), 6, 0), add_chain(3, ['C'], math.pi)),
        'pyridine': lambda: (add_ring((0, 0), 6, 30, {2: 'N'}), add_chain(0, ['C', 'O'], math.radians(30))),
        'furan': lambda: (add_ring((0, 0), 5, 90, {0: 'O'}), add_chain(2, ['C', 'N'], math.radians(220))),
        'amide': lambda: (add_chain_from_scratch(pos, els, bonds, b, ['C', 'C', 'O', 'N'], branch=2)),
        'naphthalene': lambda: _fused(pos, els, bonds, b),
        'ether': lambda: (add_ring((0, 0), 6, 0, {3: 'O'}), add_chain(0, ['C', 'C'], 0)),
        'thiophene': lambda: (add_ring((0, 0), 5, 90, {0: 'S'}), add_chain(1, ['N'], math.radians(150))),
        'chain': lambda: add_chain_from_scratch(pos, els, bonds, b, ['C', 'C', 'C', 'O', 'C'], branch=None),
    }
    shapes[kind]()
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    ox, oy = cx - (min(xs) + max(xs)) / 2, cy - (min(ys) + max(ys)) / 2
    return [(x + ox, y + oy) for x, y in pos], bonds, [ELEMENT[e] for e in els]


def add_chain_from_scratch(pos, els, bonds, b, elements, branch=None):
    """A zigzag chain; `branch` is the index of an atom that carries one extra atom upwards."""
    x, y = 0.0, 0.0
    for k, el in enumerate(elements):
        pos.append((x, y))
        els.append(el)
        if k:
            bonds.append((k - 1, k))
        x += b * math.cos(math.radians(30))
        y += b * math.sin(math.radians(30)) * (1 if k % 2 == 0 else -1)
    if branch is not None:
        bx, by = pos[branch]
        pos.append((bx, by - b))
        els.append('O')
        bonds.append((branch, len(pos) - 1))


def _fused(pos, els, bonds, b):
    """Two six-rings sharing a bond."""
    h = b * math.sqrt(3) / 2
    for cx in (-h, h):
        base = len(pos)
        corners = ring(cx, 0, b, 6, 30)
        for p in corners:
            pos.append(p)
            els.append('C')
        bonds.extend((base + k, base + (k + 1) % 6) for k in range(6))
    # the two rings overlap on one bond: merge the duplicated atoms
    keep, remap = [], {}
    for i, p in enumerate(pos):
        for j, q in enumerate(keep):
            if abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01:
                remap[i] = j
                break
        else:
            remap[i] = len(keep)
            keep.append(p)
    els2 = [None] * len(keep)
    for i, e in enumerate(els):
        els2[remap[i]] = e
    new_bonds = sorted({tuple(sorted((remap[a], remap[c]))) for a, c in bonds})
    pos[:] = keep
    els[:] = els2
    bonds[:] = new_bonds


def draw_molecule(fig, kind, cx, cy, b=6.5, r=2.4, opacity=None, name='molecule', glows=()):
    pos, bonds, fills = molecule(kind, cx, cy, b)
    return fig.graph(pos, bonds, fills, r, bond_w=1.0, node_w=0.7, opacity=opacity, glows=glows, name=name)


def noise(cx, cy, w, h, n=7, seed=1, b=6.5):
    """A scatter of atoms and a few random bonds: what the sampler starts from. (pos, bonds, fills)."""
    rnd = random.Random(seed)
    pts = []
    while len(pts) < n:
        p = (cx + (rnd.random() - 0.5) * w, cy + (rnd.random() - 0.5) * h)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > (0.85 * b) ** 2 for q in pts):
            pts.append(p)
    bonds = [(i, j) for i in range(n) for j in range(i + 1, n)
             if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 < (1.7 * b) ** 2 and rnd.random() < 0.55]
    fills = [ELEMENT[rnd.choice('CCCNO')] for _ in range(n)]
    return pts, bonds, fills


def draw_noise(fig, cx, cy, w=26, h=24, n=7, seed=1, r=2.3, name='noise'):
    pos, bonds, fills = noise(cx, cy, w, h, n, seed)
    return fig.graph(pos, bonds, fills, r, bond_w=1.0, node_w=0.7, opacity=55, name=name)


def frozen_symbol():
    """Legend symbol: the snowflake badge."""
    def draw(fig, x, y):
        fig.ellipse(x + 5.5, y, 5.5, fill=WHITE, stroke=ICE_D, sw=0.9, effects=[shadow(0.5, opacity=30)])
        snowflake(fig, x + 5.5, y, 3.6, ICE_D, 0.85)
    return draw, 11


def delta_symbol():
    """Legend symbol: the delta-W badge."""
    def draw(fig, x, y):
        fig.badge(x + 5.5, y, 5.5, [('Δ', 'mm'), ('W', 'mi')], size=5.0)
    return draw, 11
