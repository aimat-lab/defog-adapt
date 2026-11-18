"""Contact sheet of candidate pairs for the banner: the same noise denoised by the frozen base alone (left) and by
the base with a logP adapter (right), drawn exactly as the banner draws molecules.

Reads ``candidates.json`` and ``states.npz`` from the folder in $BANNER_WORK (written by record_trajectories.py
record). Build:  BANNER_WORK=... python3 ~/.claude/skills/figure/scripts/build.py candidates.py
"""
import json
import os

import numpy as np
from figlib import *  # noqa: F401,F403
from molecules import fit_layout, molecule_layout, draw_state

NAME = 'candidates'
COLS, ROWS = 2, 6
CELL_W, CELL_H = 228, 62
SIZE = (468, 18 + ROWS * CELL_H + 4)

MOL_W, MOL_H, BOND_MAX = 88, 40, 12.0
X_BASE, X_ARROW0, X_ARROW1, X_ADAPT = 56, 106, 126, 176     # inside a cell


def final_state(states, kind, run):
    X = states[f'{kind}_X'][run][-1]
    E = states[f'{kind}_E'][run][-1]
    bonds = [(i, j, int(E[i, j])) for i in range(len(X)) for j in range(i + 1, len(X)) if E[i, j]]
    return X, bonds


def build(fig):
    work = os.environ['BANNER_WORK']
    with open(os.path.join(work, 'candidates.json')) as fh:
        data = json.load(fh)
    states = np.load(os.path.join(work, 'states.npz'))
    atom_types = data['meta']['atom_types']
    fig.text(6, 9, 'Same noise: frozen base alone (left) and with a logP adapter (right, target %.1f). '
                   'Ranked automatically; pick by number.' % data['meta']['target'], size=7, color=GREY)
    for rank, cand in enumerate(data['candidates'][:COLS * ROWS]):
        col, row = rank % COLS, rank // COLS
        x0, y0 = 6 + col * (CELL_W + 4), 18 + row * CELL_H
        cy = y0 + 25
        fig.text(x0, y0 + 8, f'#{rank}', size=7.5, font='rcm')
        for kind, cx in (('base', x0 + X_BASE), ('adapter', x0 + X_ADAPT)):
            X, bonds = final_state(states, kind, cand['run'])
            unit = molecule_layout(X, bonds, atom_types)
            pos, _ = fit_layout(unit, MOL_W, MOL_H, BOND_MAX, cx, cy)
            draw_state(fig, pos, X, bonds, atom_types, name=f'{kind} {rank}')
            fig.text(cx, y0 + 57, f'logP = {cand[kind]["logp"]:.1f}', size=6.5, color=GREY, justify='center')
        fig.arrow([(x0 + X_ARROW0, cy), (x0 + X_ARROW1, cy)], sw=0.9, head=3.0, name='adapters')
