#!/usr/bin/env python
"""Record REAL denoising trajectories for the README banner.

The banner shows one frozen DeFoG base denoising noise into a molecule, and then the same base with a logP
adapter denoising the SAME noise. Both are recorded here from the actual sampler, so every atom and bond flip
in the animation is a flip the model made.

This is not part of the package or its tests. It needs the local ZINC checkpoints, and it writes the data that
``banner.py`` draws. Run it in an environment with ``defog_adapt`` and RDKit; a GPU makes it fast.

    # 1. many same-noise pairs, ranked for legibility -> a candidate list to choose from
    python record_trajectories.py record --base BASE.ckpt --adapter LOGP_ADAPTER.ckpt --work WORK

    # 2. (optional) re-rank the stored runs after changing the scoring
    python record_trajectories.py rank --work WORK

    # 3. full trajectories of the chosen pair -> trajectory.json (what the figure reads)
    python record_trajectories.py export --work WORK --rank 0 --out trajectory.json

Settings of the recorded runs: 16 heavy atoms, 200 steps, polydec time distortion, eta 0 (calm: an atom or bond
that is right stays right), logP target 4.5 at the adapter's measured best guidance weight of 1.5, probability-
space product blending. Eta 0 and a fixed size are chosen so that the animation is readable; they are not the
benchmark configuration (500 steps, eta 25).

REPRODUCING a pair: sampling is deterministic given ``(seed, batch, index)`` and the same library, checkpoints
and hardware class, and those three numbers are stored in ``trajectory.json``. The pair was chosen from many
(a contact sheet of the ranked candidates), for legibility, not at random: the banner illustrates the process
with real samples, it is not a measurement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen

RDLogger.DisableLog("rdApp.*")

BOND_CLASSES = ["none", "single", "double", "triple"]
#: elements that read well in the animation (no phosphorus, no iodine, no charges)
LEGIBLE_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br"}


# ============================================================================ sampling with recording
def _recorder(sampler_cls):
    """Subclass of ``sampler_cls`` that keeps the discrete state (class index per node, per node pair) before
    the first step and after every step, for the whole batch."""

    class Recorded(sampler_cls):
        def _run_loop(self, X, E, y, node_mask, use_cfg, show_progress, on_step=None):
            self.X_states = [X.argmax(-1).cpu().numpy().astype(np.int8)]
            self.E_states = [E.argmax(-1).cpu().numpy().astype(np.int8)]
            return super()._run_loop(X, E, y, node_mask, use_cfg, show_progress, on_step=on_step)

        def _advance(self, t_int, X, E, y, node_mask, use_cfg):
            out = super()._advance(t_int, X, E, y, node_mask, use_cfg)
            self.X_states.append(out[0].argmax(-1).cpu().numpy().astype(np.int8))
            self.E_states.append(out[1].argmax(-1).cpu().numpy().astype(np.int8))
            return out

    return Recorded


def load_models(base_path, adapter_path, device):
    from defog_adapt.core import AdaLNAdapter, DeFoGModel

    base = DeFoGModel.load(base_path, device="cpu").to(device).eval()
    adapter = AdaLNAdapter.load(adapter_path, device=device).eval()
    adapter.check_compatible(base)
    return base, adapter


def sample_pair(base, adapter, seed, batch, n, steps, target, weight, device):
    """Two recorded runs of ``batch`` graphs each, from IDENTICAL initial noise (same seed, same size)."""
    from defog_adapt.core import AdaptedSampler, AdapterComposition, ConditionBranch, Sampler

    ctmc = dict(sample_steps=steps, eta=0.0, omega=0.0, time_distortion="polydec")
    runs = {}
    for kind in ("base", "adapter"):
        torch.manual_seed(seed)
        if kind == "base":
            sampler = _recorder(Sampler)(base, **ctmc)
        else:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([target]), weight)], base=base,
                                      mode="product")
            sampler = _recorder(AdaptedSampler)(base, comp, **ctmc)
        graphs = sampler.sample(batch, num_nodes=n, device=torch.device(device), show_progress=False)
        runs[kind] = dict(X=np.stack(sampler.X_states), E=np.stack(sampler.E_states), graphs=graphs)
    assert np.array_equal(runs["base"]["X"][0], runs["adapter"]["X"][0]) \
        and np.array_equal(runs["base"]["E"][0], runs["adapter"]["E"][0]), "the two runs must start from the same noise"
    return runs


# ============================================================================ legibility of a final molecule
def mol_from_state(X, E, atom_types):
    """RDKit molecule from a final discrete state (class per node, class per pair); None if it does not sanitise."""
    rw = Chem.RWMol()
    for c in X:
        rw.AddAtom(Chem.Atom(atom_types[int(c)]))
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    n = len(X)
    for i, j in itertools.combinations(range(n), 2):
        c = int(E[i, j])
        if c:
            rw.AddBond(i, j, order[c])
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:                                          # noqa: BLE001
        return None
    return m


def _segments_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return orient(p, q, r) * orient(p, q, s) < 0 and orient(r, s, p) * orient(r, s, q) < 0


def layout_ok(mol):
    """A 2D depiction with no overlapping atoms and no crossing bonds (so it can be drawn as ball and stick)."""
    AllChem.Compute2DCoords(mol)
    pts = [tuple(mol.GetConformer().GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    mean = float(np.mean([math.dist(pts[i], pts[j]) for i, j in bonds]))
    bonded = {frozenset(b) for b in bonds}
    for i, j in itertools.combinations(range(len(pts)), 2):
        if frozenset((i, j)) not in bonded and math.dist(pts[i], pts[j]) < 0.8 * mean:
            return False
    for (a, b), (c, d) in itertools.combinations(bonds, 2):
        if len({a, b, c, d}) == 4 and _segments_cross(pts[a], pts[b], pts[c], pts[d]):
            return False
    return True


def legible(mol):
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return False
    if {a.GetSymbol() for a in mol.GetAtoms()} - LEGIBLE_ELEMENTS:
        return False
    sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if not sizes or min(sizes) < 5 or max(sizes) > 7 or len(sizes) > 3:
        return False
    return layout_ok(mol)


def final_layout(mol):
    """2D coordinates of a final molecule in units of its mean bond length."""
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    mean = np.mean([np.linalg.norm(pts[b.GetBeginAtomIdx()] - pts[b.GetEndAtomIdx()]) for b in mol.GetBonds()])
    return pts / mean


def trajectory_metrics(X, E, mol):
    """How watchable a recorded run is when drawn on the final molecule's layout.

    ``late_chord``: over the last half of the steps, the mean total length (beyond 1.6 bond lengths) of bonds that
    are present in the state but NOT in the final molecule. A bond that the model only removes at the very end is
    a real transient, but drawn across a finished-looking molecule it reads as a drawing error.
    ``last_flip``: the last step at which anything changed (how early the run settles).
    """
    pts = final_layout(mol)
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    steps = len(X) - 1
    final_bond = E[-1] != 0
    chord = np.zeros(steps + 1)
    for k in range(steps + 1):
        spurious = np.triu((E[k] != 0) & ~final_bond, 1)
        chord[k] = np.maximum(0.0, D[spurious] - 1.6).sum()
    changed = np.where((X[1:] != X[:-1]).any(1) | (E[1:] != E[:-1]).reshape(steps, -1).any(1))[0]
    return dict(late_chord=float(chord[steps // 2:].mean()), last_flip=int(changed.max()) if len(changed) else 0)


def pair_score(logp, mols, metrics, noise_hetero, target, steps):
    """Higher is better: the adapter lands on the target, the base is clearly lower, both runs are clean to
    watch, the pair has colour, and the shared noise has some colour too."""
    score = -abs(logp["adapter"] - target)
    score -= 0.5 * max(0.0, logp["base"] - 1.6)
    score -= 0.5 * max(0.0, -0.8 - logp["base"])
    elements = {a.GetSymbol() for m in mols.values() for a in m.GetAtoms()}
    score += 0.15 * len(elements)
    score -= 1.0 * sum(m["late_chord"] for m in metrics.values())
    score -= 0.5 * sum(max(0.0, 0.7 - m["last_flip"] / steps) for m in metrics.values())   # settled before 70 %
    score -= 0.25 * max(0, 3 - noise_hetero)
    return score


def rank_candidates(work):
    """Rank every legible pair among the recorded runs (from ``states.npz``) and write ``candidates.json``."""
    with open(os.path.join(work, "meta.json")) as fh:
        meta = json.load(fh)
    st = np.load(os.path.join(work, "states.npz"))
    atom_types, steps = meta["atom_types"], meta["steps"]
    total = st["base_X"].shape[0]
    candidates = []
    for run in range(total):
        mols = {kind: mol_from_state(st[f"{kind}_X"][run][-1], st[f"{kind}_E"][run][-1], atom_types)
                for kind in ("base", "adapter")}
        if not all(legible(m) for m in mols.values()):
            continue
        logp = {kind: float(Crippen.MolLogP(m)) for kind, m in mols.items()}
        met = {kind: trajectory_metrics(st[f"{kind}_X"][run], st[f"{kind}_E"][run], mols[kind]) for kind in mols}
        hetero = int(sum(atom_types[int(c)] != "C" for c in st["base_X"][run][0]))
        candidates.append(dict(
            seed=run // meta["batch"], batch=meta["batch"], index=run % meta["batch"], run=run,
            score=pair_score(logp, mols, met, hetero, meta["target"], steps), noise_hetero=hetero,
            base=dict(smiles=Chem.MolToSmiles(mols["base"]), logp=logp["base"], **met["base"]),
            adapter=dict(smiles=Chem.MolToSmiles(mols["adapter"]), logp=logp["adapter"], **met["adapter"])))
    candidates.sort(key=lambda c: -c["score"])
    with open(os.path.join(work, "candidates.json"), "w") as fh:
        json.dump(dict(meta=meta, candidates=candidates), fh, indent=1)
    print(f"{len(candidates)} legible pairs out of {total}; the best:")
    for rank, c in enumerate(candidates[:12]):
        print(f"  #{rank:2d} logP {c['base']['logp']:5.2f} -> {c['adapter']['logp']:5.2f}   "
              f"late chords {c['base']['late_chord']:.2f}/{c['adapter']['late_chord']:.2f}   "
              f"{c['base']['smiles']}   ->   {c['adapter']['smiles']}")


# ============================================================================ commands
def cmd_record(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, adapter = load_models(args.base, args.adapter, device)
    from defog_adapt.data import zinc_reference

    atom_types = list(zinc_reference.ATOM_TYPES)
    os.makedirs(args.work, exist_ok=True)
    states = {"base_X": [], "base_E": [], "adapter_X": [], "adapter_E": []}
    for seed in range(args.seeds):
        runs = sample_pair(base, adapter, seed, args.batch, args.n, args.steps, args.target, args.weight, device)
        for k in ("X", "E"):
            for kind in ("base", "adapter"):
                states[f"{kind}_{k}"].append(np.moveaxis(runs[kind][k], 1, 0))        # -> (batch, steps+1, ...)
        print(f"seed {seed}: {(seed + 1) * args.batch} runs recorded", flush=True)
    np.savez_compressed(os.path.join(args.work, "states.npz"), **{k: np.concatenate(v) for k, v in states.items()})
    meta = dict(n=args.n, steps=args.steps, target=args.target, weight=args.weight, atom_types=atom_types,
                batch=args.batch, seeds=args.seeds)
    with open(os.path.join(args.work, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    rank_candidates(args.work)


def cmd_rank(args):
    rank_candidates(args.work)


def cmd_export(args):
    with open(os.path.join(args.work, "candidates.json")) as fh:
        data = json.load(fh)
    meta, cand = data["meta"], data["candidates"][args.rank]
    st = np.load(os.path.join(args.work, "states.npz"))
    run, steps = cand["run"], meta["steps"]
    s = np.arange(steps + 1) / steps
    flow_t = (2 * s - s * s).round(5).tolist()                       # polydec: the t the model actually sees

    def trajectory(kind):
        X, E = st[f"{kind}_X"][run], st[f"{kind}_E"][run]
        bonds = [[[int(i), int(j), int(E[k, i, j])] for i, j in itertools.combinations(range(meta["n"]), 2)
                  if E[k, i, j]] for k in range(steps + 1)]
        return dict(smiles=cand[kind]["smiles"], logp=round(cand[kind]["logp"], 2),
                    X=X.astype(int).tolist(), bonds=bonds)

    out = dict(
        meta=dict(meta, seed=cand["seed"], batch_index=cand["index"], eta=0.0, omega=0.0, time_distortion="polydec",
                  bond_classes=BOND_CLASSES, flow_t=flow_t,
                  model="frozen ZINC250k DeFoG base (E1 recipe, then two rounds of structural RL)",
                  adapter="logP adapter (cross-attention, Fourier bands), probability-space guidance",
                  note="chosen from ranked candidates for legibility; real samples, not a measurement"),
        runs={kind: trajectory(kind) for kind in ("base", "adapter")})
    out["noise"] = dict(X=out["runs"]["base"]["X"][0], bonds=out["runs"]["base"]["bonds"][0])
    assert out["noise"] == dict(X=out["runs"]["adapter"]["X"][0], bonds=out["runs"]["adapter"]["bonds"][0])
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB): base logP {out['runs']['base']['logp']} "
          f"{out['runs']['base']['smiles']}  ->  adapter logP {out['runs']['adapter']['logp']} "
          f"{out['runs']['adapter']['smiles']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record many same-noise pairs and rank them")
    r.add_argument("--base", required=True)
    r.add_argument("--adapter", required=True)
    r.add_argument("--work", required=True, help="folder for candidates.json and states.npz")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--batch", type=int, default=24)
    r.add_argument("--n", type=int, default=16, help="heavy atoms")
    r.add_argument("--steps", type=int, default=200)
    r.add_argument("--target", type=float, default=4.5)
    r.add_argument("--weight", type=float, default=1.5)
    r.set_defaults(fn=cmd_record)
    k = sub.add_parser("rank", help="re-rank the stored runs (after changing the scoring)")
    k.add_argument("--work", required=True)
    k.set_defaults(fn=cmd_rank)
    e = sub.add_parser("export", help="write the full trajectories of one ranked candidate as JSON")
    e.add_argument("--work", required=True)
    e.add_argument("--rank", type=int, default=0)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
