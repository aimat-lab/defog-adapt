#!/usr/bin/env python
"""Assemble the building blocks into a property-steered ZINC sampler.

This is the "recipe" half of "building blocks plus a recipe". The library gives you the parts; this
script shows how they were wired for every recorded conditional result:

    frozen base  +  one adapter per property (composed)  +  learned P(n | target)  +  Feynman-Kac
    steering scored by a learned head  ->  decode  ->  score

Steer one property, with every part:

    python examples/steer_zinc.py --base ckpts/zinc_rl2_seed42/best_model --method fk \\
        --property logp --adapter runs/logp_adapter.ckpt --head ckpts/heads/logp_head.ckpt \\
        --size ckpts/heads/logp_head_size.ckpt --target 4.0 --n 10

Steer two at once (the adapters multiply, the size models multiply, the energies add):

    python examples/steer_zinc.py --base ... --method fk --mode mean \\
        --property logp --adapter ... --head ... --size ... --target 4.0 \\
        --property qed  --adapter ... --head ... --size ... --target 0.8

Or measure it, with the shape of the benchmark protocol the recorded numbers used (targets drawn from
real validation molecules, several generations per target, the property of every valid molecule read
off the decoded graph, a pooled error to the target):

    python examples/steer_zinc.py --base ... --property logp --adapter ... --evaluate 100 --n 10

Each ``--property`` takes its own ``--adapter`` (and optionally ``--head``, ``--size``, ``--target``,
``--weight``); give the flags in the same order. Only ``--adapter`` is required.

The parts, and where the recorded evidence says each one matters:

``--method adapter``  The adapter composition as the proposal, sampled with ``AdaptedSampler``.
                      Classifier-free guidance blended in PROBABILITY space at the property's own best
                      weight (1.5 for logP, TPSA and SAScore; 2.0 for QED).
``--method fk``       Adds Feynman-Kac steering on top: ``FeynmanKacSampler`` runs a particle system,
                      scores each particle's predicted clean molecule with the head, and resamples
                      toward the target. The energy goes through ``StandardisedEnergy``, so ``--beta``
                      is dimensionless and means the same for logP as for QED. It is the largest lever
                      measured (logP error 0.3132 -> 0.2299 on validation), at a cost in diversity
                      (uniqueness about 0.76), and the value of beta matters far more than the
                      particle count.
``--size``            Draws the node count from P(n | target) instead of the dataset marginal. A bias
                      correction worth -0.0103 on logP and -0.0058 on QED; the size model must have
                      been fit under the same label convention as the adapter.
``--mode``            How several adapters combine: "product" or "mean". The best one is pair-
                      dependent (mean won logP + QED, product won logP + TPSA), so do not hard-code it.

What it does not do: it is not the benchmark harness. The recorded numbers came from a separate
evaluation script; this one follows the same protocol shape, so numbers agree within the run-to-run
noise (about 0.011 on a 100-target logP error) but are not bit-reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import decode_valid, label_graphs, property_fn, zinc_vocabulary  # noqa: E402

from defog_adapt.core import (  # noqa: E402
    AdaLNAdapter, AdaptedSampler, AdapterComposition, ComposedSizeDistribution, ConditionBranch,
    DeFoGModel, FeynmanKacSampler, LearnedPropertyEnergy, LearnedSizeDistribution,
    MoleculePropertyEnergy, PropertyHead, SizeBranch, StandardisedEnergy,
)
from defog_adapt.data import vocabulary, zinc_reference  # noqa: E402
from defog_adapt.domains import MoleculeDomain  # noqa: E402
from defog_adapt.domains.molecule import build_encoders, mol_to_smiles  # noqa: E402
from defog_adapt.presets import ADAPTER_SAMPLING, ADAPTER_WEIGHT, FK_STEERING  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@dataclass
class Steered:
    """One steered property: its adapter, the optional head and size model, and the target."""
    name: str
    adapter: AdaLNAdapter
    head: Optional[PropertyHead]
    size_model: Optional[LearnedSizeDistribution]
    target: Optional[float]
    weight: float

    @property
    def scale(self) -> float:
        """The property's standard deviation: what makes the Feynman-Kac energy dimensionless."""
        if self.head is not None:
            return float(self.head.prop_std)
        return float(self.adapter.cond_std.reshape(-1)[0])


def pair_check(spec: Steered) -> None:
    """The adapter, head and size model must agree about what a target means.

    Compared on a scale-free tolerance, 1% of a standard deviation: the parts legitimately differ by
    a few parts in ten thousand of a std (they are fit on slightly different subsets), which is noise,
    while the real failure (one trained on source labels and another on decoded ones) is a third of a
    standard deviation and is refused."""
    mean = float(spec.adapter.cond_mean.reshape(-1)[0])
    std = float(spec.adapter.cond_std.reshape(-1)[0])
    if spec.head is not None and abs(float(spec.head.prop_mean) - mean) > 0.01 * std:
        raise RuntimeError(
            f"{spec.name}: the head's mean ({float(spec.head.prop_mean):.4f}) and the adapter's "
            f"({mean:.4f}) differ by more than 1% of a standard deviation. They were probably trained "
            f"under different label conventions (source vs decoded); steering would pull against "
            f"the adapter.")
    if spec.size_model is not None:
        spec.size_model.check_compatible(spec.adapter, tol=0.01 * std)


def size_draw(specs: List[Steered], n: int, mode: str):
    """``(size_dist, condition)`` for ``n`` samples, or ``(None, None)`` for the dataset marginal.

    One size model is used as it is. Several are combined into a product of experts over the node-count
    grid, the same log-ratio algebra the rate matrices are blended with; that recovers about 87% of what
    a joint size model would. A composition built for ``n`` rows cannot serve a different count (pairing
    one molecule's size draw with another's target would be silent), so build one per call."""
    if any(s.size_model is None for s in specs):
        return None, None
    branches = [SizeBranch(dist=s.size_model,
                           condition=torch.full((n, s.size_model.cond_dim), float(s.target)),
                           weight=1.0) for s in specs]
    if len(branches) == 1:
        return branches[0].dist, branches[0].condition
    return ComposedSizeDistribution(branches, mode=mode), None


def build_sampler(base, specs, *, method, mode, ctmc, beta, domain, atom_enc, bond_enc):
    composition = AdapterComposition(
        [ConditionBranch(s.adapter, torch.tensor([float(s.target)]), s.weight) for s in specs],
        base=base, mode=mode)
    if method == "adapter":
        return AdaptedSampler(base, composition, **ctmc)
    terms = []
    for s in specs:
        if s.head is not None:
            energy = LearnedPropertyEnergy(s.head, s.target, domain, atom_enc, bond_enc, scale=s.scale)
        else:                                  # no head: score with the closed-form property instead
            energy = MoleculePropertyEnergy(domain, property_fn(s.name), s.target, scale=s.scale)
        terms.append((energy, s.scale, 1.0))
    # Standardised: beta is a squared z-score, so one value carries from one property to the next.
    return FeynmanKacSampler(base, StandardisedEnergy(terms), composition=composition,
                             **{**FK_STEERING, "beta": beta}, **ctmc)


def generate(base, specs, n, *, method, mode, ctmc, beta, domain, atom_enc, bond_enc, device, use_size):
    sampler = build_sampler(base, specs, method=method, mode=mode, ctmc=ctmc, beta=beta,
                            domain=domain, atom_enc=atom_enc, bond_enc=bond_enc)
    size_dist, condition = size_draw(specs, n, mode) if use_size else (None, None)
    return sampler.sample(n, size_dist=size_dist, condition=condition, device=device,
                          show_progress=False)


def read_properties(graphs, specs, atom_dec, bond_dec):
    """For each graph: its canonical SMILES (or None) and each steered property of the decoded molecule."""
    out = []
    for g in graphs:
        mols = decode_valid([g], atom_dec, bond_dec)
        if not mols:
            out.append((None, None))
            continue
        mol = mols[0]
        try:
            values = [property_fn(s.name)(mol) for s in specs]
        except Exception:                                  # noqa: BLE001
            out.append((None, None))
            continue
        out.append((Chem.MolToSmiles(mol), values))
    return out


def evaluation_targets(specs, k, seed, data_root, atom_enc, bond_enc, atom_dec, bond_dec,
                       split="validation"):
    """``k`` target tuples read off real molecules of ``split`` (their DECODED property values), so
    every target is one that some molecule actually achieves.

    ``split`` is ``"validation"`` (where configurations are chosen) or ``"test"`` (held out: one
    evaluation pass, with a configuration already frozen on validation)."""
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    reference = zinc_reference.load_reference_split(data_root)
    pool = list(reference.val_smiles if split == "validation" else reference.test_smiles)
    order = np.random.default_rng(seed).permutation(len(pool))
    _, _, kekulize = zinc_vocabulary()
    targets = []
    for i in order:
        row = []
        for s in specs:
            _, values, _ = label_graphs([pool[int(i)]], property_fn(s.name), atom_encoder=atom_enc,
                                        bond_encoder=bond_enc, atom_decoder=atom_dec,
                                        bond_decoder=bond_dec, kekulize=kekulize,
                                        property_from="decoded", keep_graphs=False)
            if values.size != 1:
                break
            row.append(float(values[0]))
        else:
            targets.append(row)
        if len(targets) == k:
            break
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="frozen base checkpoint")
    ap.add_argument("--property", action="append", required=True, help="logp | tpsa | qed | sascore")
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--head", action="append", default=[], help="property head (Feynman-Kac energy)")
    ap.add_argument("--size", action="append", default=[], help="learned size model")
    ap.add_argument("--target", action="append", type=float, default=[])
    ap.add_argument("--weight", action="append", type=float, default=[],
                    help="guidance weight (default: the property's measured optimum)")
    ap.add_argument("--method", choices=("adapter", "fk"), default="adapter")
    ap.add_argument("--mode", choices=("product", "mean"), default="product",
                    help="how several adapters (and size models) combine")
    ap.add_argument("--no-size-model", action="store_true",
                    help="ignore --size and draw sizes from the dataset marginal")
    ap.add_argument("--n", type=int, default=10, help="molecules (FK: particles) per target")
    ap.add_argument("--steps", type=int, default=ADAPTER_SAMPLING["sample_steps"])
    ap.add_argument("--eta", type=float, default=ADAPTER_SAMPLING["eta"])
    ap.add_argument("--omega", type=float, default=ADAPTER_SAMPLING["omega"])
    ap.add_argument("--beta", type=float, default=FK_STEERING["beta"],
                    help="FK tilt strength, dimensionless (default: the recorded 20)")
    ap.add_argument("--evaluate", type=int, default=0, metavar="K",
                    help="measure over K targets drawn from real validation molecules")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the SMILES and a JSON report here")
    args = ap.parse_args()

    n_props = len(args.property)
    for name, values in (("--adapter", args.adapter), ("--head", args.head), ("--size", args.size),
                         ("--target", args.target), ("--weight", args.weight)):
        if values and len(values) != n_props:
            ap.error(f"{name} was given {len(values)} times but --property {n_props} times; "
                     f"give one per property, in the same order")
    if args.method == "fk" and args.mode == "product" and n_props > 1:
        print("note: --mode product with several properties; 'mean' won logP + QED, so try both")
    if not args.evaluate and len(args.target) != n_props:
        ap.error("give one --target per property, or use --evaluate K")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    atom_types, bond_types, _ = zinc_vocabulary()
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atom_types, bond_types)
    domain = MoleculeDomain(atom_dec, bond_dec)

    base = DeFoGModel.load(args.base, device="cpu").to(device).eval()
    print(vocabulary.check_model(base, atom_types, bond_types, what=f"base {args.base}"))
    specs = []
    for k, name in enumerate(args.property):
        adapter = AdaLNAdapter.load(args.adapter[k], device=device).eval()
        adapter.check_compatible(base)
        head = PropertyHead.load(args.head[k], device=device) if args.head else None
        size = (LearnedSizeDistribution.load(args.size[k], device=device)
                if args.size and not args.no_size_model else None)
        spec = Steered(name=name, adapter=adapter, head=head, size_model=size,
                       target=args.target[k] if args.target else None,
                       weight=args.weight[k] if args.weight else ADAPTER_WEIGHT[name])
        pair_check(spec)
        specs.append(spec)
        print(f"{name}: adapter w={spec.weight}, head={'yes' if head else 'no'}, "
              f"size model={'yes' if size else 'no (dataset marginal)'}")
    if args.method == "fk" and not all(s.head is not None for s in specs):
        print("note: some properties have no --head; FK scores them with the closed-form RDKit property")

    ctmc = dict(sample_steps=args.steps, eta=args.eta, omega=args.omega,
                time_distortion=ADAPTER_SAMPLING["time_distortion"])
    common = dict(method=args.method, mode=args.mode, ctmc=ctmc, beta=args.beta, domain=domain,
                  atom_enc=atom_enc, bond_enc=bond_enc, device=device, use_size=not args.no_size_model)

    if not args.evaluate:
        for s, t in zip(specs, args.target):
            s.target = t
        start = time.time()
        graphs = generate(base, specs, args.n, **common)
        rows = read_properties(graphs, specs, atom_dec, bond_dec)
        print(f"\n{args.n} samples in {time.time() - start:.0f}s "
              f"({sum(r[0] is not None for r in rows)} valid)")
        for smi, values in rows:
            if smi is None:
                print("  (invalid)")
            else:
                print(f"  {smi:<60} " + " ".join(f"{s.name}={v:.2f}" for s, v in zip(specs, values)))
        if args.out:
            with open(args.out, "w") as fh:
                fh.write("\n".join(r[0] or "" for r in rows) + "\n")
        return 0

    # ------------------------------------------------------------------ evaluation
    targets = evaluation_targets(specs, args.evaluate, args.seed, args.data_root, atom_enc,
                                 bond_enc, atom_dec, bond_dec)
    print(f"\nevaluating {len(targets)} targets x {args.n} generations "
          f"(method={args.method}, steps={args.steps}, eta={args.eta})")
    errors = [[] for _ in specs]
    smiles, total, start = [], 0, time.time()
    for i, row in enumerate(targets):
        for s, t in zip(specs, row):
            s.target = t
        graphs = generate(base, specs, args.n, **common)
        total += len(graphs)
        for smi, values in read_properties(graphs, specs, atom_dec, bond_dec):
            if smi is None:
                continue
            smiles.append(smi)
            for j, (v, t) in enumerate(zip(values, row)):
                errors[j].append(abs(v - t))
        if (i + 1) % 10 == 0 or i + 1 == len(targets):
            mae = [float(np.mean(e)) if e else float("nan") for e in errors]
            print(f"  {i + 1:4d}/{len(targets)} targets  running MAE "
                  + " ".join(f"{s.name}={m:.4f}" for s, m in zip(specs, mae))
                  + f"  validity {len(smiles) / max(1, total):.3f}  [{time.time() - start:.0f}s]")
    mae = [float(np.mean(e)) if e else float("nan") for e in errors]
    report = {
        "method": args.method, "mode": args.mode, "n_targets": len(targets), "n_per_target": args.n,
        "steps": args.steps, "eta": args.eta, "beta": args.beta if args.method == "fk" else None,
        "size_model": all(s.size_model is not None for s in specs) and not args.no_size_model,
        "properties": [s.name for s in specs], "weights": [s.weight for s in specs],
        "mae": dict(zip((s.name for s in specs), mae)),
        "total_mae": float(np.mean(mae)),            # the unweighted mean of the per-property errors
        "validity": len(smiles) / max(1, total),
        "uniqueness": len(set(smiles)) / max(1, len(smiles)),
        "seed": args.seed, "seconds": round(time.time() - start, 1)}
    print("\n" + json.dumps(report, indent=2))
    if args.out:
        with open(args.out + ".json", "w") as fh:
            json.dump(report, fh, indent=2)
        with open(args.out + ".smi", "w") as fh:
            fh.write("\n".join(smiles) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
