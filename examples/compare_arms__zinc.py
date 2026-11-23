#!/usr/bin/env python
"""Compare the ways of conditioning a ZINC model on logP, on one benchmark.

This is the harness behind the README's results figure. It measures five arms on the same targets:

    base         the unconditional base as it is: nothing steers it. The do-nothing reference.
    conditional  the same recipe trained end to end on logP (``training__e1.py --CONDITION "'logp'"``).
    film         a frozen-base adapter with FiLM only (``adapter_training__zinc.py`` with the
                 cross-attention and the Fourier bands switched off).
    xattn        a frozen-base adapter with node-to-condition cross-attention and Fourier bands.
    xattn_fk     the ``xattn`` adapter with Feynman-Kac steering on top, scored by a learned property head.

Run the arms and write one JSON per (arm, seed), then summarise a directory of them:

    python examples/compare_arms__zinc.py run --split validation --out OUT --base BASE \\
        --conditional E2E/best_model.ckpt --film FILM.ckpt --film-weight 2.0 \\
        --xattn XATTN.ckpt --xattn-weight 1.5 --head HEAD.ckpt --size SIZE.ckpt --seeds 42
    python examples/compare_arms__zinc.py summarize OUT

The protocol is the benchmark's (docs/about/results-and-caveats.md, "How it was measured"): ``--targets`` target values drawn from the
real molecules of a split, ``--n`` generations per target, and the error is the pooled mean absolute
difference between each valid generation's logP, read off the DECODED graph, and its target. The seed
fixes the targets, so every arm of a seed sees the same ones. Everything else is the recorded conditional
configuration: 500 steps, eta 25, omega 0, polydec time distortion, Feynman-Kac at beta 20 with 10
particles.

Three choices that a number from this script depends on:

* Every conditional arm draws its node counts from the learned P(n | target) (``--size``), because the
  size draw is part of what conditioning means. The base cannot condition, and draws from the dataset
  marginal. ``--no-size-model`` puts every arm on the marginal.
* The conditional model is sampled at guidance weight 1 by default. That is the exact conditional it was
  trained to be, and it needs no blend: a model like this is guided, if at all, by blending rate matrices,
  which was measured to break above weight 1 for adapters (docs/about/results-and-caveats.md, "What these
  numbers do and do not show"). ``--conditional-weight`` exists so that this is measured, not assumed.
* ``--split test`` is the held-out split: one evaluation pass, with every setting frozen on validation. A
  result is refused for reuse if it was made under any other configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import read_condition_sidecar, zinc_vocabulary  # noqa: E402
from steer_zinc import Steered, evaluation_targets, generate, pair_check, read_properties  # noqa: E402

from rdkit import RDLogger  # noqa: E402

import defog_adapt  # noqa: E402
from defog_adapt.core import (  # noqa: E402
    CrossAttentionAdapter, DeFoGModel, LearnedSizeDistribution, PropertyHead, Sampler,
)
from defog_adapt.data import vocabulary  # noqa: E402
from defog_adapt.domains import MoleculeDomain  # noqa: E402
from defog_adapt.domains.molecule import build_encoders  # noqa: E402
from defog_adapt.presets import ADAPTER_SAMPLING, FK_STEERING  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROPERTY = "logp"
ARMS = ("base", "conditional", "film", "xattn", "xattn_fk")
LABELS = {
    "base": "unconditional base",
    "conditional": "end-to-end conditional",
    "film": "FiLM adapter",
    "xattn": "cross-attention adapter",
    "xattn_fk": "+ Feynman-Kac",
}
#: What makes two runs of one (arm, seed) the same experiment. A result file that disagrees on any of it
#: is a different experiment, and silently reusing it would put its number under this configuration's name.
IDENTITY = ("arm", "seed", "split", "n_targets", "n_per_target", "sampling", "guidance_weight", "beta",
            "size_model", "checkpoints")


# ============================================================================ pairing checks
def check_conditional_pairing(stats: dict, size_model) -> None:
    """The end-to-end model, its condition file and the size model must agree about what a target means.

    The same scale-free tolerance as ``steer_zinc.pair_check``: 1% of a standard deviation separates the
    parts being fit on slightly different subsets (noise) from one trained on source labels and another on
    decoded ones (a third of a standard deviation)."""
    if stats["property"] != PROPERTY or stats["property_from"] != "decoded":
        raise RuntimeError(
            f"the conditional model was trained on {stats['property']} labels from the "
            f"{stats['property_from']} molecule; this comparison is {PROPERTY} on decoded labels")
    if size_model is None:
        return
    std = float(stats["std"])
    mean_there, std_there = float(size_model.cond_mean.reshape(-1)[0]), float(size_model.cond_std.reshape(-1)[0])
    if abs(float(stats["mean"]) - mean_there) > 0.01 * std or abs(std - std_there) > 0.01 * std:
        raise RuntimeError(
            f"the conditional model's statistics (mean {stats['mean']:.4f}, std {std:.4f}) and the size "
            f"model's ({mean_there:.4f}, {std_there:.4f}) differ by more than 1% of a standard deviation. "
            f"They were probably fit under different label conventions (source vs decoded).")


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ============================================================================ the arms
def conditional_draw(model, stats: dict, size_model, ctmc: dict, weight: float, device) -> Callable:
    """Draw ``n`` molecules from the end-to-end model for one target.

    The two conditions are not the same number. The size model takes the RAW target and normalises it
    inside; the model takes it STANDARDISED by the caller, with the statistics it was trained under. The
    sampler hands one ``condition`` to both, which would be right for one of them only, so the sizes are
    drawn here and passed in as fixed node counts.
    """
    sampler = Sampler(model, guidance_scale=float(weight), **ctmc)
    mean, std = float(stats["mean"]), float(stats["std"])

    def draw(target: float, n: int) -> list:
        sizes = None
        if size_model is not None:
            sizes = size_model.sample(n, condition=torch.full((n, 1), float(target)), device=device)
        condition = torch.full((n, 1), (float(target) - mean) / std)
        return sampler.sample(n, num_nodes=sizes, condition=condition, device=device, show_progress=False)

    return draw


def base_draw(base, ctmc: dict, device) -> Callable:
    sampler = Sampler(base, **ctmc)

    def draw(target: float, n: int) -> list:               # the target is not an input: this is the point
        return sampler.sample(n, device=device, show_progress=False)

    return draw


def adapter_draw(base, spec: Steered, *, method: str, ctmc: dict, beta: float, domain, atom_enc, bond_enc,
                 device, use_size: bool) -> Callable:
    def draw(target: float, n: int) -> list:
        spec.target = float(target)
        return generate(base, [spec], n, method=method, mode="product", ctmc=ctmc, beta=beta, domain=domain,
                        atom_enc=atom_enc, bond_enc=bond_enc, device=device, use_size=use_size)

    return draw


# ============================================================================ measuring one arm
def evaluate(draw: Callable, targets: Sequence[float], n: int, atom_dec, bond_dec, log=print) -> dict:
    """The benchmark's numbers for one arm: ``draw(target, n)`` for each target, scored on the decoded graph."""
    errors: List[List[float]] = []
    smiles: List[str] = []
    n_total, start = 0, time.time()
    scorer = [SimpleNamespace(name=PROPERTY)]
    for i, target in enumerate(targets):
        graphs = draw(float(target), n)
        n_total += len(graphs)
        row = []
        for smi, values in read_properties(graphs, scorer, atom_dec, bond_dec):
            if smi is None:
                continue
            smiles.append(smi)
            row.append(abs(values[0] - float(target)))
        errors.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(targets):
            pooled = [x for r in errors for x in r]
            log(f"    {i + 1:4d}/{len(targets)} targets  running error "
                f"{_fmt(float(np.mean(pooled)) if pooled else None)}  validity "
                f"{len(smiles) / max(1, n_total):.3f}  [{time.time() - start:.0f}s]")
    pooled = [x for r in errors for x in r]
    return {
        "mae": float(np.mean(pooled)) if pooled else None,
        "validity": len(smiles) / max(1, n_total),
        "uniqueness": len(set(smiles)) / max(1, len(smiles)),
        "n_valid": len(smiles), "n_total": n_total,
        "targets": [float(t) for t in targets],
        "per_target_mae": [float(np.mean(r)) if r else None for r in errors],
        "per_target_valid": [len(r) for r in errors],
        "smiles": smiles,
        "seconds": round(time.time() - start, 1),
    }


def _fmt(value: Optional[float]) -> str:
    """A number for a log line; an arm with no valid molecule has no error to show."""
    return "nan" if value is None else f"{value:.4f}"


def same_experiment(a: dict, b: dict) -> List[str]:
    """The identity fields on which two result records disagree (empty: they are the same experiment)."""
    return [key for key in IDENTITY if a.get(key) != b.get(key)]


# ============================================================================ summarising
def _number(value: Optional[float]) -> float:
    """A run with no valid molecule has no error (``None``); it reads as ``nan``, not as a crash and not as 0."""
    return float("nan") if value is None else float(value)


def _mean_sd(values: Sequence[Optional[float]]):
    """Mean and sample sd, with an arm that had nothing valid propagating as ``nan``."""
    values = [_number(v) for v in values]
    if not values:
        return None, None
    mean = float(np.mean(values))
    return mean, (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def summarize(directory, order: Sequence[str] = ARMS) -> dict:
    """Mean and sample sd over seeds for every arm in ``directory``, and the paired gain between arms.

    The paired gain is the per-seed difference between one arm and the next one (in ``order``) that is
    present, over the seeds both have: the seeds share their targets, so that difference is not diluted
    by which targets a seed happened to draw."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*__seed*.json"))]
    if not records:
        raise FileNotFoundError(f"no '<arm>__seed<k>.json' results in {directory}")
    protocol_keys = ("split", "n_targets", "n_per_target", "sampling")
    protocols = {json.dumps({k: r[k] for k in protocol_keys}, sort_keys=True) for r in records}
    if len(protocols) != 1:
        raise RuntimeError(f"the results in {directory} were not all made under one protocol: {sorted(protocols)}")
    unknown = sorted({r["arm"] for r in records} - set(ARMS))
    if unknown:
        raise RuntimeError(f"unknown arms in {directory}: {unknown}")

    arms: Dict[str, dict] = {}
    for arm in order:
        runs = sorted((r for r in records if r["arm"] == arm), key=lambda r: r["seed"])
        if not runs:
            continue
        if len({r["seed"] for r in runs}) != len(runs):
            raise RuntimeError(f"arm {arm} has two results for one seed in {directory}")
        mean, sd = _mean_sd([r["mae"] for r in runs])
        arms[arm] = {
            "label": LABELS[arm], "seeds": [r["seed"] for r in runs], "mae_per_seed": [r["mae"] for r in runs],
            "mae_mean": mean, "mae_sd": sd,
            "validity_mean": _mean_sd([r["validity"] for r in runs])[0],
            "uniqueness_mean": _mean_sd([r["uniqueness"] for r in runs])[0],
            "guidance_weight": runs[0].get("guidance_weight"), "beta": runs[0].get("beta"),
            "size_model": runs[0].get("size_model"),
        }
    present = [a for a in order if a in arms]
    paired = {}
    for before, after in zip(present, present[1:]):
        b = {s: _number(v) for s, v in zip(arms[before]["seeds"], arms[before]["mae_per_seed"])}
        a = {s: _number(v) for s, v in zip(arms[after]["seeds"], arms[after]["mae_per_seed"])}
        shared = sorted(set(a) & set(b))
        diffs = [a[s] - b[s] for s in shared]
        mean, sd = _mean_sd(diffs)
        paired[f"{after} - {before}"] = {"seeds": shared, "per_seed": diffs, "mean": mean, "sd": sd}
    summary = {"protocol": json.loads(protocols.pop()), "arms": arms, "paired": paired}
    Path(directory, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: dict) -> str:
    p = summary["protocol"]
    lines = [f"{PROPERTY} error on the {p['split']} split: {p['n_targets']} targets x {p['n_per_target']} "
             f"generations, {p['sampling']['sample_steps']} steps (mean +- sd over seeds; lower is better)", "",
             "| arm | error | validity | uniqueness | per seed |", "|---|---|---|---|---|"]
    for arm, a in summary["arms"].items():
        sd = "" if a["mae_sd"] is None else f" +- {_fmt(a['mae_sd'])}"
        seeds = ", ".join(f"{s}: {_fmt(v)}" for s, v in zip(a["seeds"], a["mae_per_seed"]))
        lines.append(f"| {a['label']} | {_fmt(a['mae_mean'])}{sd} | {a['validity_mean']:.3f} | "
                     f"{a['uniqueness_mean']:.3f} | {seeds} |")
    if summary["paired"]:
        lines += ["", "paired difference per seed (the later arm minus the earlier one):"]
        for name, d in summary["paired"].items():
            sd = "" if d["sd"] is None else f" +- {d['sd']:.4f}"
            lines.append(f"  {name}: {d['mean']:+.4f}{sd}   ({', '.join(f'{x:+.4f}' for x in d['per_seed'])})")
    return "\n".join(lines)


# ============================================================================ command line
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure arms and write one JSON per (arm, seed)")
    run.add_argument("--out", required=True, help="directory for the results (created if missing)")
    run.add_argument("--split", required=True, choices=("validation", "test"),
                     help="where the targets come from; 'test' is the held-out, read-once split")
    run.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    run.add_argument("--seeds", nargs="+", type=int, default=[42])
    run.add_argument("--targets", type=int, default=100, help="targets per seed (the benchmark's 100)")
    run.add_argument("--n", type=int, default=10, help="generations per target (Feynman-Kac: particles)")
    run.add_argument("--base", help="the frozen base (also the unconditional arm)")
    run.add_argument("--conditional", help="the end-to-end model; its condition.json must sit beside it")
    run.add_argument("--conditional-stats", help="the condition file, if it is not beside the model")
    run.add_argument("--conditional-weight", type=float, default=1.0,
                     help="guidance weight of the end-to-end model (1 = no guidance)")
    run.add_argument("--film", help="the FiLM-only adapter")
    run.add_argument("--film-weight", type=float, help="its guidance weight (no default: choose it on validation)")
    run.add_argument("--xattn", help="the cross-attention adapter")
    run.add_argument("--xattn-weight", type=float, help="its guidance weight (also Feynman-Kac's proposal)")
    run.add_argument("--head", help="the property head Feynman-Kac scores with")
    run.add_argument("--size", help="the learned size model P(n | target)")
    run.add_argument("--no-size-model", action="store_true", help="draw every arm's sizes from the marginal")
    run.add_argument("--beta", type=float, default=FK_STEERING["beta"], help="Feynman-Kac tilt (dimensionless)")
    run.add_argument("--steps", type=int, default=ADAPTER_SAMPLING["sample_steps"])
    run.add_argument("--eta", type=float, default=ADAPTER_SAMPLING["eta"])
    run.add_argument("--omega", type=float, default=ADAPTER_SAMPLING["omega"])
    run.add_argument("--data-root", default=None)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    summ = sub.add_parser("summarize", help="mean +- sd over seeds of a directory of results")
    summ.add_argument("directory")
    return ap


def require(parser, args) -> None:
    """Every arm names what it needs; a missing piece is an error now, not an hour into a run."""
    needs = {"base": ["base"], "conditional": ["conditional"], "film": ["base", "film", "film_weight"],
             "xattn": ["base", "xattn", "xattn_weight"], "xattn_fk": ["base", "xattn", "xattn_weight", "head"]}
    for arm in args.arms:
        missing = [f"--{k.replace('_', '-')}" for k in needs[arm] if getattr(args, k) is None]
        if missing:
            parser.error(f"arm '{arm}' needs {', '.join(missing)}")
        if arm != "base" and not args.no_size_model and args.size is None:
            parser.error(f"arm '{arm}' draws its node counts from a learned size model: give --size, "
                         f"or --no-size-model to put every arm on the dataset marginal")


def run_command(args) -> int:
    device = torch.device(args.device)
    atom_types, bond_types, _ = zinc_vocabulary()
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atom_types, bond_types)
    domain = MoleculeDomain(atom_dec, bond_dec)
    ctmc = dict(sample_steps=args.steps, eta=args.eta, omega=args.omega,
                time_distortion=ADAPTER_SAMPLING["time_distortion"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_size = not args.no_size_model

    def load_model(path, what):
        model = DeFoGModel.load(path, device="cpu").to(device).eval()
        print(vocabulary.check_model(model, atom_types, bond_types, what=f"{what} {Path(path).name}"))
        return model

    base = load_model(args.base, "base") if args.base else None
    size = (LearnedSizeDistribution.load(args.size, device=device) if use_size and args.size else None)
    files = {"base": args.base, "size": args.size if use_size else None, "head": args.head}
    draws: Dict[str, Callable] = {}
    weights: Dict[str, Optional[float]] = {a: None for a in args.arms}
    betas: Dict[str, Optional[float]] = {a: None for a in args.arms}
    for arm in args.arms:
        if arm == "base":
            draws[arm] = base_draw(base, ctmc, device)
        elif arm == "conditional":
            model = load_model(args.conditional, "conditional")
            stats_path = args.conditional_stats or Path(args.conditional).with_name("condition.json")
            stats = read_condition_sidecar(stats_path)
            check_conditional_pairing(stats, size)
            if model.cond_dim != 1:
                raise RuntimeError(f"{args.conditional} has cond_dim={model.cond_dim}; expected a width-1 condition")
            draws[arm] = conditional_draw(model, stats, size, ctmc, args.conditional_weight, device)
            weights[arm] = float(args.conditional_weight)
            files.update(conditional=args.conditional, conditional_stats=str(stats_path))
        else:
            path, weight = (args.film, args.film_weight) if arm == "film" else (args.xattn, args.xattn_weight)
            adapter = CrossAttentionAdapter.load(path, device=device).eval()
            adapter.check_compatible(base)
            head = PropertyHead.load(args.head, device=device) if arm == "xattn_fk" else None
            spec = Steered(name=PROPERTY, adapter=adapter, head=head, size_model=size, target=None,
                           weight=float(weight))
            pair_check(spec)
            draws[arm] = adapter_draw(base, spec, method="fk" if arm == "xattn_fk" else "adapter", ctmc=ctmc,
                                      beta=args.beta, domain=domain, atom_enc=atom_enc, bond_enc=bond_enc,
                                      device=device, use_size=use_size)
            weights[arm] = float(weight)
            betas[arm] = float(args.beta) if arm == "xattn_fk" else None
            files["film" if arm == "film" else "xattn"] = path

    checkpoints = {name: sha256_of(path) for name, path in files.items() if path}
    if args.split == "test":
        print("=" * 78 + "\nTEST SPLIT: held out. One evaluation pass, with the configuration frozen on "
              "validation.\n" + "=" * 78)
    for seed in args.seeds:
        targets = [row[0] for row in evaluation_targets(
            [SimpleNamespace(name=PROPERTY)], args.targets, seed, args.data_root, atom_enc, bond_enc, atom_dec,
            bond_dec, split=args.split)]
        for arm in args.arms:
            identity = {
                "arm": arm, "seed": seed, "split": args.split, "n_targets": len(targets),
                "n_per_target": args.n, "sampling": ctmc, "guidance_weight": weights[arm], "beta": betas[arm],
                "size_model": bool(use_size and arm != "base" and size is not None),
                "checkpoints": {k: v for k, v in checkpoints.items() if k in _used_files(arm)},
            }
            path = out / f"{arm}__seed{seed}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                differs = same_experiment(previous, identity)
                if differs:
                    raise RuntimeError(f"{path} was made under another configuration (differs in {differs}); "
                                       f"refusing to reuse it or overwrite it. Use another --out.")
                print(f"[{arm} seed {seed}] done already: error {_fmt(previous['mae'])}")
                continue
            print(f"[{arm} seed {seed}] {len(targets)} targets x {args.n} on {args.split}"
                  + (f", guidance {weights[arm]}" if weights[arm] is not None else "")
                  + (f", beta {betas[arm]}" if betas[arm] is not None else ""))
            torch.manual_seed(seed)
            result = evaluate(draws[arm], targets, args.n, atom_dec, bond_dec)
            record = {**identity, "label": LABELS[arm], "property": PROPERTY, **result,
                      "library_version": defog_adapt.__version__}
            tmp = path.with_suffix(".json.partial")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            os.replace(tmp, path)                          # a result file exists only when complete
            print(f"[{arm} seed {seed}] error {_fmt(result['mae'])}  validity {result['validity']:.3f}  "
                  f"uniqueness {result['uniqueness']:.3f}  ({result['seconds']:.0f}s)")
    return 0


def _used_files(arm: str) -> set:
    """Which checkpoints an arm's number depends on (its identity must not change when another arm's does)."""
    return {"base": {"base"}, "conditional": {"conditional", "conditional_stats", "size"},
            "film": {"base", "film", "size"}, "xattn": {"base", "xattn", "size"},
            "xattn_fk": {"base", "xattn", "head", "size"}}[arm]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summarize":
        print(format_summary(summarize(args.directory)))
        return 0
    require(parser, args)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
