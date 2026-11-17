#!/usr/bin/env python
"""Fit a standalone property head (and, optionally, a learned size model) for a frozen base.

    python examples/fit_property_head__zinc.py --base ckpts/zinc_rl2_seed42/best_model \\
        --property logp --out ckpts/heads/logp_head --with-size-model

A head is a graph -> scalar regressor. It needs NO adapter: it is fitted by grounding regression on
the same (graph, label) pairs an adapter is trained on, with no adapter in the loop, so a head can
be added to an adapter that already exists. Two things use it:

* Feynman-Kac steering. :class:`~defog_adapt.core.LearnedPropertyEnergy` scores each predicted-
  clean particle by the squared error of the head's prediction to the target. The particles are
  DISCRETE one-hot graphs, which is why a head works there where soft-input coupling did not.
* RL fine-tuning. ``HeadPropertyMatchReward`` uses the head as the reward for properties with no
  closed form, and the early-stop probe selects on it.

THE LABELS MUST MATCH THE ADAPTER'S. A head trained on source-SMILES labels and paired with an
adapter trained on decoded-graph labels would disagree about what the target means at the end of the
range where the lost formal charge bites, and the energy would then pull against the adapter, with
both parts individually correct. The effect is large: a logP head whose error against decoded labels
was 0.083 reads 0.461 against source labels, purely from the 0.385 source-versus-decoded logP gap. So
pass the same ``--property-from`` here as the adapter was trained with.

Judge a head against its OWN property's spread, in the property's own units: the held-out error is
printed beside the standard deviation, and a head no better than predicting the mean is flagged. The
recorded heads scored 4 to 18 times tighter than the adapters they steer (against decoded labels:
logP 0.0185, QED 0.0212, TPSA 0.623, SAScore 0.0366).

``--with-size-model`` also fits ``P(n | property)``, the learned node-count distribution, from the
SAME rows and the SAME label convention, and writes it beside the head as ``<out>_size.ckpt``.
Drawing graph sizes from it instead of the dataset marginal is a bias correction on the size draw
(it does not narrow it). On the test split it was worth -0.0103 logP error and -0.0058 QED error,
and in the RL comparison about half of the total QED gain. The fitter reports ``gain_nats`` over the
marginal; below 0.02 the property carries no information about size, and a size model that does not
beat the marginal is not a neutral addition, it is a second thing to keep in sync that buys nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import label_graphs, property_fn, require_matching_vocabulary, zinc_vocabulary  # noqa: E402

from rdkit import RDLogger  # noqa: E402

from defog_adapt.core import DeFoGModel, PropertyHead, fit_property_head, fit_size_model  # noqa: E402
from defog_adapt.core.data import to_dense  # noqa: E402
from defog_adapt.data import vocabulary, zinc_reference  # noqa: E402
from defog_adapt.domains.molecule import build_encoders  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def fit_size(args, values, node_counts, fit_idx, base, device, out):
    """Fit and save ``P(n | property)`` beside the head.

    The grid ceiling is the base's ``max_nodes``: a size model must not put mass on a graph the base
    cannot build.
    """
    y = values[fit_idx].astype("float32")
    n = node_counts[fit_idx]
    ceiling = int(getattr(base, "max_nodes", int(n.max())))
    low, high = int(n.min()), min(int(n.max()), ceiling)
    keep = n <= high
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} molecules above the base's max_nodes={ceiling}")
    start = time.time()
    model, metrics = fit_size_model(
        torch.from_numpy(y[keep]), torch.from_numpy(n[keep]), min_size=low, max_size=high,
        hidden=args.size_hidden, layers=args.size_layers, epochs=args.size_epochs, lr=args.size_lr,
        seed=args.seed, device=device, property_name=args.property, property_from=args.property_from)
    size_out = out.replace(".ckpt", "_size.ckpt")
    model.save(size_out)
    print(f"size model: grid {low}..{high}, {sum(p.numel() for p in model.parameters()):,} "
          f"parameters, {time.time() - start:.0f}s")
    print(f"  held-out NLL {metrics['nll_val']:.4f} vs marginal {metrics['nll_marginal']:.4f}  "
          f"-> gain {metrics['gain_nats']:+.4f} nats   shrink {metrics['shrink']:.3f}")
    if metrics["gain_nats"] < 0.02:
        print("WARNING: this property carries essentially no information about graph size. Do NOT use "
              "the learned size draw for it; the marginal is as good and has no moving parts.")
    print(f"saved {size_out}")
    return {k: v for k, v in metrics.items() if k != "history"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="frozen base checkpoint (with or without .ckpt)")
    ap.add_argument("--property", required=True, help="logp | tpsa | qed | sascore")
    ap.add_argument("--property-from", default="decoded", choices=("source", "decoded"))
    ap.add_argument("--out", required=True, help="output checkpoint, with or without .ckpt")
    ap.add_argument("--data-root", default=None, help="reference data directory (default: the cache)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=5000,
                    help="molecules held out to report an honest error. The fit reports training "
                         "loss, which says nothing about whether the head generalises, and a head "
                         "that does not is an energy that steers toward its own errors.")
    ap.add_argument("--limit", type=int, default=None, help="cap the molecules (smoke runs)")
    ap.add_argument("--with-size-model", action="store_true",
                    help="also fit P(n | property) -> <out>_size.ckpt")
    ap.add_argument("--size-only", action="store_true",
                    help="fit ONLY the size model (a retrofit for an existing head)")
    ap.add_argument("--size-hidden", type=int, default=512)
    ap.add_argument("--size-layers", type=int, default=2)
    ap.add_argument("--size-epochs", type=int, default=200)
    ap.add_argument("--size-lr", type=float, default=1e-3)
    args = ap.parse_args()

    prop = property_fn(args.property)
    atom_types, bond_types, kekulize = zinc_vocabulary()
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atom_types, bond_types)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = DeFoGModel.load(args.base, device="cpu")
    # The head's input width IS the base's class counts. A mismatch trains a head on channels that
    # mean different elements, which converges and is useless.
    print(vocabulary.check_model(base, atom_types, bond_types, what=f"base {args.base}"))
    na, nb = int(base.output_dims["X"]), int(base.output_dims["E"])
    print(f"{na} atom / {nb} edge classes; property={args.property} from={args.property_from}")

    smiles = list(zinc_reference.load_reference_split(args.data_root).train_smiles)
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"source molecules: {len(smiles)}")

    start = time.time()
    graphs, values, n_skipped = label_graphs(
        smiles, prop, atom_encoder=atom_enc, bond_encoder=bond_enc, atom_decoder=atom_dec,
        bond_decoder=bond_dec, kekulize=kekulize, property_from=args.property_from)
    node_counts = np.asarray([int(g.x.size(0)) for g in graphs])
    print(f"{len(graphs)} graphs (skipped {n_skipped}) in {time.time() - start:.0f}s; "
          f"{args.property} mean={values.mean():.4f} std={values.std():.4f} "
          f"range=[{values.min():.4f}, {values.max():.4f}]")
    try:
        require_matching_vocabulary(n_skipped, n_skipped + len(graphs))
    except RuntimeError as err:
        sys.exit(f"REFUSING: {err}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(graphs))
    n_hold = min(args.holdout, len(graphs) // 5)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    fit_graphs = [graphs[i] for i in fit_idx]
    hold_graphs = [graphs[i] for i in hold_idx]
    print(f"fit on {len(fit_graphs)}, held out {len(hold_graphs)}")

    out = args.out if args.out.endswith(".ckpt") else args.out + ".ckpt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    size_meta = None
    if args.with_size_model or args.size_only:
        size_meta = fit_size(args, values, node_counts, fit_idx, base, device, out)
    if args.size_only:
        print("--size-only: skipping the head.")
        return 0

    head = PropertyHead(na, nb, hid=args.hidden, layers=args.layers,
                        prop_mean=float(values[fit_idx].mean()),
                        prop_std=float(values[fit_idx].std())).to(device)
    print(f"head: {sum(p.numel() for p in head.parameters()):,} parameters "
          f"(hidden {args.hidden}, {args.layers} layers)")
    start = time.time()
    fit_property_head(head, fit_graphs, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                      seed=args.seed, device=device)
    print(f"fit done in {(time.time() - start) / 60:.1f} min")

    # Held-out error in the property's own units. This is the number that says whether the head is
    # usable as an energy; the training loss does not.
    head.eval()
    from torch_geometric.loader import DataLoader

    predicted, truth = [], []
    with torch.no_grad():
        for batch in DataLoader(hold_graphs, batch_size=256, shuffle=False):
            dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(mask)
            pred = head.predict(dense.X.to(device), dense.E.to(device), mask.to(device))
            predicted.append(pred.detach().cpu().numpy().reshape(-1))
            truth.append(batch.cond.numpy().reshape(-1))
    predicted, truth = np.concatenate(predicted), np.concatenate(truth)
    mae = float(np.abs(predicted - truth).mean())
    corr = float(np.corrcoef(predicted, truth)[0, 1])
    spread = float(values.std())
    print(f"held-out MAE {mae:.4f}   corr {corr:.4f}   (property std {spread:.4f}, "
          f"so MAE/std = {mae / spread:.3f})")
    if mae >= spread:
        print("WARNING: the error is not better than predicting the dataset mean. This head carries "
              "no signal and would make the Feynman-Kac energy a no-op.")

    # PropertyHead.save / .load own the file format; do not hand-roll a dict here.
    head.save(out)
    meta = {"property": args.property, "property_from": args.property_from, "base": args.base,
            "seed": args.seed, "n_fit": len(fit_graphs), "n_holdout": len(hold_graphs),
            "holdout_mae": mae, "holdout_corr": corr, "property_std": spread,
            "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden, "layers": args.layers,
            "size_model": size_meta}
    Path(out.replace(".ckpt", ".json")).write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
