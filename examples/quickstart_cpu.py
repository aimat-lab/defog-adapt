"""CPU quickstart: every building block, end to end, on a toy problem, in about a minute.

NOTHING HERE IS A RESULT. The data is a few dozen hand-written molecules and the models are tiny,
so the generated molecules are poor and the steering is weak. What this checks is that the pieces
fit together, and it is the shortest honest tour of the API:

    1. a base DeFoG model             (DeFoGModel, trained unconditionally)
    2. a frozen-base adapter          (CrossAttentionAdapter + AdapterModule, conditioned on logP)
    3. a property head                (PropertyHead, a graph -> logP regressor: the FK "ruler")
    4. a learned size model           (fit_size_model, P(n | logP))
    5. conditional sampling           (AdaptedSampler over an AdapterComposition)
    6. Feynman-Kac steering           (FeynmanKacSampler + StandardisedEnergy)
    7. decoding and scoring           (pyg_data_to_mol, RDKit)

The real pipeline is the same code at scale; see ``examples/README.md`` for the recipes that train
each piece on ZINC250k, and ``examples/steer_zinc.py`` for the assembly with real checkpoints.

    pip install "defog-adapt[chem]"
    python examples/quickstart_cpu.py
    python examples/quickstart_cpu.py --smoke     # seconds; checks only that it runs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import label_graphs, mean_abs_error, properties_of, property_fn  # noqa: E402

from defog_adapt.core import (  # noqa: E402
    CrossAttentionAdapter, AdaptedSampler, AdapterComposition, AdapterModule, ConditionBranch, DeFoGModel,
    FeynmanKacSampler, LearnedPropertyEnergy, PropertyHead, StandardisedEnergy, fit_property_head,
    fit_size_model,
)
from defog_adapt.domains import MoleculeDomain  # noqa: E402
from defog_adapt.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles  # noqa: E402
from defog_adapt.presets import ADAPTER_WEIGHT, FK_STEERING  # noqa: E402

RDLogger.DisableLog("rdApp.*")

#: Small molecules over C, N and O. Enough variety in size and polarity for logP to vary.
SMILES = """
CCO CCCO CCCCO CCCCCO CCCCCCO CCN CCCN CCCCN CCCCCN CCOC CCCOC CCOCC CCCOCC CCCCOCC
CC(C)O CC(C)N CC(C)C CCC(C)C CCCC(C)C CC(C)CO CC(C)CC CC(=O)C CCC(=O)C CCCC(=O)C CC(=O)CC
CC(=O)O CCC(=O)O CCCC(=O)O CC(=O)N CCC(=O)N CCCC(=O)N CC(=O)OC CCC(=O)OC CC(=O)OCC
NCCO NCCN OCCO OCCCO NCCCN OCC(O)CO NCC(O)CO CNC CCNCC CCCNCCC CN(C)C CCN(CC)CC
C1CCCCC1 C1CCCC1 C1CCNCC1 C1CCOCC1 C1CCOC1 C1CCNC1 C1CNCCN1 C1COCCN1 OC1CCCCC1 NC1CCCCC1
c1ccccc1 Cc1ccccc1 CCc1ccccc1 Oc1ccccc1 Nc1ccccc1 COc1ccccc1 CC(=O)c1ccccc1 OCc1ccccc1
c1ccncc1 Cc1ccncc1 c1ccoc1 c1cnccn1 Cc1ccccc1O Cc1ccccc1N OC(=O)c1ccccc1 NC(=O)c1ccccc1
""".split()

ATOMS, BONDS, KEKULIZE = ["C", "N", "O"], ["SINGLE", "DOUBLE", "TRIPLE"], True


def fit_trainer(max_epochs: int) -> pl.Trainer:
    return pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      enable_checkpointing=False, gradient_clip_val=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="a few seconds; only checks that it runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    smoke = args.smoke
    pl.seed_everything(args.seed, workers=True)
    t0 = time.time()

    # ------------------------------------------------------------------ data
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(ATOMS, BONDS)
    prop = property_fn("logp")
    # property_from="decoded": the label describes the molecule the GRAPH represents, which is the
    # convention every recorded adapter, head and size model used (see examples/_common.py).
    graphs, logp, skipped = label_graphs(
        SMILES, prop, atom_encoder=atom_enc, bond_encoder=bond_enc, atom_decoder=atom_dec,
        bond_decoder=bond_dec, kekulize=KEKULIZE, property_from="decoded")
    mean, std = float(logp.mean()), float(logp.std())
    low, high = (float(x) for x in np.percentile(logp, [10, 90]))
    print(f"{len(graphs)} molecules ({skipped} skipped); logP mean {mean:.2f} std {std:.2f}; "
          f"targets: low {low:.2f}, high {high:.2f}")
    loader = DataLoader(graphs, batch_size=16, shuffle=True)

    # ------------------------------------------------------------------ 1. the base model
    base = DeFoGModel.from_dataloader(
        loader, n_layers=2, hidden_dim=32, hidden_mlp_dim=64, n_heads=2, dropout=0.0,
        noise_type="marginal", extra_features_type="rrwp", rrwp_steps=4, molecular_features=False,
        lr=2e-3, sample_time_distortion="polydec", train_time_distortion="polydec")
    fit_trainer(4 if smoke else 120).fit(base, loader)
    base.eval()
    print(f"[{time.time() - t0:4.0f}s] base: {sum(p.numel() for p in base.parameters()):,} parameters")

    # ------------------------------------------------------------------ 2. a frozen-base adapter
    # Only the adapter trains; the base is frozen, so the unconditional model stays exactly as it
    # was and adapters for different properties can be swapped and composed.
    adapter = CrossAttentionAdapter.for_base(
        base, cond_dim=1, hidden=32, cond_fourier=3, xattn_tokens=8, xattn_dim=16, xattn_heads=2,
        cond_mean=[mean], cond_std=[std], name="logp_adapter", cond_type="logp")
    fit_trainer(4 if smoke else 120).fit(AdapterModule(base, adapter, lr=2e-3), loader)
    adapter.eval()
    print(f"[{time.time() - t0:4.0f}s] adapter: {sum(p.numel() for p in adapter.parameters()):,} "
          f"parameters (the base is frozen)")

    # ------------------------------------------------------------------ 3. a property head
    # A graph -> logP regressor, fitted with no adapter in the loop. FK uses it to score each
    # predicted clean molecule, so it is the ruler the steering is measured against.
    head = PropertyHead(int(base.output_dims["X"]), int(base.output_dims["E"]), hid=32, layers=2,
                        prop_mean=mean, prop_std=std)
    fit_property_head(head, graphs, epochs=3 if smoke else 60, lr=2e-3, batch_size=16, seed=0)
    head.eval()

    # ------------------------------------------------------------------ 4. a learned size model
    sizes = torch.tensor([int(g.x.shape[0]) for g in graphs])
    size_model, info = fit_size_model(
        torch.tensor(logp, dtype=torch.float32), sizes, min_size=int(sizes.min()),
        max_size=int(min(sizes.max(), base.max_nodes)), hidden=32, layers=2,
        epochs=5 if smoke else 150, lr=2e-3, seed=0, property_name="logp",
        property_from="decoded")
    print(f"[{time.time() - t0:4.0f}s] size model gain over the marginal: {info['gain_nats']:+.3f} nats")

    # ------------------------------------------------------------------ 5-7. sample, steer, score
    domain = MoleculeDomain(ATOMS, bond_dec, reference_smiles=SMILES)
    steps, n, weight = (6, 8, ADAPTER_WEIGHT["logp"]) if smoke else (60, 32, ADAPTER_WEIGHT["logp"])
    ctmc = dict(sample_steps=steps, eta=5.0, omega=0.0, time_distortion="polydec")
    device = torch.device("cpu")

    def run(label: str, target: float, *, use_size: bool, use_fk: bool) -> dict:
        composition = AdapterComposition(
            [ConditionBranch(adapter, torch.tensor([target]), weight)], base=base, mode="product")
        # One row broadcasts to every particle; per-particle rows work too.
        size_kwargs = dict(size_dist=size_model, condition=torch.tensor([[target]])) if use_size else {}
        if use_fk:
            # The energy is a squared error in logP units; StandardisedEnergy divides it by the
            # variance so beta is dimensionless and means the same for any property.
            energy = StandardisedEnergy([(
                LearnedPropertyEnergy(head, target, domain, atom_enc, bond_enc, scale=std), std)])
            sampler = FeynmanKacSampler(base, energy, composition=composition, **FK_STEERING, **ctmc)
        else:
            sampler = AdaptedSampler(base, composition, **ctmc)
        samples = sampler.sample(n, device=device, show_progress=False, **size_kwargs)
        values = properties_of(samples, atom_dec, bond_dec, prop)
        return {"label": label, "target": target, "valid": values.size / n,
                "mean": float(values.mean()) if values.size else float("nan"),
                "mae": mean_abs_error(values, target), "samples": samples}

    print(f"\n{'':<26}{'target':>8}{'valid':>8}{'mean':>8}{'MAE':>8}")
    torch.manual_seed(args.seed)
    rows = []
    for target in (low, high):
        for label, use_size, use_fk in (("adapter", False, False), ("adapter + size model", True, False),
                                        ("adapter + size + FK", True, True)):
            row = run(label, target, use_size=use_size, use_fk=use_fk)
            rows.append(row)
            mae = "   n/a" if row["mae"] is None else f"{row['mae']:8.2f}"
            print(f"{row['label']:<26}{target:8.2f}{row['valid']:8.2f}{row['mean']:8.2f}{mae}")

    best = next(r for r in rows if r["label"].endswith("FK") and r["target"] == high)
    shown = []
    for graph in best["samples"]:
        mol = pyg_data_to_mol(graph, atom_dec, bond_dec)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi:
            shown.append(smi)
        if len(shown) == 5:
            break
    print(f"\nsteered towards logP {high:.2f}: {', '.join(shown) if shown else '(none valid)'}")
    print(f"[{time.time() - t0:4.0f}s] done.")
    print("Read this as a wiring check, not a result: with a few dozen molecules, a tiny model and "
          f"{n} samples per row,\nthe differences between rows are inside sampling noise (and a head "
          "this poor is a poor ruler for FK).\nThe recorded numbers, at scale, are in docs/about/results-and-caveats.md.")


if __name__ == "__main__":
    main()
