"""Train ONE frozen-base adapter for a single scalar molecular property on ZINC250k.

    python examples/adapter_training__zinc.py --BASE_CKPT "'ckpts/zinc_rl2_seed42/best_model'" \\
        --PROPERTY "'logp'"
    python examples/adapter_training__zinc.py --BASE_CKPT "'...'" --PROPERTY "'qed'" --__TESTING__ True

The base (an unconditional ZINC model) is FROZEN. Only the adapter trains, with the base's own
denoising cross-entropy, so a trained adapter is a conditional denoiser p(x1 | x_t, c) that sits on
top of an unchanged unconditional model. That is what keeps adapters swappable (one frozen base,
one small file per property) and composable (two adapters multiply into a joint condition).

The architecture is ``defog_adapt.presets.ADAPTER_ARCHITECTURE``: a FiLM trunk, plus node-to-
condition cross-attention (64 tokens, width 128, 16 heads), plus three Fourier bands on the scalar
condition. Cross-attention is the half that carries the effect (at 60 epochs, -0.153 logP error for
cross-attention against -0.024 for the Fourier bands, -0.192 combined); each atom queries the
condition instead of every atom receiving the same broadcast correction. The training settings
(80 epochs, learning rate 4e-4, batch 24) are one measured recipe: 20 epochs was badly under-trained
for this architecture (0.4299 against 0.3596 at 40 and 0.3250 at 80), and the learning rate must
match what the comparison baselines used or a bigger adapter is trained at half the rate of the
thing it is supposed to beat.

``PROPERTY_FROM`` decides which molecule the label describes, and it matters. A graph stores atoms
and bonds but not formal charges, so the property of the SOURCE SMILES and of the molecule the
GRAPH represents differ, by up to 1.6 log units at the low end of logP (charged amines and
carboxylates are what make a molecule low-logP). Every recorded result used ``decoded``. The head
and the size model that go with this adapter must use the same convention, or the Feynman-Kac
energy pulls against the adapter at exactly the end of the range where it matters.

An adapter cannot widen what the frozen base can reach. The low end of logP is where steering was
weakest with the first (FiLM-only) adapter: the lowest third of targets improved by 2% where the
middle and top thirds improved by 22% and 26%. Part of that is the base being handed the wrong atom
count (a learned size model recovered 8% on that third), and the rest is the base's own reach.

Read an adapter ARCHITECTURE sweep against the seed noise. Two seeds of the identical recipe differ
by 0.058 logP error (0.3068 against 0.3646), which is larger than most of the capacity differences
between configurations; only the step from FiLM-only to cross-attention (-0.19) and the epoch count
(-0.07) clear it.

At the end it evaluates single-property steering: it steers to the 5th and 95th percentile of the
property and reports the achieved mean, the error to the target and the validity, over a guidance-
weight sweep, with the dataset distribution and the generated one plotted together. That is a quick
check on one run, not the 100-target benchmark protocol the recorded results were measured with
(see docs/results.md).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import RDLogger

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    label_graphs, mean_abs_error, properties_of, property_fn, require_matching_vocabulary,
    sample_in_chunks, zinc_vocabulary,
)

from defog_adapt.core import (  # noqa: E402
    AdaLNAdapter, AdaptedSampler, AdapterComposition, AdapterModule, ConditionBranch, DeFoGModel,
    Sampler,
)
from defog_adapt.data import vocabulary, zinc_reference  # noqa: E402
from defog_adapt.domains.molecule import build_encoders  # noqa: E402
from defog_adapt.presets import (  # noqa: E402
    ADAPTER_ARCHITECTURE, ADAPTER_SAMPLING, ADAPTER_TRAINING, ADAPTER_WEIGHT,
)

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Parameters
# ============================================================================
# :param BASE_CKPT: the frozen unconditional base (a checkpoint path). Required. Its class ORDER is
#     the vocabulary the adapter is trained against, and it is checked below.
BASE_CKPT: str = None
# :param PROPERTY: logp | tpsa | qed | sascore. QED is bounded in [0, 1] and its mass sits
#     mid-range, so a low error on it can be earned by regressing to the middle: judge it across the
#     range. SAScore needs RDKit's Contrib scorer, which the pip wheel ships.
PROPERTY: str = "logp"
# :param PROPERTY_FROM: "decoded" (the molecule the graph represents; every recorded result) or
#     "source" (the input SMILES). See the module docstring.
PROPERTY_FROM: str = "decoded"
DATA_ROOT: str = None
DOWNLOAD: bool = False

# --- Adapter architecture (defog_adapt.presets.ADAPTER_ARCHITECTURE) ---
H_HIDDEN: int = ADAPTER_ARCHITECTURE["hidden"]
TIME_CONDITIONED: bool = ADAPTER_ARCHITECTURE["time_conditioned"]
STREAMS: list = list(ADAPTER_ARCHITECTURE["streams"])
# :param COND_FOURIER: Fourier bands on the scalar target. Without them the trunk sees the property
#     as ONE raw float while the flow time gets a 64-dimensional sinusoidal embedding. 3 is the
#     ceiling that keeps neighbouring targets correlated enough to interpolate between. 0 gives the
#     FiLM-only adapter.
COND_FOURIER: int = ADAPTER_ARCHITECTURE["cond_fourier"]
# :param XATTN_TOKENS: node-to-condition cross-attention tokens (0 turns it off).
XATTN_TOKENS: int = ADAPTER_ARCHITECTURE["xattn_tokens"]
XATTN_DIM: int = ADAPTER_ARCHITECTURE["xattn_dim"]
XATTN_HEADS: int = ADAPTER_ARCHITECTURE["xattn_heads"]

# --- Training (defog_adapt.presets.ADAPTER_TRAINING) ---
# These three together are the measured recipe; changing one in isolation does not reproduce it.
# MAX_TIME_HOURS bounds ``trainer.fit`` and nothing else, so it must clear the epoch count or
# training stops early and the run still reports a checkpoint.
EPOCHS: int = ADAPTER_TRAINING["epochs"]
BATCH_SIZE: int = ADAPTER_TRAINING["batch_size"]
LEARNING_RATE: float = ADAPTER_TRAINING["lr"]
GRAD_CLIP: float = ADAPTER_TRAINING["grad_clip"]
MAX_TIME_HOURS: float = 20.0

# --- Evaluation (the benchmark's sampling configuration) ---
EVAL_STEPS: int = ADAPTER_SAMPLING["sample_steps"]
ETA: float = ADAPTER_SAMPLING["eta"]
OMEGA: float = ADAPTER_SAMPLING["omega"]
TIME_DISTORTION: str = ADAPTER_SAMPLING["time_distortion"]
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
# Blending is in probability space, where the measured optimum is well above 1 (about 1.5 for logP,
# TPSA and SAScore and 2.0 for QED, with validity falling away past ~2.5), so the grid straddles it.
GUIDANCE_WEIGHTS: list = [1.0, 1.5, 2.0, 2.5, 3.0]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32

# --- Mid-training probe ---
PROBE_EVERY_K: int = 5
PROBE_N: int = 32
PROBE_STEPS: int = 100
# :param PROBE_WEIGHT: the guidance weight the probe (and the plot) use; None means this property's
#     measured optimum (``presets.ADAPTER_WEIGHT``). Probing at a weight other than the one the
#     adapter ships at is how a capacity comparison ends up reading a weight difference as a gap.
PROBE_WEIGHT: float = None

# :param CKPT_EVERY_K: write the adapter every K epochs (0 disables). The adapter is otherwise
#     written only after ``trainer.fit`` returns, so a run killed by a wall-clock limit loses
#     everything. The saved checkpoints can also be EVALUATED properly after the fact, at whatever
#     weight and sample count a protocol demands: a mid-training probe is not a substitute (probe
#     and final evaluation disagreed by +0.112 and -0.131 on a quantity of about 0.35).
CKPT_EVERY_K: int = 10

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


class AdapterEpochCheckpoint(pl.Callback):
    """Write the adapter to ``{property}_adapter_ep{N}`` every K epochs."""

    def __init__(self, e, adapter, every_k: int, prop: str):
        super().__init__()
        self.e, self.adapter, self.every_k, self.prop = e, adapter, int(every_k), prop
        self.saved = []

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.every_k:
            return
        epoch = int(trainer.current_epoch) + 1            # 1-based: "after N epochs of training"
        if epoch % self.every_k:
            return
        try:
            path = self.adapter.save(os.path.join(self.e.path, f"{self.prop}_adapter_ep{epoch}"))
            self.saved.append({"epoch": epoch, "path": path})
            self.e["checkpoints"] = self.saved
            self.e.log(f"[epoch {epoch}] checkpoint -> {path}")
        except Exception as ex:                            # noqa: BLE001 - never let a save kill the run
            self.e.log(f"[epoch {epoch}] CHECKPOINT FAILED (non-fatal): {ex}")


class AdapterPropProbe(pl.Callback):
    """Per-epoch loss log plus an every-K-epoch steering probe (achieved property when steering to
    the low and high targets), so that training is visible."""

    def __init__(self, e, atom_decoder, bond_decoder, prop_fn, targets, weight, every_k, n, steps, chunk):
        super().__init__()
        self.e, self.ad, self.bd, self.prop_fn = e, atom_decoder, bond_decoder, prop_fn
        self.targets, self.weight, self.every_k = targets, weight, every_k
        self.n, self.steps, self.chunk = n, steps, chunk

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get("adapter/loss_epoch",
                                            trainer.callback_metrics.get("adapter/loss"))
        self.e.log(f"[epoch {epoch}] adapter/loss={float(loss):.4f}" if loss is not None
                   else f"[epoch {epoch}] done")
        if not self.every_k or (epoch + 1) % self.every_k != 0:
            return
        try:
            self._probe(pl_module, epoch)
        except Exception as ex:                            # noqa: BLE001
            self.e.log(f"[epoch {epoch}] PROBE failed (non-fatal): {ex}")

    @torch.no_grad()
    def _probe(self, pl_module, epoch):
        out = {}
        for level, target in self.targets.items():
            comp = AdapterComposition([ConditionBranch(pl_module.adapter, torch.tensor([target]),
                                                       self.weight)], base=pl_module.base,
                                      mode="product")
            sampler = AdaptedSampler(pl_module.base, comp, eta=self.e.ETA, omega=self.e.OMEGA,
                                     sample_steps=self.steps, time_distortion=self.e.TIME_DISTORTION)
            samples = sample_in_chunks(sampler, self.n, self.chunk, pl_module.device)
            values = properties_of(samples, self.ad, self.bd, self.prop_fn)
            out[level] = (float(values.mean()) if values.size else float("nan"), int(values.size))
        self.e.log(f"[epoch {epoch}] PROBE(w={self.weight}) " + "  ".join(
            f"{level}->{target:.1f}: achieved={out[level][0]:.2f} (n={out[level][1]})"
            for level, target in self.targets.items()))


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    if not e.BASE_CKPT:
        raise ValueError("BASE_CKPT is required: the frozen unconditional base")
    e.log(f"frozen-base ADAPTER training, property={e.PROPERTY}")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop = property_fn(e.PROPERTY)
    probe_weight = e.PROBE_WEIGHT if e.PROBE_WEIGHT is not None else ADAPTER_WEIGHT[e.PROPERTY]

    atom_types, bond_types, kekulize = zinc_vocabulary()
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atom_types, bond_types)
    e.log(f"vocabulary: {len(atom_types)} atoms {atom_types}, bonds {bond_types}, kekulize={kekulize}")
    e.log(f"property_from={e.PROPERTY_FROM}" + (
        "  (the label describes the GRAPH; charges are dropped)" if e.PROPERTY_FROM == "decoded"
        else "  (the label describes the SOURCE SMILES; charges included)"))

    # The hash-pinned reference split, not a loose CSV: an adapter trained on molecules the base
    # never saw would be learning to steer a distribution shift as well as the property.
    split = zinc_reference.load_reference_split(e.DATA_ROOT, download=e.DOWNLOAD)
    graphs, vals, n_skipped = label_graphs(
        split.train_smiles, prop, atom_encoder=atom_enc, bond_encoder=bond_enc,
        atom_decoder=atom_dec, bond_decoder=bond_dec, kekulize=kekulize,
        property_from=e.PROPERTY_FROM)
    require_matching_vocabulary(n_skipped, n_skipped + len(graphs))
    cond_mean, cond_std = float(vals.mean()), float(vals.std())
    e.log(f"{len(graphs)} graphs ({n_skipped} skipped); {e.PROPERTY} mean={cond_mean:.2f} "
          f"std={cond_std:.2f}")
    e["encoding"] = {"atom_types": atom_types, "bond_types": bond_types, "kekulize": kekulize,
                     "n_graphs": len(graphs), "n_skipped": n_skipped, "property_from": e.PROPERTY_FROM}

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(graphs, batch_size=e.BATCH_SIZE, shuffle=True)
    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, f"expected an unconditional base, got cond_dim={base.cond_dim}"
    # The adapter modulates the base's own channels, so a vocabulary mismatch would train it against
    # classes that mean something else. That gives a converging loss and a useless adapter, with
    # nothing in between to notice, so check it here.
    e.log(vocabulary.check_model(base, atom_types, bond_types, what=f"base {e.BASE_CKPT}"))

    adapter = AdaLNAdapter.for_base(
        base, cond_dim=1, hidden=e.H_HIDDEN, time_conditioned=e.TIME_CONDITIONED,
        streams=tuple(e.STREAMS), cond_mean=[cond_mean], cond_std=[cond_std],
        cond_fourier=e.COND_FOURIER, xattn_tokens=e.XATTN_TOKENS, xattn_dim=e.XATTN_DIM,
        xattn_heads=e.XATTN_HEADS, name=f"{e.PROPERTY}_adapter", cond_type=e.PROPERTY)
    e["adapter/num_params"] = sum(p.numel() for p in adapter.parameters())
    e.log(f"adapter: {e['adapter/num_params']:,} parameters (cond_fourier={e.COND_FOURIER}, "
          f"xattn {e.XATTN_TOKENS}/{e.XATTN_DIM}/{e.XATTN_HEADS}); base "
          f"{sum(p.numel() for p in base.parameters()):,} frozen")
    module = AdapterModule(base, adapter, cond_attr="cond", lr=e.LEARNING_RATE)

    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    e["eval/targets"] = targets
    e.log(f"targets ({e.PROPERTY}): {targets}")
    callbacks = [AdapterPropProbe(e, atom_dec, bond_dec, prop, targets, probe_weight,
                                  e.PROBE_EVERY_K, e.PROBE_N, e.PROBE_STEPS, e.EVAL_CHUNK)]
    if e.CKPT_EVERY_K:
        callbacks.append(AdapterEpochCheckpoint(e, adapter, e.CKPT_EVERY_K, e.PROPERTY))
    trainer = pl.Trainer(max_epochs=e.EPOCHS, max_time={"hours": e.MAX_TIME_HOURS},
                         accelerator="auto", devices=1, enable_progress_bar=True,
                         enable_checkpointing=False, logger=False,
                         gradient_clip_val=e.GRAD_CLIP, callbacks=callbacks)
    e.log(f"training: epochs<={e.EPOCHS} max_time={e.MAX_TIME_HOURS}h batch={e.BATCH_SIZE} "
          f"lr={e.LEARNING_RATE} ckpt_every={e.CKPT_EVERY_K or 'off'}")
    trainer.fit(module, train_dataloaders=train_loader)

    ckpt = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter"))
    with open(os.path.join(e.path, f"{e.PROPERTY}_adapter_stats.json"), "w") as fh:
        json.dump({"property": e.PROPERTY, "property_from": e.PROPERTY_FROM, "mean": cond_mean,
                   "std": cond_std, "targets": targets, "atom_types": atom_types,
                   "percentiles": {str(p): float(np.percentile(vals, p))
                                   for p in [5, 25, 50, 75, 95]}}, fh)
    e.log(f"saved adapter -> {ckpt}")

    # -- Evaluation: single-property steering -----------------------------------------------
    e.log("=" * 60)
    base, adapter = base.to(device).eval(), adapter.to(device).eval()
    ctmc = dict(eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS, time_distortion=e.TIME_DISTORTION)

    e.log(f"baseline: sampling {e.N_BASELINE} unconditional molecules")
    base_vals = properties_of(
        sample_in_chunks(Sampler(base, **ctmc), e.N_BASELINE, e.EVAL_CHUNK, device),
        atom_dec, bond_dec, prop)
    e.log(f"baseline {e.PROPERTY} mean={base_vals.mean():.2f} (n={base_vals.size})")

    results = {"property": e.PROPERTY, "baseline_mean": float(base_vals.mean()), "targets": targets,
               "per_level": {}}
    generated = {}
    for level, target in targets.items():
        results["per_level"][level] = {"target": target, "per_w": {}}
        for w in e.GUIDANCE_WEIGHTS:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([target]), w)],
                                      base=base, mode="product")
            samples = sample_in_chunks(AdaptedSampler(base, comp, **ctmc), e.N_PER_TARGET,
                                       e.EVAL_CHUNK, device)
            values = properties_of(samples, atom_dec, bond_dec, prop)
            results["per_level"][level]["per_w"][str(w)] = {
                "n_valid": int(values.size), "validity": values.size / max(1, len(samples)),
                "mean": float(values.mean()) if values.size else None,
                "mae": mean_abs_error(values, target)}
            if abs(w - probe_weight) < 1e-9:
                generated[level] = values
            e.log(f"  {level} target={target:.2f} w={w}: n={values.size} "
                  f"mean={values.mean() if values.size else float('nan'):.2f} "
                  f"mae={results['per_level'][level]['per_w'][str(w)]['mae']}")
    e.commit_json("adapter_steering_metrics.json", results)

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = np.percentile(vals, [1, 99])
    bins = np.linspace(lo, hi, 50)
    ax.hist(vals, bins=bins, density=True, color="0.7", label="dataset", zorder=1)
    colors = {"low": "#2c7fb8", "high": "#d95f0e"}
    for level, target in targets.items():
        values = generated.get(level, np.array([]))
        if values.size:
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.5,
                    color=colors.get(level), label=f"generated {level} (mean {values.mean():.1f})",
                    zorder=2)
        ax.axvline(target, ls="--", color=colors.get(level), lw=2,
                   label=f"target {level} = {target:.1f}")
    ax.set(xlabel=e.PROPERTY, ylabel="density",
           title=f"Adapter steering: {e.PROPERTY} (lr {e.LEARNING_RATE}, w={probe_weight})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    e.commit_fig(f"steering_{e.PROPERTY}.png", fig)
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.2
    e.H_HIDDEN = 32
    e.XATTN_TOKENS = 8
    e.XATTN_DIM = 16
    e.XATTN_HEADS = 2
    e.EVAL_STEPS = 5
    e.PROBE_STEPS = 5
    e.PROBE_EVERY_K = 1
    e.CKPT_EVERY_K = 1                    # exercise the checkpoint path: it is the fix for a real loss
    e.PROBE_N = 6
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.GUIDANCE_WEIGHTS = [1.5]
    e.PROBE_WEIGHT = 1.5

    # Truncate the split AFTER it has been loaded and verified, or a "smoke" run encodes all 224k
    # training molecules. BASE_CKPT is left alone: a smoke test should exercise the base the caller
    # passed, since the vocabulary/base agreement check is one of the things worth testing.
    real = zinc_reference.load_reference_split

    def small(*args, **kwargs):
        split = real(*args, **kwargs)
        return zinc_reference.ZincReferenceSplit(
            train_smiles=split.train_smiles[:300], val_smiles=split.val_smiles[:50],
            test_smiles=split.test_smiles[:50],
            provenance={**split.provenance, "TRUNCATED_FOR_SMOKE_TEST": True})

    zinc_reference.load_reference_split = small


experiment.run_if_main()
