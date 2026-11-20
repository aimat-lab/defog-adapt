"""Helpers shared by the example recipes.

One copy, on purpose. The adapter, the property head, the learned size model and the RL fine-tune
all have to label molecules the same way, and three things must agree between them or training
converges happily and the result is useless:

* the vocabulary: the ORDER of the atom and bond classes, which is the base model's channel order;
* the kekulize flag;
* which molecule a label describes (``property_from``, below).

A second copy of any of this is a place for it to drift, and a drifted property function produces
a head that is confidently wrong.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

from defog_adapt.data import zinc_reference
from defog_adapt.domains.molecule import (
    build_encoders, mol_to_smiles, pyg_data_to_mol, smiles_to_pyg_data,
)


# ============================================================================ properties
def _sa_score(mol) -> float:
    """Synthetic accessibility, roughly 1 (easy) to 10 (hard).

    RDKit ships the scorer under ``Contrib``, outside the importable package tree, so the path is
    added here rather than left to fail at first use, hours into a run. Unlike logP and TPSA it is
    a heuristic over fragment frequencies, bounded in practice to about [1, 8] on drug-like
    molecules and NOT symmetric: most of ZINC sits near 2-3.
    """
    global _SASCORER
    if _SASCORER is None:
        from rdkit import RDConfig

        path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if path not in sys.path:
            sys.path.append(path)
        import sascorer

        _SASCORER = sascorer
    return float(_SASCORER.calculateScore(mol))


_SASCORER = None

#: The four properties with a recorded adapter result. QED is bounded in [0, 1] and its mass sits
#: mid-range, unlike logP and TPSA, so a low error on it can be earned by regressing to the middle:
#: judge it across the range, not on the mean.
PROPERTY_FNS: Dict[str, Callable] = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
    "qed": lambda m: float(QED.qed(m)),
    "sascore": _sa_score,
}


def property_fn(name: str) -> Callable:
    if name not in PROPERTY_FNS:
        raise ValueError(f"unknown property {name!r}; have {sorted(PROPERTY_FNS)}")
    return PROPERTY_FNS[name]


# ============================================================================ vocabulary
def zinc_vocabulary() -> Tuple[List[str], List[str], bool]:
    """``(atom_types, bond_types, kekulize)`` of the ZINC base used throughout.

    Nine elements in the reference split's own order, three kekulized bond types. The atom order
    is not a free choice: a model's output channels are indexed by it, so an adapter trained
    against a different order is trained against classes that decode to different elements.
    """
    return list(zinc_reference.ATOM_TYPES), list(zinc_reference.BOND_TYPES), True


# ============================================================================ labelled graphs
def decoded_molecule(data, atom_decoder, bond_decoder):
    """The molecule a GRAPH represents: decode it, round-trip it through SMILES, parse it.

    ``None`` when the graph does not decode to a molecule. This is the molecule a ``"decoded"`` label
    describes (see :func:`label_graphs`), and the one every generated sample is scored on, so a label and
    a score always refer to the same thing.
    """
    decoded = pyg_data_to_mol(data, atom_decoder, bond_decoder)
    back = mol_to_smiles(decoded) if decoded is not None else None
    return Chem.MolFromSmiles(back) if back else None


def attach_condition(
    graphs: Sequence,
    smiles: Optional[Sequence[str]],
    prop_fn: Callable,
    *,
    atom_decoder,
    bond_decoder,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[list, Optional[list], float, float, int]:
    """Label already-encoded graphs for END-TO-END conditional training.

    Returns ``(graphs, smiles, mean, std, n_skipped)``. Each kept graph gets ``graph.y``, a ``(1, 1)``
    tensor holding the STANDARDISED property of the molecule the graph represents (the ``"decoded"``
    convention every recorded adapter, head and size model used). ``y`` is where a ``DeFoGModel`` with
    ``cond_dim > 0`` reads its condition, and the model expects it already standardised.

    A conditional model, unlike an adapter, stores no statistics: ``mean`` and ``std`` belong to the
    caller, who must keep them to condition the model at sampling time. Pass the TRAIN split's values
    when labelling a validation split; left out, they are computed from the graphs given (``std`` is
    the population one, as in ``adapter_training__zinc.py``).

    A graph whose property cannot be measured is dropped from ``graphs`` and ``smiles`` together, so the
    two stay aligned one for one (``smiles`` may be ``None``, and is then returned as ``None``).
    """
    kept_graphs, kept_smiles, values, n_skipped = [], [], [], 0
    for position, graph in enumerate(graphs):
        mol = decoded_molecule(graph, atom_decoder, bond_decoder)
        try:
            value = float(prop_fn(mol)) if mol is not None else float("nan")
        except Exception:                            # noqa: BLE001 - a property may fail on odd molecules
            value = float("nan")
        if not np.isfinite(value):
            n_skipped += 1
            continue
        kept_graphs.append(graph)
        values.append(value)
        if smiles is not None:
            kept_smiles.append(smiles[position])
    if not values:
        raise RuntimeError("no graph could be labelled: the vocabulary and the data disagree")
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if mean is None else float(mean)
    std = float(values.std()) if std is None else float(std)
    if not std > 0:
        raise ValueError(f"the standard deviation of the condition is {std}; nothing to condition on")
    for graph, value in zip(kept_graphs, values):
        graph.y = torch.tensor([[(value - mean) / std]], dtype=torch.float)
    return kept_graphs, (kept_smiles if smiles is not None else None), mean, std, n_skipped


def accumulation_plan(batch_size: int, micro_batch_size: Optional[int],
                      ema_decay: Optional[float]) -> Tuple[int, int, Optional[float]]:
    """Fit a recipe's batch size onto a smaller GPU without changing the recipe.

    Returns ``(micro_batch, accumulate, ema_decay_per_micro_batch)``. ``batch_size`` is what the optimizer
    sees per step, as recorded; ``micro_batch_size`` is what is put on the GPU at once (``None``: all of it).
    The optimizer steps once per ``accumulate = batch_size // micro_batch`` micro-batches, so the batch size,
    the number of steps per epoch and the learning-rate schedule are all unchanged: a memory knob, not a
    different recipe.

    The EMA callback updates after every batch it is given, which is every MICRO-batch. Parameters only move
    at optimizer steps, so ``accumulate`` updates in a row average the same parameters; decaying each by
    ``ema_decay ** (1 / accumulate)`` makes them add up to the one ``ema_decay`` update per optimizer step
    that the recipe specifies. Left alone, the average would silently run ``accumulate`` times shorter.
    """
    micro = batch_size if not micro_batch_size else int(micro_batch_size)
    if micro <= 0 or micro > batch_size or batch_size % micro:
        raise ValueError(f"MICRO_BATCH_SIZE must divide BATCH_SIZE ({batch_size}); got {micro_batch_size}")
    accumulate = batch_size // micro
    decay = float(ema_decay) ** (1.0 / accumulate) if ema_decay else ema_decay
    return micro, accumulate, decay


#: Where an end-to-end conditional model keeps what it needs to be conditioned: the property's name and
#: the mean and std that standardised it. The model itself stores neither.
CONDITION_FILE = "condition.json"


def write_condition_sidecar(directory, stats) -> str:
    """Write the condition's statistics beside the checkpoints, or check them against the ones there.

    The model does not store them, and a conditional model cannot be sampled without them. A chained run
    recomputes them on every link; they have to come out the same, or the links are not training one model.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, CONDITION_FILE)
    if os.path.exists(path):
        old = read_condition_sidecar(path)
        same = (old["property"] == stats["property"]
                and all(math.isclose(old[k], stats[k], rel_tol=1e-6, abs_tol=1e-9) for k in ("mean", "std")))
        if not same:
            raise RuntimeError(
                f"{path} holds the condition statistics of a different run ({old['property']}: mean "
                f"{old['mean']:.6f}, std {old['std']:.6f}) than this one ({stats['property']}: mean "
                f"{stats['mean']:.6f}, std {stats['std']:.6f}). A chain link must continue the same model.")
        return path
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    os.replace(tmp, path)                                 # rename only when complete
    return path


def read_condition_sidecar(path) -> dict:
    """The statistics ``write_condition_sidecar`` wrote: ``property``, ``property_from``, ``mean``, ``std``."""
    with open(path, encoding="utf-8") as fh:
        stats = json.load(fh)
    missing = [k for k in ("property", "property_from", "mean", "std") if k not in stats]
    if missing:
        raise ValueError(f"{path} is not a condition file: missing {missing}")
    return stats


def label_graphs(
    smiles: Sequence[str],
    prop_fn: Callable,
    *,
    atom_encoder,
    bond_encoder,
    atom_decoder,
    bond_decoder,
    kekulize: bool,
    property_from: str = "decoded",
    keep_graphs: bool = True,
) -> Tuple[list, np.ndarray, int]:
    """Encode molecules and measure their property: ``(graphs, values, n_skipped)``.

    Each graph carries the RAW (unnormalised) value as a ``(1, 1)`` tensor in ``.cond``.

    ``property_from`` picks which molecule the label describes:

    ``"source"``   the input SMILES exactly as given.
    ``"decoded"``  the molecule the GRAPH represents (encode, decode, measure).

    They differ because a graph stores atoms and bonds but NOT formal charges or stereochemistry.
    A third of ZINC carries a formal charge, and protonated amines and carboxylates are exactly
    what make a molecule low-logP, so dropping the charge moves logP up, and only at the low
    end: at the 5th percentile the source and graph values differ by about 1.6 log units. Trained
    on source labels, the adapter sees (graph, label) pairs whose low-end labels describe
    something the graph is not. ``"decoded"`` does not give the model new reach; it makes the
    declared range honest, so a requested target means what it says. Every recorded adapter,
    head and size model used ``"decoded"``, and they must all agree on it.
    """
    if property_from not in ("source", "decoded"):
        raise ValueError(f"property_from must be 'source' or 'decoded', got {property_from!r}")
    graphs, values, n_skipped = [], [], 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skipped += 1
            continue
        data = smiles_to_pyg_data(smi, atom_encoder, bond_encoder, kekulize=kekulize)
        if data is None:
            n_skipped += 1
            continue
        target = mol
        if property_from == "decoded":
            target = decoded_molecule(data, atom_decoder, bond_decoder)
            if target is None:                       # cannot measure what the graph is
                n_skipped += 1
                continue
        try:
            value = prop_fn(target)
        except Exception:                            # noqa: BLE001 - a property may fail on odd molecules
            n_skipped += 1
            continue
        if keep_graphs:
            data.cond = torch.tensor([[value]], dtype=torch.float)
            graphs.append(data)
        values.append(value)
    return graphs, np.asarray(values, dtype=float), n_skipped


def require_matching_vocabulary(n_skipped: int, n_total: int, what: str = "molecules") -> None:
    """Refuse to continue if a meaningful fraction failed to encode.

    Under a vocabulary that fits the data this is near zero. Anything more means the vocabulary
    and the data disagree, and the model would be trained on a quietly filtered dataset.
    """
    if n_skipped > 0.02 * max(1, n_total):
        raise RuntimeError(
            f"{n_skipped} of {n_total} {what} failed to encode or measure. Under a matching "
            f"vocabulary this should be near zero; check the vocabulary against the base.")


# ============================================================================ sampling + scoring
def sample_in_chunks(sampler, n: int, chunk: int, device, *, size_dist=None, condition=None) -> list:
    """Draw ``n`` graphs in batches of ``chunk``.

    ``size_dist`` and ``condition`` are forwarded to the sampler. Leaving them out does not raise,
    it silently falls back to the dataset's size marginal, which scores a DIFFERENT policy than the
    one being asked about whenever a learned size model is in play.
    """
    out, done = [], 0
    while done < n:
        k = min(chunk, n - done)
        cond = condition
        if condition is not None and condition.shape[0] > 1:      # per-sample rows: slice the chunk
            cond = condition[done: done + k]                      # (one row broadcasts as it is)
        out += sampler.sample(k, size_dist=size_dist, condition=cond, device=device,
                              show_progress=False)
        done += k
    return out


def decode_valid(samples, atom_decoder, bond_decoder) -> List["Chem.Mol"]:
    """RDKit molecules for the samples that decode to a valid molecule (round-tripped via SMILES)."""
    mols = []
    for graph in samples:
        mol = pyg_data_to_mol(graph, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        parsed = Chem.MolFromSmiles(smi) if smi is not None else None
        if parsed is not None:
            mols.append(parsed)
    return mols


def properties_of(samples, atom_decoder, bond_decoder, prop_fn) -> np.ndarray:
    """The property of each valid sample, as an array (invalid samples are dropped)."""
    values = []
    for mol in decode_valid(samples, atom_decoder, bond_decoder):
        try:
            values.append(prop_fn(mol))
        except Exception:                            # noqa: BLE001
            pass
    return np.asarray(values, dtype=float)


def mean_abs_error(values: np.ndarray, target: float) -> Optional[float]:
    return float(np.mean(np.abs(values - target))) if values.size else None
