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
            decoded = pyg_data_to_mol(data, atom_decoder, bond_decoder)
            back = mol_to_smiles(decoded) if decoded is not None else None
            target = Chem.MolFromSmiles(back) if back else None
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
