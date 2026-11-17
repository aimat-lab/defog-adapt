"""Energies for Feynman-Kac steering that score a predicted clean graph.

An energy is ``energy_fn(X1, E1, node_mask) -> (K,)``, lower is better; the FK potential is
``phi = -beta * energy`` (see :class:`~defog_adapt.core.feynman_kac.FeynmanKacSampler`).
:class:`~defog_adapt.core.property_head.LearnedPropertyEnergy` is the learned counterpart of
:class:`MoleculePropertyEnergy` for properties without a closed form, and
:class:`StandardisedEnergy` is what makes ``beta`` mean the same thing for either.
"""

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from .data import dense_to_pyg


class MoleculePropertyEnergy:
    """Energy for molecular property steering: ``E(x1) = (prop(mol) - target)^2``.

    Decodes each dense one-hot graph to an RDKit molecule via a duck-typed
    ``domain`` (anything with ``.decode(pyg_data) -> Optional[Mol]``, e.g.
    :class:`~defog_adapt.domains.molecule.MoleculeDomain`) and evaluates ``prop_callback``
    (e.g. ``rdkit.Chem.Crippen.MolLogP``). Molecules that fail to decode get
    ``invalid_energy`` (so ``r ~ 0`` -- they lie off the source support). Depends
    only on ``x1``, so results can be cached per molecule.
    """

    def __init__(self, domain, prop_callback: Callable, target: float,
                 invalid_energy: float = None, *,
                 invalid_z2: float = 25.0, scale: float = 1.0):
        self.domain, self.prop = domain, prop_callback
        self.scale = float(scale)
        # Same contract as LearnedPropertyEnergy: the sentinel is a flag in squared
        # standard deviations, pre-multiplied so an outer 1/scale**2 leaves it invariant.
        self.target = target
        self.invalid = (float(invalid_energy) if invalid_energy is not None
                        else float(invalid_z2) * self.scale ** 2)

    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), float(self.invalid))
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is not None:
                try:
                    out[i] = (float(self.prop(mol)) - self.target) ** 2
                except Exception:
                    pass
        return out


Term = Union[Tuple[Callable, float], Tuple[Callable, float, float]]


class StandardisedEnergy:
    """A weighted sum of property energies, each divided by its property's variance.

    ``total(x1) = sum_i weight_i * energy_i(x1) / scale_i ** 2``

    Use this to hand an energy to :class:`~defog_adapt.core.feynman_kac.FeynmanKacSampler`.
    ``MoleculePropertyEnergy`` and ``LearnedPropertyEnergy`` return a squared error in the
    property's OWN units, and the FK potential is ``-beta * energy``, so with a raw energy
    ``beta`` means "per logP-unit squared" for one property and "per QED-unit squared" for
    another. A ``beta`` tuned on logP (std 1.16) then carries about 76 times less pressure on
    QED (std 0.14): the particle weights stay close to uniform, the effective-sample-size
    gate never fires, and a ``beta`` sweep reads as "Feynman-Kac does not help this property"
    when it never engaged. Dividing by the variance makes the energy a squared z-score, so
    ``beta`` is dimensionless and carries from one property to the next. The recorded runs
    (``defog_adapt.presets.FK_STEERING``) are in these units.

    Args:
        terms: ``(energy, scale)`` or ``(energy, scale, weight)`` tuples. ``scale`` is the
            property's standard deviation (``head.prop_std`` for a learned head, or measured
            over the training set); ``weight`` defaults to 1 and only matters for the
            RELATIVE importance of several properties -- for one property it is the same as
            rescaling ``beta``.

    ``invalid`` is the sentinel an undecodable graph receives, after normalisation, and is only
    defined for a single term. Both energy classes state theirs in squared standard deviations
    and pre-multiply it by ``scale ** 2`` for exactly this reason, so it comes out here as the
    same number for every property instead of being rescaled by the property's units.
    """

    def __init__(self, terms: Sequence[Term]):
        if not terms:
            raise ValueError("StandardisedEnergy needs at least one (energy, scale) term")
        parsed = []
        for k, term in enumerate(terms):
            if len(term) == 2:
                energy, scale = term
                weight = 1.0
            elif len(term) == 3:
                energy, scale, weight = term
            else:
                raise ValueError(f"term {k} must be (energy, scale) or (energy, scale, weight), "
                                 f"got a {len(term)}-tuple")
            scale, weight = float(scale), float(weight)
            if not callable(energy):
                raise TypeError(f"term {k}: the energy must be callable (X1, E1, node_mask)")
            if not (math.isfinite(scale) and scale > 0):
                raise ValueError(f"term {k}: scale must be a positive finite standard "
                                 f"deviation, got {scale!r}")
            if not math.isfinite(weight):
                raise ValueError(f"term {k}: weight must be finite, got {weight!r}")
            parsed.append((energy, scale, weight))
        self.terms = parsed

    @property
    def invalid(self) -> Optional[float]:
        """The normalised invalid-graph sentinel, or ``None`` for a sum of several terms."""
        if len(self.terms) != 1:
            return None
        energy, scale, weight = self.terms[0]
        raw = getattr(energy, "invalid", None)
        return None if raw is None else float(raw) * weight / (scale * scale)

    def __call__(self, X1: torch.Tensor, E1: torch.Tensor, node_mask: torch.Tensor):
        total = None
        for energy, scale, weight in self.terms:
            value = energy(X1, E1, node_mask) * (weight / (scale * scale))
            total = value if total is None else total + value
        return total
