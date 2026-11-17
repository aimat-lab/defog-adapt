"""Energies for Feynman-Kac steering that score a predicted clean graph.

An energy is ``energy_fn(X1, E1, node_mask) -> (K,)``, lower is better; the FK potential is
``phi = -beta * energy`` (see :class:`~defog_adapt.core.feynman_kac.FeynmanKacSampler`).
:class:`~defog_adapt.core.property_head.LearnedPropertyEnergy` is the learned counterpart of
:class:`MoleculePropertyEnergy` for properties without a closed form.
"""

from typing import Callable

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
