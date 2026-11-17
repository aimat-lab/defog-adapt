"""`MolecularFeatures`: the chemistry-aware extra features every recorded molecular model used.

Per atom: CHARGE (nominal valency minus the valency its current bonds use) and VALENCY (the bond
orders it currently has); globally: the normalised molecular WEIGHT. They are computed from the
NOISY graph on every forward pass, in training and in sampling, so they have to be right on
one-hot graphs, on padded batches, and on the dense, off-manifold graphs sampling passes through.

The traps this code documents, each pinned here: bond ORDERS, not class indices (aromatic is 1.5,
not 4); padding must not add phantom atoms' weight; and the valency is capped so a transiently
dense noisy graph cannot feed an amplifying signal back into the network.
"""

import pytest
import torch
import torch.nn.functional as F

from defog_adapt.core.features import MolecularFeatures

# Atom classes: 0 = C (valence 4, weight 12), 1 = O (valence 2, weight 16), 2 = N (valence 3, weight 14)
VALENCIES = [4, 2, 3]
WEIGHTS = [12.0, 16.0, 14.0]
MAX_WEIGHT = 350.0


def feats(**kw):
    return MolecularFeatures(VALENCIES, WEIGHTS, max_weight=MAX_WEIGHT, **kw)


def graph(atom_classes, bonds, de=4, pad_to=None):
    """Dense one-hot (X, E) for one molecule. ``bonds`` = [(i, j, bond_class)]; class 0 = no bond.
    Padding rows are all-zero in X, as the sampler produces them."""
    n = len(atom_classes)
    size = pad_to or n
    X = torch.zeros(1, size, len(VALENCIES))
    for i, c in enumerate(atom_classes):
        X[0, i, c] = 1.0
    E = torch.zeros(1, size, size, de)
    E[..., 0] = 1.0
    for i, j, c in bonds:
        E[0, i, j] = E[0, j, i] = F.one_hot(torch.tensor(c), de).float()
    mask = torch.zeros(1, size, dtype=torch.bool)
    mask[0, :n] = True
    return X, E, mask


def run(f, X, E, mask):
    return f({"X_t": X, "E_t": E, "node_mask": mask})


def test_the_output_dimensions_are_declared_correctly():
    assert feats().output_dims() == {"X": 2, "E": 0, "y": 1}
    X, E, mask = graph([0, 1], [(0, 1, 1)])
    X_extra, y_extra = run(feats(), X, E, mask)
    assert X_extra.shape == (1, 2, 2) and y_extra.shape == (1, 1)


def test_charge_and_valency_of_a_single_bond():
    """C-O single bond: C has used 1 of 4 (charge 3), O has used 1 of 2 (charge 1)."""
    X_extra, _ = run(feats(), *graph([0, 1], [(0, 1, 1)]))
    charge, valency = X_extra[0, :, 0], X_extra[0, :, 1]
    assert charge.tolist() == [3.0, 1.0]
    assert valency.tolist() == [1.0, 1.0]


def test_bond_order_not_class_index_is_what_counts():
    """A double bond (class 2) uses 2 of each atom's valence; a triple (class 3) uses 3."""
    for cls, order in ((1, 1.0), (2, 2.0), (3, 3.0)):
        X_extra, _ = run(feats(), *graph([0, 0], [(0, 1, cls)]))
        assert X_extra[0, :, 1].tolist() == [order, order], cls


def test_aromatic_counts_as_one_and_a_half_when_there_is_a_fifth_edge_class():
    X_extra, _ = run(feats(), *graph([0, 0], [(0, 1, 4)], de=5))
    assert X_extra[0, :, 1].tolist() == [1.5, 1.5], "aromatic is order 1.5, not class index 4"


def test_a_fully_bonded_atom_has_zero_charge():
    """Methane-like carbon: four single bonds use all four valences."""
    X_extra, _ = run(feats(), *graph([0, 1, 1, 1, 1], [(0, i, 1) for i in range(1, 5)]))
    assert X_extra[0, 0, 0].item() == 0.0 and X_extra[0, 0, 1].item() == 4.0


def test_an_over_bonded_atom_has_negative_charge():
    X_extra, _ = run(feats(), *graph([1, 0, 0, 0], [(0, i, 1) for i in range(1, 4)]))
    assert X_extra[0, 0, 0].item() == -1.0          # oxygen with three bonds: 2 - 3


def test_molecular_weight_is_the_normalised_sum_of_atom_weights():
    _, y = run(feats(), *graph([0, 1, 2], [(0, 1, 1), (1, 2, 1)]))
    assert y.item() == pytest.approx((12.0 + 16.0 + 14.0) / MAX_WEIGHT)


def test_padding_does_not_add_phantom_atoms_to_the_weight():
    """Padded rows of X are all-zero, so argmax picks class 0 (carbon). Without the mask each
    padding slot would add a phantom carbon's weight -- a batch-composition-dependent offset on
    a global conditioning signal."""
    X, E, mask = graph([0, 1], [(0, 1, 1)], pad_to=6)
    _, y = run(feats(), X, E, mask)
    assert y.item() == pytest.approx((12.0 + 16.0) / MAX_WEIGHT), "padding leaked into the weight"
    # and a batch of two graphs of different size gets each its own weight
    Xa, Ea, ma = graph([0, 1], [(0, 1, 1)], pad_to=4)
    Xb, Eb, mb = graph([2], [], pad_to=4)
    _, y = run(feats(), torch.cat([Xa, Xb]), torch.cat([Ea, Eb]), torch.cat([ma, mb]))
    assert y[:, 0].tolist() == pytest.approx([28.0 / MAX_WEIGHT, 14.0 / MAX_WEIGHT])


def test_the_weight_falls_back_to_unmasked_when_no_mask_is_given():
    X, E, _ = graph([0, 1], [(0, 1, 1)])
    _, y = MolecularFeatures(VALENCIES, WEIGHTS, max_weight=MAX_WEIGHT)({"X_t": X, "E_t": E})
    assert y.item() == pytest.approx(28.0 / MAX_WEIGHT)


def test_valency_is_capped_so_a_dense_noisy_graph_cannot_run_away():
    """During sampling the intermediate graph can be transiently very dense. Triple bonds to 14
    neighbours would be valency 42; the cap keeps it at 12 and charge at nominal - 12."""
    n = 15
    X, E, mask = graph([0] * n, [(0, j, 3) for j in range(1, n)])
    X_extra, _ = run(feats(), X, E, mask)
    assert X_extra[0, 0, 1].item() == 12.0
    assert X_extra[0, 0, 0].item() == 4.0 - 12.0
    X_extra2, _ = run(feats(max_valency=6.0), X, E, mask)
    assert X_extra2[0, 0, 1].item() == 6.0


def test_the_features_are_unchanged_by_the_order_of_the_atoms():
    """Permutation equivariance: relabelling the atoms permutes the per-atom features the same way."""
    X, E, mask = graph([0, 1, 2], [(0, 1, 1), (1, 2, 2)])
    perm = torch.tensor([2, 0, 1])
    Xp, Ep = X[:, perm], E[:, perm][:, :, perm]
    a, ya = run(feats(), X, E, mask)
    b, yb = run(feats(), Xp, Ep, mask)
    assert torch.equal(a[:, perm], b)
    assert torch.allclose(ya, yb)
