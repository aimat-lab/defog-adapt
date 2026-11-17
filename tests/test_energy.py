"""`StandardisedEnergy`: the variance normalisation that makes the Feynman-Kac ``beta`` portable.

``MoleculePropertyEnergy`` and ``LearnedPropertyEnergy`` return a squared error in the property's
own units, and the FK potential is ``-beta * energy``. So a raw energy gives ``beta`` a different
meaning for every property, and one tuned on logP (std 1.16) pushes about 76 times less on QED
(std 0.14). The recorded FK results are in the dimensionless, standardised units, so the wrapper
has to divide by the VARIANCE (not the standard deviation), apply the weights it is given, and
leave the invalid-graph sentinel as the same number whatever the property's units.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from defog_adapt.core import FeynmanKacSampler, MoleculePropertyEnergy, StandardisedEnergy

CPU = torch.device("cpu")


class Fixed:
    """An energy that returns fixed per-particle values and, optionally, an invalid sentinel."""

    def __init__(self, values, invalid=None, dtype=torch.float32):
        self.values = torch.tensor(values, dtype=dtype)
        if invalid is not None:
            self.invalid = invalid

    def __call__(self, X1, E1, node_mask):
        return self.values.clone()


def call(energy):
    """Invoke an energy on placeholder tensors (the fixed energies ignore them)."""
    return energy(torch.zeros(1), torch.zeros(1), torch.zeros(1))


# ============================================================ the arithmetic
def test_a_single_term_is_divided_by_its_variance_not_its_standard_deviation():
    out = call(StandardisedEnergy([(Fixed([1.0, 4.0, 9.0]), 2.0)]))
    assert out.tolist() == [0.25, 1.0, 2.25]


def test_several_terms_are_weighted_and_summed():
    a = Fixed([1.0, 2.0])
    b = Fixed([8.0, 4.0])
    out = call(StandardisedEnergy([(a, 1.0, 1.0), (b, 2.0, 3.0)]))
    assert out.tolist() == [1.0 + 3.0 * 8.0 / 4.0, 2.0 + 3.0 * 4.0 / 4.0]       # [7, 5]


def test_the_weight_defaults_to_one():
    a, b = Fixed([1.0, 2.0]), Fixed([3.0, 5.0])
    two = call(StandardisedEnergy([(a, 2.0), (b, 3.0)]))
    three = call(StandardisedEnergy([(a, 2.0, 1.0), (b, 3.0, 1.0)]))
    assert torch.equal(two, three)


def test_equal_errors_in_standard_deviations_give_equal_energies_in_any_units():
    """The property of the whole class. 0.5 standard deviations off target is 0.25 whether the
    property is logP or QED; the raw energies differ by the ratio of the variances."""
    logp_std, qed_std = 1.16, 0.14
    logp = Fixed([(0.5 * logp_std) ** 2])
    qed = Fixed([(0.5 * qed_std) ** 2])
    assert call(logp).item() / call(qed).item() == pytest.approx((logp_std / qed_std) ** 2)
    a = call(StandardisedEnergy([(logp, logp_std)])).item()
    b = call(StandardisedEnergy([(qed, qed_std)])).item()
    assert a == pytest.approx(0.25) and b == pytest.approx(0.25)


# ============================================================ the invalid sentinel
class _NeverDecodes:
    def decode(self, data):
        return None


def _undecodable_batch(k=3, n=4, dx=3, de=2):
    X = F.one_hot(torch.zeros(k, n, dtype=torch.long), dx).float()
    E = F.one_hot(torch.zeros(k, n, n, dtype=torch.long), de).float()
    return X, E, torch.ones(k, n, dtype=torch.bool)


@pytest.mark.parametrize("std", [1.16, 0.14, 37.0])
def test_an_undecodable_graph_scores_the_same_standardised_value_in_every_unit(std):
    """The sentinel is a flag, not a measurement. The energies state it in squared standard
    deviations and pre-multiply by scale**2, so dividing by scale**2 gives back 25 -- for logP,
    for QED and for TPSA alike -- instead of 1e3 becoming 746 here and 52,743 there."""
    raw = MoleculePropertyEnergy(_NeverDecodes(), lambda mol: 0.0, target=0.0, scale=std)
    X, E, mask = _undecodable_batch()
    assert raw(X, E, mask).tolist() == pytest.approx([25.0 * std ** 2] * 3)
    energy = StandardisedEnergy([(raw, std)])
    assert energy(X, E, mask).tolist() == pytest.approx([25.0] * 3)
    assert energy.invalid == pytest.approx(25.0)


def test_the_sentinel_is_reported_only_for_a_single_term_that_has_one():
    sentinel = Fixed([1.0], invalid=50.0)
    assert StandardisedEnergy([(sentinel, 5.0)]).invalid == pytest.approx(2.0)      # 50 / 25
    assert StandardisedEnergy([(sentinel, 5.0, 4.0)]).invalid == pytest.approx(8.0)  # weighted
    assert StandardisedEnergy([(Fixed([1.0]), 5.0)]).invalid is None                 # no sentinel
    assert StandardisedEnergy([(sentinel, 5.0), (sentinel, 5.0)]).invalid is None    # a sum


# ============================================================ refusals
@pytest.mark.parametrize("terms, error", [
    ([], ValueError),
    ([(Fixed([1.0]), 0.0)], ValueError),
    ([(Fixed([1.0]), -1.0)], ValueError),
    ([(Fixed([1.0]), float("nan"))], ValueError),
    ([(Fixed([1.0]), float("inf"))], ValueError),
    ([(Fixed([1.0]), 1.0, float("nan"))], ValueError),
    ([(Fixed([1.0]),)], ValueError),
    ([(Fixed([1.0]), 1.0, 1.0, 1.0)], ValueError),
    ([("not callable", 1.0)], TypeError),
])
def test_malformed_terms_are_refused_at_construction(terms, error):
    with pytest.raises(error):
        StandardisedEnergy(terms)


def test_it_keeps_the_dtype_of_the_energies_it_wraps():
    out = call(StandardisedEnergy([(Fixed([1.0, 2.0], dtype=torch.float64), 2.0)]))
    assert out.dtype == torch.float64


# ============================================================ with the sampler
def toy_energy(X1, E1, node_mask):
    return E1[..., 1:].sum(dim=(1, 2, 3)).float()


def _sample(small_model, energy, beta):
    sampler = FeynmanKacSampler(small_model, energy, beta=beta, eta=0.0, omega=0.0,
                                time_distortion="identity", warmup_frac=0.4, sample_steps=20)
    torch.manual_seed(3)
    return sampler.sample(16, device=CPU, show_progress=False)


def test_at_unit_scale_the_wrapper_is_transparent_to_the_sampler(small_model):
    """Scale 1, weight 1 multiplies by exactly 1.0, so under one seed the particles must come out
    bit for bit as they do with the raw energy -- the wrapper adds nothing but the division."""
    raw = _sample(small_model, toy_energy, beta=2.0)
    wrapped = _sample(small_model, StandardisedEnergy([(toy_energy, 1.0)]), beta=2.0)
    for a, b in zip(raw, wrapped):
        assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)


def test_the_scale_changes_how_hard_the_sampler_steers(small_model):
    """A large scale is a small effective beta: the same energy, standardised by a large
    standard deviation, must steer less than by a small one. (If the wrapper ignored the scale,
    the two would be identical.)"""
    def mean_edges(scale):
        out = _sample(small_model, StandardisedEnergy([(toy_energy, scale)]), beta=2.0)
        return torch.tensor([g.edge_index.shape[1] / 2 for g in out]).mean().item()

    assert mean_edges(0.5) < mean_edges(50.0)
