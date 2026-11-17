"""Adapter branches are blended in probability space.

The predicted clean-graph marginals of the branches are blended (a geometric product of
experts) and ONE rate matrix is derived from the result. An earlier placement built a rate
matrix per branch and blended those; it is not part of this package, because above w=1 it
collapsed validity (logP, w=2: validity 0.526 and MAE 5.59, against 0.982 and 0.542 here).

These tests pin the properties that make the probability-space placement what it claims to be:
the algebra of the blend, one rate matrix per step regardless of the number of branches, the
default weight, and the end-to-end paths. (The comparison against the removed placement lives in
the research log, not here: it needs the removed code to run.)
"""

import pytest
import torch

from defog_adapt.core import (AdaLNAdapter, AdapterComposition, AdaptedSampler,
                              ConditionBranch, DeFoGModel)
from defog_adapt.core.data import to_dense

from defog_adapt.domains.molecule import build_encoders, smiles_to_pyg_data
from torch_geometric.loader import DataLoader


def build_tiny_model():
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(["C", "N", "O"], ["SINGLE", "DOUBLE"])
    smis = ["CCO", "CCN", "CCC", "CNO", "OCC", "NCC"]
    graphs = [g for g in (smiles_to_pyg_data(s, atom_enc, bond_enc) for s in smis) if g is not None]
    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    model = DeFoGModel.from_dataloader(
        loader, n_layers=2, hidden_dim=32, hidden_mlp_dim=64, n_heads=2, dropout=0.0,
        noise_type="marginal", extra_features_type="rrwp", rrwp_steps=3,
        molecular_features=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loader


def a_state(model, loader, seed=0):
    """One noisy graph state to step from."""
    batch = next(iter(loader))
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)
    bs = dense.X.size(0)
    torch.manual_seed(seed)
    noisy = model._apply_noise(dense.X, dense.E, torch.zeros(bs, 0), mask)
    return noisy, mask, bs


def a_trained_adapter(model, seed=0):
    """An adapter whose gates are NOT zero -- a zero-gate adapter is an exact no-op,
    which would make every blend trivially agree and prove nothing."""
    torch.manual_seed(seed)
    adapter = AdaLNAdapter.for_base(model, cond_dim=1, hidden=16, time_conditioned=True).eval()
    with torch.no_grad():
        for lay in adapter.gate:
            for s in lay:
                lay[s].weight.normal_(0, 0.05)
                lay[s].bias.normal_(0, 0.05)
    return adapter


def count_rate_matrix_calls(model, comp, noisy, mask):
    """How many times one composed denoise step asks for a rate matrix."""
    bs = noisy["X_t"].size(0)
    t = torch.full((bs, 1), 0.5)
    designer = model.rate_matrix_designer
    real = designer.compute_rate_matrices
    n = [0]

    def counted(*a, **kw):
        n[0] += 1
        return real(*a, **kw)

    designer.compute_rate_matrices = counted
    try:
        torch.manual_seed(123)
        model.denoise_step(t, t + 0.05, noisy["X_t"], noisy["E_t"], noisy["y_t"], mask,
                           eta=0.0, omega=0.0, composition=comp)
    finally:
        designer.compute_rate_matrices = real
    return n[0]


class TestBlendAlgebra:
    """The blend itself, on tensors, with no model in the way."""

    @staticmethod
    def _marginals(n_branches, seed=0):
        g = torch.Generator().manual_seed(seed)
        p = torch.rand(n_branches + 1, 2, 3, 5, generator=g) + 0.05
        return p / p.sum(-1, keepdim=True)

    def test_single_branch_at_w1_is_the_conditional(self):
        """The load-bearing identity: with one branch at w=1 the product of experts is
        lu + (lc - lu) = lc, i.e. exactly the conditional's own marginals."""
        p = self._marginals(1)
        q = DeFoGModel._blend_logp(p, torch.tensor([1.0]), "product")
        assert torch.allclose(q, torch.log(p[1] + 1e-8), atol=1e-6)

    def test_zero_weight_is_the_unconditional(self):
        p = self._marginals(1)
        q = DeFoGModel._blend_logp(p, torch.tensor([0.0]), "product")
        assert torch.allclose(q, torch.log(p[0] + 1e-8), atol=1e-6)

    def test_product_sums_the_log_ratios_and_mean_averages_them(self):
        p = self._marginals(2)
        w = torch.tensor([1.5, 0.5])
        lu, lc = torch.log(p[0] + 1e-8), torch.log(p[1:] + 1e-8)
        want = lu + 1.5 * (lc[0] - lu) + 0.5 * (lc[1] - lu)
        assert torch.allclose(DeFoGModel._blend_logp(p, w, "product"), want, atol=1e-6)
        mean = DeFoGModel._blend_logp(p, w, "mean")
        assert torch.allclose(mean, lu + (want - lu) / 2, atol=1e-6)

    def test_w_above_one_extrapolates_past_the_conditional(self):
        """The reason w=2 is usable: in log space it moves FURTHER from the base than the
        conditional itself, in the direction the conditional already points."""
        p = self._marginals(1)
        lu, lc = torch.log(p[0] + 1e-8), torch.log(p[1] + 1e-8)
        q2 = DeFoGModel._blend_logp(p, torch.tensor([2.0]), "product")
        assert torch.allclose(q2 - lu, 2.0 * (lc - lu), atol=1e-6)


class TestOneRateMatrixPerStep:
    def test_a_single_branch_derives_one_rate_matrix(self):
        """compute_rate_matrices draws its own X_1 sample, so blending per-branch rate
        matrices mixed N+1 independent draws. Blending marginals first spends exactly one."""
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        comp = AdapterComposition(
            [ConditionBranch(a_trained_adapter(model), torch.full((bs, 1), 0.7), 1.0)])
        assert count_rate_matrix_calls(model, comp, noisy, mask) == 1

    def test_two_branches_still_derive_one_rate_matrix(self):
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        comp = AdapterComposition(
            [ConditionBranch(a_trained_adapter(model, seed=1), torch.full((bs, 1), 0.7), 1.0),
             ConditionBranch(a_trained_adapter(model, seed=2), torch.full((bs, 1), -0.3), 1.0)])
        assert count_rate_matrix_calls(model, comp, noisy, mask) == 1


class TestDefaults:
    def test_default_branch_weight_is_two(self):
        """w=2 is the measured optimum in probability space (logP MAE 0.5420 against
        0.6410 at w=1)."""
        model, _ = build_tiny_model()
        adapter = a_trained_adapter(model)
        assert ConditionBranch(adapter, torch.tensor([[0.7]])).weight == 2.0

    def test_blend_space_is_not_a_parameter(self):
        """Rate-space blending was removed. Passing the old keyword must fail loudly, not be
        swallowed and silently run the other placement."""
        with pytest.raises(TypeError):
            AdapterComposition([], blend_space="rate")


class TestEndToEnd:
    def test_sampling_produces_valid_graphs(self):
        model, loader = build_tiny_model()
        adapter = a_trained_adapter(model)
        comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([[0.7]]), 1.5)])
        torch.manual_seed(7)
        out = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0).sample(
            num_samples=3, show_progress=False)
        assert len(out) == 3
        for g in out:
            assert g.x.shape[0] >= 1
            assert g.edge_index.shape[0] == 2
            assert torch.isfinite(g.x.float()).all()

    def test_two_branch_composition_runs(self):
        model, loader = build_tiny_model()
        a1, a2 = a_trained_adapter(model, seed=1), a_trained_adapter(model, seed=2)
        comp = AdapterComposition(
            [ConditionBranch(a1, torch.tensor([[0.7]]), 1.0),
             ConditionBranch(a2, torch.tensor([[-0.3]]), 1.0)])
        torch.manual_seed(7)
        out = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0).sample(
            num_samples=3, show_progress=False)
        assert len(out) == 3
