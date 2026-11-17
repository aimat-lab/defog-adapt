"""`FeynmanKacSampler`: reward-tilted SMC steering over a frozen model.

The sampler runs K particles, scores each by ``phi = -beta * energy`` on its predicted CLEAN
graph, and periodically resamples toward low energy. The tests use a toy energy (the number of
edges in the predicted clean graph) on the tiny untrained model from ``conftest``: the mechanism
under test does not need a good model, only a reward the particles can differ on.
"""

import statistics

import pytest
import torch

from defog_adapt.core import FeynmanKacSampler, Sampler

CPU = torch.device("cpu")


def toy_energy(X1, E1, node_mask):
    """Edges in the predicted clean graph; lower is better."""
    return E1[..., 1:].sum(dim=(1, 2, 3)).float()


def n_edges(samples):
    return torch.tensor([s.edge_index.shape[1] / 2 for s in samples])


def fk(model, **kw):
    kw.setdefault("sample_steps", 20)
    return FeynmanKacSampler(model, toy_energy, eta=0.0, omega=0.0, time_distortion="identity",
                             warmup_frac=0.4, **kw)


def test_it_returns_one_graph_per_particle(small_model):
    out = fk(small_model, beta=1.0).sample(9, device=CPU, show_progress=False)
    assert len(out) == 9 and all(g.x.shape[0] >= 1 for g in out)


def test_it_steers_toward_lower_energy_than_plain_sampling(small_model):
    """The point of the method. Same seed, same model, same step count: with a strong beta the
    particles that end up with fewer predicted edges are the ones that survive resampling, so the
    final graphs have fewer edges on average. Averaged over several seeds so one lucky draw cannot
    pass it."""
    gaps = []
    for seed in range(3):
        plain = Sampler(small_model, sample_steps=20, eta=0.0, omega=0.0,
                        time_distortion="identity")
        torch.manual_seed(seed)
        base = n_edges(plain.sample(64, device=CPU, show_progress=False)).mean().item()
        torch.manual_seed(seed)
        steered = n_edges(fk(small_model, beta=2.0).sample(
            64, device=CPU, show_progress=False)).mean().item()
        gaps.append(base - steered)
    assert statistics.mean(gaps) > 1.0, f"FK did not reduce the edge count: gaps {gaps}"
    assert all(g > 0 for g in gaps), f"FK failed to reduce it on some seed: gaps {gaps}"


def test_a_stronger_beta_steers_harder(small_model):
    def mean_edges(beta):
        torch.manual_seed(5)
        return n_edges(fk(small_model, beta=beta).sample(
            64, device=CPU, show_progress=False)).mean().item()

    assert mean_edges(4.0) < mean_edges(0.05)


def test_it_is_reproducible_under_a_seed(small_model):
    def run():
        torch.manual_seed(21)
        return fk(small_model, beta=1.0).sample(12, device=CPU, show_progress=False)

    for a, b in zip(run(), run()):
        assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)


def test_the_potential_is_minus_beta_times_the_energy(small_model):
    """phi = -beta * energy, so that exp(phi) weights low energy up."""
    sampler = fk(small_model, beta=3.0)
    X = torch.zeros(3, 4, 4)
    E = torch.zeros(3, 4, 4, 2)
    mask = torch.ones(3, 4, dtype=torch.bool)
    sampler._predict_clean = lambda *a, **k: (X, E)
    sampler.energy_fn = lambda X1, E1, m: torch.tensor([0.0, 1.0, 2.5])
    phi = sampler._potential(X, E, torch.zeros(3, 0), torch.full((3, 1), 0.9), mask)
    assert phi.tolist() == pytest.approx([0.0, -3.0, -7.5])


def test_beta_is_stored_as_a_plain_float(small_model):
    assert isinstance(fk(small_model, beta=2).beta, float) and fk(small_model, beta=2).beta == 2.0


def test_a_threshold_that_can_never_be_crossed_is_refused(small_model):
    """The effective sample size is never below 1, so ess_frac * K <= 1 would silently turn the
    run into plain sampling. That has cost real runs, so it is an error."""
    with pytest.raises(ValueError, match="never resample"):
        fk(small_model, beta=1.0, ess_frac=0.1).sample(10, device=CPU, show_progress=False)
    # one particle above the line is fine
    fk(small_model, beta=1.0, ess_frac=0.5).sample(10, device=CPU, show_progress=False)


def test_an_unknown_resample_scheme_is_refused_at_construction(small_model):
    with pytest.raises(ValueError, match="resample_scheme"):
        fk(small_model, beta=1.0, resample_scheme="stratified-ish")


@pytest.mark.parametrize("scheme", ["multinomial", "systematic", "residual"])
def test_every_scheme_runs_and_returns_all_particles(small_model, scheme):
    out = fk(small_model, beta=1.0, resample_scheme=scheme).sample(
        10, device=CPU, show_progress=False)
    assert len(out) == 10


def test_a_composition_is_scored_through_the_adapter_not_the_bare_base(small_model):
    """With an adapter composition as the proposal, the reward must be evaluated on the
    ADAPTER-conditioned prediction. Count how often the adapter's modulation is requested."""
    from defog_adapt.core import AdaLNAdapter, AdapterComposition, ConditionBranch

    base = small_model.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    adapter = AdaLNAdapter.for_base(base, cond_dim=1, hidden=16).eval()
    calls = []
    real_forward = adapter.forward

    def counted(c, t=None):
        calls.append(1)
        return real_forward(c, t=t)

    adapter.forward = counted
    comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([0.5]), 1.0)], base=base)
    sampler = fk(base, beta=1.0, composition=comp)
    sampler.sample(6, device=CPU, show_progress=False)
    # the denoise steps call the adapter via the composition; the reward scoring adds more calls
    # at every checkpoint, so the total must exceed the number of denoise steps alone
    assert len(calls) > sampler.sample_steps, len(calls)
