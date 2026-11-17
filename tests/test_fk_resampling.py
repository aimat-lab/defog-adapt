"""
The three FK resampling schemes: all unbiased, differing in offspring-count variance.

Why this matters. Resampling clones high-weight particles, and a particle that deserved
one child but randomly got none is gone for good -- that is where FK's diversity goes.
The sampler used plain multinomial unconditionally, which is the HIGHEST-variance
unbiased scheme available. Systematic and residual are standard lower-variance
alternatives, and swapping them in costs nothing at sampling time.

The property that must hold for any of them to be a legitimate drop-in is that expected
offspring stays K*w_i -- otherwise the scheme silently reweights the target distribution
and every FK number becomes a number about a different distribution.
"""
import torch

from defog_adapt.core.feynman_kac import _resample_indices

SCHEMES = ("multinomial", "systematic", "residual")


def _weights():
    torch.manual_seed(0)
    return torch.softmax(
        torch.tensor([0., -.5, -1., -1.5, -2., -2.5, -3., -4., -5., -8.]), 0)


def test_every_scheme_returns_k_valid_parents():
    w = _weights()
    for K in (10, 16, 64):
        ww = torch.softmax(torch.randn(K), 0)
        for scheme in SCHEMES:
            idx = _resample_indices(ww, K, scheme)
            assert idx.numel() == K, (scheme, K, idx.numel())
            assert int(idx.min()) >= 0 and int(idx.max()) < K, (scheme, K)


def test_every_scheme_is_unbiased():
    """Expected offspring must be K*w. A biased scheme would tilt the target distribution
    on top of the energy, and nothing downstream would notice."""
    w, K, N = _weights(), 10, 6000
    for scheme in SCHEMES:
        torch.manual_seed(1)
        counts = torch.zeros(K)
        for _ in range(N):
            counts += torch.bincount(_resample_indices(w, K, scheme), minlength=K).float()
        err = (counts / N - K * w).abs().max()
        assert float(err) < 0.05, (scheme, float(err))


def test_systematic_keeps_more_parents_than_multinomial():
    """The whole point of the change. Systematic gives each particle floor or ceil of its
    expected offspring, so it cannot drop one that deserved a child."""
    w, K, N = _weights(), 10, 2000
    kept = {}
    for scheme in SCHEMES:
        torch.manual_seed(2)
        kept[scheme] = sum(len(set(_resample_indices(w, K, scheme).tolist()))
                           for _ in range(N)) / N
    assert kept["systematic"] > kept["multinomial"], kept
    assert kept["residual"] > kept["multinomial"], kept


def test_unknown_scheme_is_refused():
    w = _weights()
    try:
        _resample_indices(w, 10, "stratified")
    except ValueError as e:
        assert "stratified" in str(e)
    else:
        raise AssertionError("an unknown scheme must raise, not silently fall back")


def test_default_scheme_is_residual_and_multinomial_is_still_reachable():
    """The default moved to residual on 2026-09-01, on theoretical grounds -- unbiased and
    lower offspring-count variance -- NOT on measurement, since the observed differences
    sat inside the noise floor on both properties. Anything reproducing a pre-change number
    must pin the scheme explicitly, so multinomial has to remain available."""
    import inspect
    from defog_adapt.core.feynman_kac import FeynmanKacSampler, _resample_indices
    sig = inspect.signature(FeynmanKacSampler.__init__)
    assert sig.parameters["resample_scheme"].default == "residual"
    w = _weights()
    assert _resample_indices(w, 10, "multinomial").numel() == 10
