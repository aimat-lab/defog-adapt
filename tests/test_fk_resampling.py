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


# ===========================================================================
# beta schedules
# ===========================================================================
def test_a_bare_number_is_a_constant_schedule():
    """Everything written before schedules existed passes a float, and must keep meaning
    exactly what it meant."""
    from defog_adapt.core.feynman_kac import make_beta_schedule
    f = make_beta_schedule(20.0)
    assert [f(u) for u in (0.0, 0.5, 1.0)] == [20.0, 20.0, 20.0]


def test_ramp_endpoints_are_exact():
    """u=0 is the FIRST checkpoint and u=1 the LAST, so the stated endpoints are the
    endpoints. Measured against total steps instead, a '2 to 20' ramp would really run
    from 12.8 to 20 under warmup 0.6 and the caption would be fiction."""
    from defog_adapt.core.feynman_kac import make_beta_schedule
    for kind in ("linear", "cosine", "exponential"):
        f = make_beta_schedule({"kind": kind, "start": 2.0, "end": 20.0})
        assert abs(f(0.0) - 2.0) < 1e-9, kind
        assert abs(f(1.0) - 20.0) < 1e-9, kind


def test_ramps_are_monotone_between_their_endpoints():
    from defog_adapt.core.feynman_kac import make_beta_schedule
    for kind in ("linear", "cosine", "exponential"):
        f = make_beta_schedule({"kind": kind, "start": 2.0, "end": 20.0})
        vals = [f(u / 20) for u in range(21)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), (kind, vals)


def test_bad_schedules_are_refused_not_silently_defaulted():
    """A misspelled kind or a missing endpoint must raise. Falling back to a constant would
    run an unsteered-by-schedule arm under a ramp's name, and the results table would
    record a comparison that never happened."""
    from defog_adapt.core.feynman_kac import make_beta_schedule
    import pytest as _pytest
    for bad in ({"kind": "sawtooth", "start": 1, "end": 2},
                {"kind": "linear", "start": 1},
                {"kind": "exponential", "start": 0.0, "end": 2}):
        try:
            make_beta_schedule(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")
    try:
        make_beta_schedule("twenty")
    except TypeError:
        pass
    else:
        raise AssertionError("a string was accepted as a beta spec")


def test_sampler_exposes_the_schedule_and_a_scalar_beta():
    """self.beta stays a float for the code and diagnostics that read it as one, and it is
    the value at the END of the ramp rather than an average, so it is never mistaken for
    the pressure applied throughout."""
    import inspect
    from defog_adapt.core.feynman_kac import FeynmanKacSampler
    src = inspect.getsource(FeynmanKacSampler.__init__)
    assert "self.beta_fn" in src and "self.beta_spec" in src
    assert "self.beta_fn(1.0)" in src
