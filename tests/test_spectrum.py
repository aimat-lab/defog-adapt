"""`Band`, `SpectrumEnergy` and `band_masses`: Feynman-Kac steering toward spectral features.

The predictor is any black box (``list[Mol] -> (n, len(grid))``), so these tests use a synthetic
one and a stub decoder. What is worth protecting is what the module's own documentation says its
energy depends on:

* the band penalty is ONE-SIDED. A molecule with a very strong carbonyl has satisfied "give me a
  carbonyl", not overshot it; a two-sided squared error ranks it as badly as one with none, which
  is what the first version did;
* bands are scored in units of their own scale, or the energies are too small for the Feynman-Kac
  weights to differ and every request returns the same molecules;
* nothing a predictor does (NaN output, an exception, an all-zero spectrum) may reach the sampler
  as a NaN or a crash, because one NaN weight takes the whole run with it;
* the predictor is called ONCE per batch with only the decodable graphs.
"""

import numpy as np
import pytest
import torch

from defog_adapt.core import Band, SpectrumEnergy, band_masses

GRID = np.linspace(500.0, 4000.0, 701)        # 5 cm-1 spacing


def spectrum(mass_in_band: float, centre=1710.0, half_width=30.0) -> np.ndarray:
    """A normalised spectrum with ``mass_in_band`` of its absorption inside the window and the
    rest spread evenly over the points outside it."""
    inside = np.abs(GRID - centre) <= half_width
    out = np.zeros_like(GRID)
    out[inside] = mass_in_band / inside.sum()
    out[~inside] = (1.0 - mass_in_band) / (~inside).sum()
    return out


def energy(bands=None, **kw):
    return SpectrumEnergy(domain=None, predictor=lambda mols: None, grid=GRID,
                          bands=bands if bands is not None else [Band(1710.0, 30.0, target=0.07)],
                          **kw)


# ============================================================ Band
def test_band_validates_its_arguments():
    for bad in (dict(half_width=0.0), dict(half_width=-5.0), dict(target=-0.1),
                dict(target=1.5), dict(mode="sideways"), dict(scale=0.0), dict(scale=-1.0)):
        with pytest.raises(ValueError):
            Band(1710.0, **bad)


def test_band_mode_defaults_read_the_obvious_way():
    assert Band(1710.0, target=0.07).mode == "at_least"          # "give me a carbonyl"
    assert Band(3400.0, target=0.0).mode == "at_most"            # "give me no free O-H"
    assert Band(1710.0, target=0.07, mode="at_most").mode == "at_most"


def test_band_scale_defaults_to_the_target_with_a_floor():
    assert Band(1710.0, target=0.07).scale == pytest.approx(0.07)
    assert Band(2230.0, target=0.007).scale == pytest.approx(Band.MIN_SCALE)   # tiny target
    assert Band(3400.0, target=0.0).scale == pytest.approx(Band.MIN_SCALE)     # absence request
    assert Band(1710.0, target=0.07, scale=0.5).scale == 0.5


def test_band_window_is_inclusive_on_both_edges():
    b = Band(1000.0, half_width=50.0)
    grid = np.array([949.0, 950.0, 1000.0, 1050.0, 1051.0])
    assert b.window(grid).tolist() == [False, True, True, True, False]


def test_at_least_is_one_sided():
    """THE property. Falling short is penalised; exceeding the target is not."""
    b = Band(1710.0, target=0.07)
    short, met, strong = b.shortfall(np.array([0.0, 0.07, 0.5]))
    assert short == pytest.approx(1.0)                  # (0.07 - 0) / 0.07
    assert met == 0.0
    assert strong == 0.0, "a very strong band must not be penalised for overshooting"


def test_at_most_is_one_sided_the_other_way():
    b = Band(3400.0, target=0.0, mode="at_most")
    absent, little, lots = b.shortfall(np.array([0.0, 0.01, 0.1]))
    assert absent == 0.0
    assert little == pytest.approx(0.01 / Band.MIN_SCALE)
    assert lots == pytest.approx(0.1 / Band.MIN_SCALE)


# ============================================================ SpectrumEnergy.score
def test_meeting_the_band_scores_zero_and_missing_it_scores_by_scale():
    e = energy()
    got = e.score(np.stack([spectrum(0.07), spectrum(0.0)]))
    assert got[0] == pytest.approx(0.0, abs=1e-9)
    assert got[1] == pytest.approx(1.0)       # one scale-unit short, squared


def test_a_stronger_band_never_scores_worse():
    e = energy()
    scores = e.score(np.stack([spectrum(m) for m in (0.0, 0.02, 0.05, 0.07, 0.3, 0.9)]))
    assert np.all(np.diff(scores) <= 1e-12), scores
    assert scores[-1] == pytest.approx(0.0, abs=1e-9)


def test_energies_are_in_units_of_the_band_scale_not_raw_mass():
    """Raw fractions of absorption are ~0.07, so a squared raw error is ~1e-3 and exp(-beta*E)
    barely differs between particles. Scaling by the band's scale makes a total miss cost 1."""
    e = energy()
    assert e.score(spectrum(0.0)[None])[0] > 0.5      # raw squared error would be 0.0049


def test_band_weight_multiplies_its_contribution():
    base = energy([Band(1710.0, 30.0, target=0.07, weight=1.0)])
    heavy = energy([Band(1710.0, 30.0, target=0.07, weight=3.0)])
    s = spectrum(0.0)[None]
    assert heavy.score(s)[0] == pytest.approx(3.0 * base.score(s)[0])


def test_two_bands_add():
    both = energy([Band(1710.0, 30.0, target=0.07), Band(2230.0, 20.0, target=0.02)])
    only_first = energy([Band(1710.0, 30.0, target=0.07)])
    only_second = energy([Band(2230.0, 20.0, target=0.02)])
    s = spectrum(0.0)[None]
    assert both.score(s)[0] == pytest.approx(only_first.score(s)[0] + only_second.score(s)[0])


def test_an_all_zero_spectrum_is_scored_invalid_not_divided_by_zero():
    e = energy(invalid_energy=1e3)
    got = e.score(np.stack([np.zeros_like(GRID), spectrum(0.07)]))
    assert got[0] == 1e3 and np.isfinite(got).all()
    assert got[1] == pytest.approx(0.0, abs=1e-9)


def test_nan_and_inf_in_a_prediction_never_reach_the_energy():
    """A phase that cannot report at some wavenumber returns NaN there. That is no absorption,
    not a poisoned weight -- and it must be handled INSIDE the band's window too, where it would
    otherwise turn the band's mass, and so the energy, into NaN."""
    e = energy()
    inside = np.flatnonzero(np.abs(GRID - 1710.0) <= 30.0)
    s = spectrum(0.07).copy()
    s[inside[:2]] = np.nan
    s[inside[2:4]] = np.inf
    s[inside[4:6]] = -np.inf
    s[:50] = np.nan                                   # and outside it
    got = e.score(s[None])
    assert np.isfinite(got).all(), got


def test_a_reference_spectrum_is_matched_two_sidedly():
    """``reference`` is for a whole measured shape: identical scores 0, anything else > 0 in
    BOTH directions (unlike a band, which only penalises falling short)."""
    ref = spectrum(0.2)
    e = SpectrumEnergy(None, lambda m: None, GRID, bands=[], reference=ref)
    got = e.score(np.stack([ref, spectrum(0.0), spectrum(0.6)]))
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert got[1] > 0 and got[2] > 0


def test_nan_reference_entries_are_unmeasured_and_excluded():
    """An instrument that did not cover a region made no claim about it, so what a prediction
    does there cannot move the reference term -- while what it does where it WAS measured can."""
    ref = spectrum(0.2).copy()
    ref[:100] = np.nan
    e = SpectrumEnergy(None, lambda m: None, GRID, bands=[], reference=ref)
    a = spectrum(0.2).copy()
    b = a.copy()
    b[:100] = 50.0 * a.max()                       # wildly different, but only where unmeasured
    ea, eb = e.score(np.stack([a, b]))
    assert ea == pytest.approx(eb, abs=1e-12)
    c = a.copy()
    c[200:300] *= 5.0                              # a change in the MEASURED region
    assert e.score(c[None])[0] > ea + 1e-9


def test_constructor_refuses_requests_it_cannot_score():
    with pytest.raises(ValueError, match="at least one band or a reference"):
        SpectrumEnergy(None, lambda m: None, GRID)
    with pytest.raises(ValueError, match="selects no point"):
        SpectrumEnergy(None, lambda m: None, GRID, bands=[Band(9000.0, 10.0)])
    with pytest.raises(ValueError, match="grid has"):
        SpectrumEnergy(None, lambda m: None, GRID, bands=[], reference=np.ones(10))
    with pytest.raises(ValueError, match="entirely NaN"):
        SpectrumEnergy(None, lambda m: None, GRID, bands=[], reference=np.full_like(GRID, np.nan))


# ============================================================ __call__ (batching, failures)
class StubDomain:
    """Decodes a graph to its node count, except 2-node graphs, which 'do not parse'."""

    def decode(self, data):
        n = int(data.x.shape[0])
        return None if n == 2 else n


def _batch(sizes, dx=2, de=2):
    bs, n = len(sizes), max(sizes)
    X = torch.zeros(bs, n, dx)
    X[..., 0] = 1.0
    E = torch.zeros(bs, n, n, de)
    E[..., 0] = 1.0
    mask = torch.zeros(bs, n, dtype=torch.bool)
    for i, k in enumerate(sizes):
        mask[i, :k] = True
    return X, E, mask


def _predictor(calls):
    """3-atom molecules have a strong carbonyl band; 4-atom ones have none."""
    def predict(mols):
        calls.append(list(mols))
        return np.stack([spectrum(0.5 if m == 3 else 0.0) for m in mols])
    return predict


def test_the_predictor_is_called_once_with_only_the_decodable_graphs():
    calls = []
    e = SpectrumEnergy(StubDomain(), _predictor(calls), GRID,
                       bands=[Band(1710.0, 30.0, target=0.07)], invalid_energy=1e3)
    out = e(*_batch([3, 2, 4]))
    assert len(calls) == 1 and calls[0] == [3, 4], "predictor must be batched and skip undecodable graphs"
    assert out.shape == (3,)
    assert out[0].item() == pytest.approx(0.0, abs=1e-9)        # strong carbonyl: satisfied
    assert out[1].item() == 1e3                                  # did not decode
    assert out[2].item() == pytest.approx(1.0)                   # no carbonyl: one scale short


def test_a_failing_predictor_does_not_take_the_run_down():
    def boom(mols):
        raise RuntimeError("model crashed")
    e = SpectrumEnergy(StubDomain(), boom, GRID, bands=[Band(1710.0, 30.0)], invalid_energy=77.0)
    assert e(*_batch([3, 4])).tolist() == [77.0, 77.0]


def test_a_batch_that_decodes_to_nothing_never_calls_the_predictor():
    calls = []
    e = SpectrumEnergy(StubDomain(), _predictor(calls), GRID, bands=[Band(1710.0, 30.0)])
    out = e(*_batch([2, 2]))
    assert calls == [] and (out == e.invalid).all()


def test_a_predictor_returning_the_wrong_shape_is_a_loud_error():
    e = SpectrumEnergy(StubDomain(), lambda mols: np.zeros((len(mols), 10)), GRID,
                       bands=[Band(1710.0, 30.0)])
    with pytest.raises(ValueError, match="expected"):
        e(*_batch([3, 4]))


# ============================================================ band_masses
def test_band_masses_report_what_each_molecule_achieved_per_band():
    bands = [Band(1710.0, 30.0), Band(2230.0, 20.0)]
    spectra = np.stack([spectrum(0.2), spectrum(0.0)])
    got = band_masses(spectra, GRID, bands)
    assert got.shape == (2, 2)
    assert got[0, 0] == pytest.approx(0.2)
    assert got[1, 0] == pytest.approx(0.0, abs=1e-12)
    # masses are fractions of each molecule's own total absorption, so scaling a spectrum
    # does not change what it achieved
    assert np.allclose(band_masses(spectra * 40.0, GRID, bands), got)
