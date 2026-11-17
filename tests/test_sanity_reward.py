"""`SanityReward`: ``r = valid + connected + rings_ok`` on real decoded molecules.

This is the reward behind the recorded unconditional RL results, and it came out of an
experiment script with no tests of its own. What is worth protecting:

* the tiers, and in particular that INVALID is strictly worst -- if a parseable but flawed
  molecule could score below an unparseable one, the gradient would point at invalidity;
* that the fixture molecules really decode. A batch that decodes to nothing makes every
  assertion below vacuously about zeros, which is exactly how an earlier stub checkpoint made
  an end-to-end smoke test look like it passed.

The batches are built from known SMILES, so the molecules and their expected scores are
independent of any model.
"""

import pytest
import torch

pytest.importorskip("rdkit")

from torch_geometric.data import Batch  # noqa: E402

from defog_adapt.core import SanityReward  # noqa: E402
from defog_adapt.core.data import dense_to_pyg, to_dense  # noqa: E402
from defog_adapt.domains import MoleculeDomain  # noqa: E402
from defog_adapt.domains.molecule import (  # noqa: E402
    build_encoders, needs_kekulize, smiles_to_pyg_data,
)

ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]

SANE = ["CCOCC1CCCCC1", "CC(=O)Nc1ccc(O)cc1", "O=C(N)c1cccnc1"]
WONKY = "C1CCCCCCCCC1"            # a 10-membered ring, outside [3, 8]
DISCONNECTED = "CCO.CCN"


@pytest.fixture(scope="module")
def enc():
    return build_encoders(ATOMS, BONDS)


@pytest.fixture(scope="module")
def domain(enc):
    _, ad, _, bd = enc
    return MoleculeDomain(ad, bd)


def _dense(smiles, enc):
    """One padded dense batch (X1, E1, node_mask) from SMILES, as a rollout supplies it."""
    ae, _, be, _ = enc
    datas = [smiles_to_pyg_data(s, ae, be, kekulize=needs_kekulize(be)) for s in smiles]
    assert all(d is not None for d in datas), "fixture SMILES must encode"
    batch = Batch.from_data_list(datas)
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)
    return dense.X, dense.E, mask


def _overvalent_carbon_batch(enc):
    """A graph that parses as nothing: a carbon bonded to five others (valence 5)."""
    n, dx, de = 6, len(ATOMS), len(BONDS) + 1
    X = torch.zeros(1, n, dx)
    X[..., 0] = 1.0                                  # all carbon
    E = torch.zeros(1, n, n, de)
    E[..., 0] = 1.0                                  # class 0 = no bond
    for j in range(1, n):
        E[0, 0, j, 0] = E[0, j, 0, 0] = 0.0
        E[0, 0, j, 1] = E[0, j, 0, 1] = 1.0          # single bond
    return X, E, torch.ones(1, n, dtype=torch.bool)


def _mixed_batch(smiles, enc):
    """The SMILES batch plus one unparseable graph as the last row, padded to a common size."""
    X1, E1, mask = _dense(smiles, enc)
    Xi, Ei, mi = _overvalent_carbon_batch(enc)
    pad = X1.size(1) - Xi.size(1)
    assert pad >= 0, "the invalid graph must be the smaller one"
    Xi = torch.nn.functional.pad(Xi, (0, 0, 0, pad))
    Ei = torch.nn.functional.pad(Ei, (0, 0, 0, pad, 0, pad))
    mi = torch.nn.functional.pad(mi, (0, pad))
    return torch.cat([X1, Xi]), torch.cat([E1, Ei]), torch.cat([mask, mi])


def _identities(domain, batch):
    X1, E1, mask = batch
    return [domain.identity(d) for d in dense_to_pyg(X1, E1, None, mask, mask.sum(-1))]


def test_the_fixture_molecules_actually_decode(domain, enc):
    """The premise. Without it every score below is about a batch of invalid graphs."""
    ids = _identities(domain, _dense(SANE + [WONKY, DISCONNECTED], enc))
    assert all(i is not None for i in ids), f"a fixture molecule did not decode: {ids}"
    assert "." in ids[-1] and "." not in ids[0]


def test_each_tier_scores_its_documented_value(domain, enc):
    """sane 3, wonky ring 2, disconnected 2: valid + connected + rings_ok."""
    r = SanityReward(domain)
    got = r(*_dense([SANE[0], WONKY, DISCONNECTED], enc)).tolist()
    assert got == [3.0, 2.0, 2.0], got


def test_every_sane_molecule_scores_three(domain, enc):
    got = SanityReward(domain)(*_dense(SANE, enc)).tolist()
    assert got == [3.0] * len(SANE), got


def test_an_unparseable_graph_scores_exactly_zero(domain, enc):
    """Invalid short-circuits to 0, strictly below every valid molecule."""
    batch = _overvalent_carbon_batch(enc)
    assert _identities(domain, batch) == [None], "premise: this graph must not decode"
    assert SanityReward(domain)(*batch).tolist() == [0.0]
    worst_valid = min(SanityReward(domain)(*_dense([WONKY, DISCONNECTED], enc)).tolist())
    assert worst_valid > 0.0


def test_the_ring_window_is_configurable(domain, enc):
    """A 10-ring is wonky at the default [3, 8] and fine once the window admits it."""
    batch = _dense([WONKY], enc)
    assert SanityReward(domain)(*batch).tolist() == [2.0]
    assert SanityReward(domain, ring_lo=3, ring_hi=12)(*batch).tolist() == [3.0]


def test_last_records_the_batch_rates(domain, enc):
    """One sane, one wonky ring, one disconnected, one unparseable. Every rate is over the WHOLE
    batch (4 graphs), not over the valid ones: valid 3/4, connected 2/4, rings ok 2/4, fully sane
    1/4, disconnected 1/4, wonky ring 1/4, and a mean reward of (3 + 2 + 2 + 0) / 4."""
    r = SanityReward(domain)
    batch = _mixed_batch([SANE[0], WONKY, DISCONNECTED], enc)
    assert _identities(domain, batch)[-1] is None, "premise: the last graph must be invalid"
    out = r(*batch)
    assert out.tolist() == [3.0, 2.0, 2.0, 0.0]
    last = r.last
    assert last["valid_frac"] == pytest.approx(3 / 4)
    assert last["connected_frac"] == pytest.approx(2 / 4)
    assert last["rings_ok_frac"] == pytest.approx(2 / 4)
    assert last["sanity_frac"] == pytest.approx(1 / 4)
    assert last["disconnected_frac"] == pytest.approx(1 / 4)
    assert last["wonky_ring_frac"] == pytest.approx(1 / 4)
    assert last["sanity_reward_mean"] == pytest.approx(7 / 4)


def test_an_empty_batch_does_not_divide_by_zero(domain):
    X = torch.zeros(0, 3, len(ATOMS))
    E = torch.zeros(0, 3, 3, len(BONDS) + 1)
    out = SanityReward(domain)(X, E, torch.zeros(0, 3, dtype=torch.bool))
    assert out.numel() == 0
