"""`WeightedSanityPropertyReward`: the four-tier sanity table and the [0, 1] terms.

This class shipped with ZERO test coverage while living in the experiment file of the
research package, so it is pinned here before anything depends on it.

What the tiering is FOR is the thing worth protecting. `PropertyMatchReward` is
connectivity-FIRST -- its disconnect penalty sits below the property clamp, so any
connected molecule outranks any on-target disconnected one, and sanity vetoes targeting
outright. This reward deliberately does the opposite: both terms are normalised to
[0, 1] and weighted, so at the default 3:1 a well-targeted molecule with one oversized
ring CAN outrank a sane but off-target one. Flatten these tiers and the class silently
becomes the thing it was written not to be.
"""

import pytest
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

from defog_adapt.core import PropertyMatchReward, WeightedSanityPropertyReward
from defog_adapt.domains.molecule import build_encoders

RDLogger.DisableLog("rdApp.*")

ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE"]

# sane / wonky-ring / disconnected. All three decode cleanly on the kekulized vocabulary,
# so the only thing separating them is the tier under test.
SANE = "CCOCC1CCCCC1"
WONKY = "C1CCCCCCCCC1"          # 10-ring, outside [3, 8]
DISCONNECTED = "CCO.CCN"


@pytest.fixture
def enc():
    return build_encoders(ATOMS, BONDS)


def _dense(smiles_list, atom_encoder, bond_encoder):
    """One padded dense batch from a list of SMILES, in the network's output space."""
    from torch_geometric.data import Batch

    from defog_adapt.core.data import to_dense
    from defog_adapt.domains.molecule import needs_kekulize, smiles_to_pyg_data

    datas = [smiles_to_pyg_data(s, atom_encoder, bond_encoder,
                                kekulize=needs_kekulize(bond_encoder)) for s in smiles_list]
    assert all(d is not None for d in datas), "fixture SMILES must encode"
    batch = Batch.from_data_list(datas)
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)
    return dense.X, dense.E, mask


# ============================================================ the tiers
def test_sanity_tiers_are_ordered_and_hit_their_documented_values(enc):
    """sane 1.00 > wonky 0.67 > disconnected 0.33 > invalid 0.00, with targeting held
    flat at zero so only the sanity term moves."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([SANE, WONKY, DISCONNECTED], ae, be)
    # w_prop=0 isolates sanity; cond is required by the signature but unused at w_prop=0.
    r = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=1.0,
                                     w_prop=0.0, w_sanity=1.0)
    got = r(X1, E1, nm, torch.zeros(3, 1)).tolist()
    assert got[0] == pytest.approx(1.00, abs=1e-6), f"sane scored {got[0]}"
    assert got[1] == pytest.approx(0.67, abs=1e-6), f"wonky ring scored {got[1]}"
    assert got[2] == pytest.approx(0.33, abs=1e-6), f"disconnected scored {got[2]}"
    assert got[0] > got[1] > got[2], f"tier order broken: {got}"


def test_targeting_is_one_on_target_and_zero_at_span_std(enc):
    """``targeting = 1 - min(|prop - target| / (span*scale), 1)``. Checked at both ends
    against RDKit's own value for the molecule, so the test cannot drift with the fixture."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([SANE, SANE], ae, be)
    true = Crippen.MolLogP(Chem.MolFromSmiles(SANE))
    scale, span = 1.0, 3.0
    r = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=scale,
                                     w_prop=1.0, w_sanity=0.0, prop_span=span)
    cond = torch.tensor([[true], [true + span * scale]])     # exactly on / exactly at the edge
    got = r(X1, E1, nm, cond).tolist()
    assert got[0] == pytest.approx(1.0, abs=1e-6), f"on-target scored {got[0]}"
    assert got[1] == pytest.approx(0.0, abs=1e-6), f"{span} std off-target scored {got[1]}"


def test_targeting_can_outrank_sanity_at_the_default_weights(enc):
    """The whole reason this class exists, stated as an inequality: on-target + one
    oversized ring must beat sane + far-off-target at 3:1. If this flips, the class has
    silently become PropertyMatchReward."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([WONKY, SANE], ae, be)
    wonky_true = Crippen.MolLogP(Chem.MolFromSmiles(WONKY))
    r = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=1.0,
                                     w_prop=3.0, w_sanity=1.0, prop_span=3.0)
    # target = the wonky molecule's own logP, so it is perfectly on-target and the sane
    # one is far off.
    cond = torch.tensor([[wonky_true], [wonky_true]])
    got = r(X1, E1, nm, cond).tolist()
    assert got[0] > got[1], (
        f"on-target wonky ({got[0]:.3f}) must outrank off-target sane ({got[1]:.3f}) "
        "at w_prop=3 / w_sanity=1")


def test_disconnected_still_receives_a_property_gradient(enc):
    """Disconnected molecules keep a targeting score -- RDKit measures the whole
    multi-fragment record -- so the policy can climb out rather than seeing a flat
    penalty. Only unparseable graphs score zero on both terms."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([DISCONNECTED, DISCONNECTED], ae, be)
    true = Crippen.MolLogP(Chem.MolFromSmiles(DISCONNECTED))
    r = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=1.0,
                                     w_prop=3.0, w_sanity=1.0, prop_span=3.0)
    got = r(X1, E1, nm, torch.tensor([[true], [true + 10.0]])).tolist()
    on, off = got
    assert on > off, f"on-target disconnected {on} must beat off-target disconnected {off}"
    assert on == pytest.approx(3.0 + 0.33, abs=1e-6), f"on-target disconnected scored {on}"


def test_output_range_is_bounded_by_the_weights(enc):
    """Both terms live in [0, 1], so the score lives in [0, w_prop + w_sanity] -- a bounded,
    non-negative range that PropertyMatchReward's -10/-4/[-3,0] tiering does not have."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([SANE, WONKY, DISCONNECTED], ae, be)
    r = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=1.0,
                                     w_prop=3.0, w_sanity=1.0)
    got = r(X1, E1, nm, torch.zeros(3, 1))
    assert bool((got >= 0.0).all()), f"negative score: {got.tolist()}"
    assert bool((got <= 4.0).all()), f"score above w_prop + w_sanity: {got.tolist()}"


def test_the_two_rewards_disagree_about_sanity_versus_targeting(enc):
    """A mutation check on the promotion: the two classes must remain DIFFERENT
    orderings. If someone collapses one into the other, this catches it -- an on-target
    disconnected molecule is preferred by the weighted reward and rejected by the
    connectivity-first one."""
    ae, ad, be, bd = enc
    X1, E1, nm = _dense([DISCONNECTED, SANE], ae, be)
    disc_true = Crippen.MolLogP(Chem.MolFromSmiles(DISCONNECTED))
    cond = torch.tensor([[disc_true], [disc_true]])

    weighted = WeightedSanityPropertyReward(ad, bd, Crippen.MolLogP, scale=1.0)(
        X1, E1, nm, cond).tolist()
    tiered = PropertyMatchReward(ad, bd, Crippen.MolLogP, scale=1.0)(
        X1, E1, nm, cond).tolist()

    assert weighted[0] > weighted[1], "weighted reward should prefer the on-target one"
    assert tiered[0] < tiered[1], "connectivity-first reward should reject the disconnected one"
