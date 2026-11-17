"""`SubgraphConstraint` and `InpaintingSampler`: freeze a core, grow new nodes around it.

RePaint-style: at every step the running state is projected onto the constraint at the CURRENT
noise level (the frozen region is re-noised to the same level as the free one, so the network's
input stays on-distribution), and after the loop the exact core is installed. What is worth
protecting is the contract a user relies on: the core you asked for is EXACTLY the first ``k``
nodes of every sample, with exactly the bonds you specified between them, whatever else was
generated around it.

The model here is the tiny untrained one from ``conftest`` (4 node classes, 2 edge classes), which
is all that is needed: the guarantee under test does not depend on the model being any good.
"""

import pytest
import torch
import torch.nn.functional as F

from defog_adapt.core import Constraint, InpaintingSampler, SubgraphConstraint

DX, DE = 4, 2
CORE_CLASSES = [0, 1, 2]
CORE_BONDS = [(0, 1), (1, 2)]                 # a chain; atoms 0 and 2 are NOT bonded


def make_core():
    X = F.one_hot(torch.tensor(CORE_CLASSES), DX).float()
    E = torch.zeros(3, 3, DE)
    E[..., 0] = 1.0                           # class 0 = no bond
    bond = F.one_hot(torch.tensor(1), DE).float()
    for i, j in CORE_BONDS:
        E[i, j] = E[j, i] = bond
    return X, E


def random_state(bs, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = F.one_hot(torch.randint(0, DX, (bs, n), generator=g), DX).float()
    idx = torch.triu(torch.randint(0, DE, (bs, n, n), generator=g), diagonal=1)
    idx = idx + idx.transpose(1, 2)
    E = F.one_hot(idx, DE).float()
    return X, E, torch.ones(bs, n, dtype=torch.bool)


# ============================================================ construction
def test_the_constraint_is_a_constraint_and_knows_its_size():
    c = SubgraphConstraint(*make_core())
    assert isinstance(c, Constraint) and c.k == 3


@pytest.mark.parametrize("bad", [
    dict(X=torch.zeros(3), E=torch.zeros(3, 3, DE)),               # X not (k, dx)
    dict(X=torch.zeros(3, DX), E=torch.zeros(3, 4, DE)),           # E not square
    dict(X=torch.zeros(3, DX), E=torch.zeros(2, 2, DE)),           # k disagrees
])
def test_malformed_cores_are_refused(bad):
    with pytest.raises(ValueError):
        SubgraphConstraint(bad["X"], bad["E"])


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        SubgraphConstraint(*make_core(), mode="paint")


# ============================================================ projection
def test_tau_one_installs_the_exact_core_and_touches_nothing_else(small_model):
    X0, E0 = make_core()
    c = SubgraphConstraint(X0, E0)
    X, E, mask = random_state(4, 6)
    Xp, Ep = c.project(X, E, 1.0, mask, small_model.limit_dist)
    assert torch.equal(Xp[:, :3], X0.expand(4, -1, -1))
    assert torch.equal(Ep[:, :3, :3], E0.expand(4, -1, -1, -1))
    assert torch.equal(Xp[:, 3:], X[:, 3:]), "the free nodes must be untouched"
    free_edges = torch.ones(6, 6, dtype=torch.bool)
    free_edges[:3, :3] = False
    assert torch.equal(Ep[:, free_edges], E[:, free_edges]), "bonds touching new nodes must be untouched"


def test_projection_does_not_mutate_its_inputs(small_model):
    c = SubgraphConstraint(*make_core())
    X, E, mask = random_state(3, 6)
    X_before, E_before = X.clone(), E.clone()
    c.project(X, E, 0.4, mask, small_model.limit_dist)
    assert torch.equal(X, X_before) and torch.equal(E, E_before)


def test_clamp_mode_installs_the_exact_core_at_every_tau(small_model):
    X0, E0 = make_core()
    c = SubgraphConstraint(X0, E0, mode="clamp")
    X, E, mask = random_state(4, 6)
    for tau in (0.0, 0.3, 0.9):
        Xp, Ep = c.project(X, E, tau, mask, small_model.limit_dist)
        assert torch.equal(Xp[:, :3], X0.expand(4, -1, -1)), f"tau={tau}"
        assert torch.equal(Ep[:, :3, :3], E0.expand(4, -1, -1, -1)), f"tau={tau}"


@pytest.mark.parametrize("tau", [0.0, 0.5, 0.8])
def test_repaint_mode_renoises_the_core_to_the_requested_level(small_model, tau):
    """p(x_tau | x_1) = tau * onehot(core) + (1 - tau) * limit. For node 0 (core class 0) under
    the uniform limit of the fixture model, P(still class 0) = tau + (1 - tau) / 4."""
    c = SubgraphConstraint(*make_core())
    bs = 6000
    X, E, mask = random_state(bs, 5)
    torch.manual_seed(0)
    Xp, _ = c.project(X, E, tau, mask, small_model.limit_dist)
    limit = float(small_model.limit_dist.X[0])
    want = tau + (1.0 - tau) * limit
    got = float((Xp[:, 0].argmax(-1) == 0).float().mean())
    assert got == pytest.approx(want, abs=0.03), f"tau={tau}: P(core class) {got:.3f} vs {want:.3f}"


def test_a_renoised_core_block_stays_symmetric_with_no_self_bonds(small_model):
    c = SubgraphConstraint(*make_core())
    X, E, mask = random_state(50, 5)
    torch.manual_seed(1)
    _, Ep = c.project(X, E, 0.5, mask, small_model.limit_dist)
    block = Ep[:, :3, :3].argmax(-1)
    assert torch.equal(block, block.transpose(1, 2)), "the core's bond block must stay symmetric"
    assert (block.diagonal(dim1=1, dim2=2) == 0).all(), "no atom may be bonded to itself"


# ============================================================ sampling
def sampler(model, constraint, **kw):
    kw.setdefault("sample_steps", 6)
    return InpaintingSampler(model, constraint, eta=0.0, omega=0.0,
                             time_distortion="identity", **kw)


def core_bonds_of(sample):
    """The undirected bonds among the first three nodes, as a set of (i, j) with i < j."""
    ei = sample.edge_index
    return {(int(u), int(v)) for u, v in ei.t().tolist() if u < v and v < 3}


def test_every_sample_has_core_plus_n_free_nodes(small_model):
    out = sampler(small_model, SubgraphConstraint(*make_core())).sample(
        5, n_free=4, device=torch.device("cpu"), show_progress=False)
    assert len(out) == 5 and all(s.x.shape[0] == 3 + 4 for s in out)


def test_the_exact_core_is_in_every_sample(small_model):
    """THE guarantee: atoms 0..2 are the requested classes and the bonds among them are exactly
    the requested chain (0-1, 1-2; not 0-2), however the rest came out."""
    torch.manual_seed(3)
    out = sampler(small_model, SubgraphConstraint(*make_core())).sample(
        12, n_free=5, device=torch.device("cpu"), show_progress=False)
    for k, s in enumerate(out):
        assert s.x[:3].argmax(-1).tolist() == CORE_CLASSES, f"sample {k}: wrong core atoms"
        assert core_bonds_of(s) == set(CORE_BONDS), f"sample {k}: core bonds {core_bonds_of(s)}"


def test_each_step_sees_the_projected_core(small_model):
    """The final install would hide a broken per-step projection, so look at what the MODEL is
    given. In clamp mode the running state is projected onto the exact core before every step, so
    every call to denoise_step must receive the exact core in slots 0..2."""
    X0, _ = make_core()
    seen = []
    real = small_model.denoise_step

    def spy(t, s, X_t, E_t, *a, **k):
        seen.append(X_t[:, :3].clone())
        return real(t, s, X_t, E_t, *a, **k)

    small_model.denoise_step = spy
    try:
        sampler(small_model, SubgraphConstraint(*make_core(), mode="clamp")).sample(
            3, n_free=3, device=torch.device("cpu"), show_progress=False)
    finally:
        small_model.denoise_step = real
    assert len(seen) == 6
    for step, x in enumerate(seen):
        assert torch.equal(x, X0.expand(3, -1, -1)), f"step {step}: the model did not see the core"


def test_time_travel_also_keeps_the_exact_core(small_model):
    torch.manual_seed(4)
    out = sampler(small_model, SubgraphConstraint(*make_core()), resample=True,
                  n_resample=2, jump_length=2).sample(
        8, n_free=4, device=torch.device("cpu"), show_progress=False)
    for k, s in enumerate(out):
        assert s.x[:3].argmax(-1).tolist() == CORE_CLASSES, f"sample {k}"
        assert core_bonds_of(s) == set(CORE_BONDS), f"sample {k}"


def test_time_travel_actually_runs_the_extra_passes(small_model):
    """If the flag were ignored the two samplers would do the same number of forward passes."""
    calls = {"plain": 0, "tt": 0}
    real = small_model.denoise_step

    def counted(key):
        def f(*a, **k):
            calls[key] += 1
            return real(*a, **k)
        return f

    for key, kw in (("plain", {}), ("tt", dict(resample=True, n_resample=2, jump_length=2))):
        small_model.denoise_step = counted(key)
        try:
            sampler(small_model, SubgraphConstraint(*make_core()), **kw).sample(
                2, n_free=3, device=torch.device("cpu"), show_progress=False)
        finally:
            small_model.denoise_step = real
    assert calls["plain"] == 6
    assert calls["tt"] > calls["plain"], calls


def test_sampling_is_reproducible_under_a_seed(small_model):
    def run():
        torch.manual_seed(11)
        return sampler(small_model, SubgraphConstraint(*make_core())).sample(
            4, n_free=4, device=torch.device("cpu"), show_progress=False)

    a, b = run(), run()
    for x, y in zip(a, b):
        assert torch.equal(x.x, y.x) and torch.equal(x.edge_index, y.edge_index)


def test_a_core_plus_growth_beyond_max_nodes_is_refused(small_model):
    too_many = small_model.max_nodes - 3 + 1
    with pytest.raises(ValueError, match="max_nodes"):
        sampler(small_model, SubgraphConstraint(*make_core())).sample(
            2, n_free=too_many, device=torch.device("cpu"), show_progress=False)


# ============================================================ encoding a core for a kekulized model
def kekulized_domain():
    from defog_adapt.domains import MoleculeDomain
    from defog_adapt.domains.molecule import build_encoders

    _, _, _, bond_dec = build_encoders(["C", "N", "O"], ["SINGLE", "DOUBLE", "TRIPLE"])
    return MoleculeDomain(["C", "N", "O"], bond_dec)


def test_an_aromatic_core_is_encoded_in_kekule_form_for_a_kekulized_vocabulary():
    """Every recorded base is kekulized, so the obvious call -- encode benzene as written -- must work
    instead of failing on a bond type the vocabulary has no class for."""
    X, E = kekulized_domain().encode("c1ccccc1")
    assert X.shape == (6, 3) and E.shape == (6, 6, 4)
    assert X.argmax(-1).tolist() == [0] * 6
    classes = E.argmax(-1)                                  # 0 none, 1 single, 2 double, 3 triple
    assert torch.equal(classes, classes.T), "the bond block must be symmetric"
    ring = [(i, (i + 1) % 6) for i in range(6)]
    assert all(classes[i, j] in (1, 2) for i, j in ring), "ring bonds are single or double, never aromatic"
    assert sorted(int(classes[i, j]) for i, j in ring) == [1, 1, 1, 2, 2, 2], "alternating Kekule form"
    assert int((classes > 0).sum()) == 12, "6 bonds, counted both ways, and no other bonds"


def test_a_kekule_smiles_gives_the_same_core_as_the_aromatic_one():
    """Both forms describe benzene; up to which of the two Kekule structures RDKit picks, the core is
    the same set of single and double bonds around the ring."""
    d = kekulized_domain()
    a = d.encode("c1ccccc1")[1].argmax(-1)
    b = d.encode("C1=CC=CC=C1")[1].argmax(-1)
    assert sorted(a.flatten().tolist()) == sorted(b.flatten().tolist())


def test_the_callers_molecule_keeps_its_aromaticity():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1")
    kekulized_domain()._mol_to_core(mol)
    assert all(b.GetIsAromatic() for b in mol.GetBonds()), "encoding must work on a copy"


def test_an_aromatic_vocabulary_still_encodes_aromatic_bonds_as_they_are():
    """Nothing changes where the vocabulary HAS an aromatic class."""
    from defog_adapt.domains import MoleculeDomain
    from defog_adapt.domains.molecule import build_encoders

    _, _, _, bond_dec = build_encoders(["C"], ["SINGLE", "DOUBLE", "AROMATIC"])
    _, E = MoleculeDomain(["C"], bond_dec).encode("c1ccccc1")
    assert int(E.argmax(-1).max()) == 3 and bool((E.argmax(-1) == 3).any()), "aromatic class kept"
    assert not bool((E.argmax(-1) == 2).any()), "no double bonds introduced"


def test_a_subgraph_of_an_aromatic_ring_is_kekulized_with_the_right_atoms():
    d = kekulized_domain()
    X, E = d.encode_subgraph("Cc1ccccc1", [1, 2, 3, 4, 5, 6])          # the ring, without the methyl
    assert X.shape == (6, 3) and int((E.argmax(-1) > 0).sum()) == 12


def test_an_unkekulizable_core_is_a_clear_error_not_a_crash():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1")
    for atom in mol.GetAtoms():                              # an aromatic ring that cannot be Kekule
        atom.SetNumExplicitHs(0)
    odd = Chem.RWMol(mol)
    odd.RemoveBond(0, 1)                                     # break the ring but keep aromatic flags
    with pytest.raises(ValueError, match="kekulize"):
        kekulized_domain()._mol_to_core(odd.GetMol())
