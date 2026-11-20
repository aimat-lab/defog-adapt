"""Labels for the end-to-end conditional model (``examples/training__e1.py --CONDITION``).

A ``DeFoGModel`` with ``cond_dim > 0`` reads its condition from ``batch.y``. When a batch has none it does
not complain: it treats the whole batch as unconditional and goes on training. So a conditional recipe that
forgot to attach the labels would run for days, converge, and hand back an unconditional model with a
condition encoder nobody ever used. These tests are about the labels, the one place that can go wrong
silently; they need neither data nor PyComex.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Crippen  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not part of an installed wheel")

from defog_adapt.domains.molecule import build_encoders, smiles_to_pyg_data  # noqa: E402

ATOMS, BONDS = ["C", "N", "O"], ["SINGLE", "DOUBLE", "TRIPLE"]
SMILES = "CCO CCN CCC CCCO CCCN CC(C)O CC(=O)O CCOC C1CCCC1 CC(C)N CCCCO CNC".split()


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(EXAMPLES))
    try:
        import _common
    finally:
        sys.path.remove(str(EXAMPLES))
    return _common


@pytest.fixture(scope="module")
def vocab():
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(ATOMS, BONDS)
    return dict(atom_enc=atom_enc, atom_dec=atom_dec, bond_enc=bond_enc, bond_dec=bond_dec)


def graphs_of(vocab, smiles):
    return [smiles_to_pyg_data(s, vocab["atom_enc"], vocab["bond_enc"], kekulize=True) for s in smiles]


def logp(mol):
    return float(Crippen.MolLogP(mol))


def label(common, vocab, graphs, smiles, prop=logp, **kw):
    return common.attach_condition(graphs, smiles, prop, atom_decoder=vocab["atom_dec"],
                                   bond_decoder=vocab["bond_dec"], **kw)


def test_the_label_is_the_property_of_the_decoded_graph_not_of_the_source(common, vocab):
    """A graph stores no formal charge, so a protonated amine's graph is methylamine. The label must
    describe the graph the model is shown, or the low end of the range means something it is not."""
    source = "C[NH3+]"
    decoded_value = logp(Chem.MolFromSmiles("CN"))
    source_value = logp(Chem.MolFromSmiles(source))
    assert abs(source_value - decoded_value) > 0.1, "premise: this molecule's two properties must differ"

    graphs, _, _, _, _ = label(common, vocab, graphs_of(vocab, [source]), [source], mean=0.0, std=1.0)
    assert float(graphs[0].y) == pytest.approx(decoded_value, abs=1e-5)


def test_the_labels_are_standardised_and_a_second_split_reuses_the_first_ones_statistics(common, vocab):
    raw = np.array([logp(Chem.MolFromSmiles(s)) for s in SMILES])
    train, _, mean, std, skipped = label(common, vocab, graphs_of(vocab, SMILES), SMILES)
    assert skipped == 0
    assert mean == pytest.approx(raw.mean()) and std == pytest.approx(raw.std())      # population std
    y = torch.cat([g.y for g in train]).squeeze(1).numpy()
    assert y.mean() == pytest.approx(0.0, abs=1e-5) and y.std() == pytest.approx(1.0, abs=1e-5)
    assert all(g.y.shape == (1, 1) for g in train)                  # what a PyG batch stacks into (B, 1)

    # A validation split is standardised with the TRAIN statistics, not its own.
    held_out = SMILES[:4]
    val, kept, _, _, _ = label(common, vocab, graphs_of(vocab, held_out), None, mean=mean, std=std)
    assert kept is None
    expected = (np.array([logp(Chem.MolFromSmiles(s)) for s in held_out]) - mean) / std
    assert torch.cat([g.y for g in val]).squeeze(1).numpy() == pytest.approx(expected, abs=1e-5)


def test_an_unmeasurable_graph_is_dropped_from_the_graphs_and_the_smiles_together(common, vocab):
    """The novelty reference is built from the kept SMILES, so the two lists must stay aligned."""
    def flaky(mol):
        if Chem.MolToSmiles(mol) == "CCN":
            raise ValueError("this property cannot be measured here")
        return logp(mol)

    graphs, kept, _, _, skipped = label(common, vocab, graphs_of(vocab, SMILES), SMILES, prop=flaky)
    assert skipped == 1 and len(graphs) == len(kept) == len(SMILES) - 1
    assert "CCN" not in kept
    for graph, smi in zip(graphs, kept):
        decoded = common.decoded_molecule(graph, vocab["atom_dec"], vocab["bond_dec"])
        assert Chem.MolToSmiles(decoded) == Chem.MolToSmiles(Chem.MolFromSmiles(smi)), \
            f"graph and SMILES are misaligned at {smi}"


def test_a_conditional_model_trained_on_these_graphs_really_receives_the_condition(common, vocab):
    """The silent failure: a batch without ``y`` trains an unconditional model and says nothing."""
    from torch_geometric.loader import DataLoader

    from defog_adapt.core import DeFoGModel

    graphs, _, _, _, _ = label(common, vocab, graphs_of(vocab, SMILES), SMILES)
    loader = DataLoader(graphs, batch_size=4)
    model = DeFoGModel.from_dataloader(loader, n_layers=2, hidden_dim=16, hidden_mlp_dim=32, n_heads=2,
                                       cond_dim=1, cond_drop_prob=0.0)
    seen = []
    embed = model._embed_condition

    def spy(cond_raw, *args, **kwargs):
        seen.append(cond_raw)
        return embed(cond_raw, *args, **kwargs)

    model._embed_condition = spy
    out = model.training_step(next(iter(loader)), 0)
    assert torch.isfinite(out["loss"])
    assert seen and seen[0] is not None, "the model trained on this batch as if it were unconditional"
    assert seen[0].shape == (4, 1)


def test_the_condition_file_round_trips_and_a_different_run_cannot_overwrite_it(common, tmp_path):
    stats = {"property": "logp", "property_from": "decoded", "mean": 2.4532, "std": 1.4301}
    path = common.write_condition_sidecar(tmp_path, stats)
    assert common.read_condition_sidecar(path) == stats

    # A chained link recomputes the same numbers, to within float noise: accepted.
    common.write_condition_sidecar(tmp_path, {**stats, "mean": stats["mean"] + 1e-9})
    # Another run, or another property, would silently change what the checkpoints mean: refused.
    with pytest.raises(RuntimeError, match="different run"):
        common.write_condition_sidecar(tmp_path, {**stats, "mean": 2.6})
    with pytest.raises(RuntimeError, match="different run"):
        common.write_condition_sidecar(tmp_path, {**stats, "property": "qed"})
    assert common.read_condition_sidecar(path) == stats                      # and nothing was overwritten


def test_a_file_that_is_not_a_condition_file_is_refused(common, tmp_path):
    bad = tmp_path / "condition.json"
    bad.write_text('{"mean": 1.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        common.read_condition_sidecar(bad)
