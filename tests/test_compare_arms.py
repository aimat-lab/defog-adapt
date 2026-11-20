"""``examples/compare_arms__zinc.py``: the harness behind the README's results figure.

The numbers it produces take days of GPU time to make and are read once, so what is tested here is the
wiring that could quietly make them mean something else: which split the targets come from, which number
reaches which model, that a result is never reused under another configuration, and how seeds are
averaged. Nothing here needs data, checkpoints or PyComex; the models are tiny and untrained.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Crippen  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not part of an installed wheel")

from torch_geometric.data import Data  # noqa: E402

from defog_adapt.core import AdaLNAdapter, DeFoGModel, PropertyHead, fit_size_model  # noqa: E402
from defog_adapt.data import zinc_reference  # noqa: E402
from defog_adapt.domains.molecule import build_encoders, smiles_to_pyg_data  # noqa: E402


@pytest.fixture(scope="module")
def harness():
    script = EXAMPLES / "compare_arms__zinc.py"
    name = "compare_arms_under_test"
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLES))
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def vocab():
    atoms, bonds = list(zinc_reference.ATOM_TYPES), list(zinc_reference.BOND_TYPES)
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atoms, bonds)
    return SimpleNamespace(atoms=atoms, bonds=bonds, atom_enc=atom_enc, atom_dec=atom_dec,
                           bond_enc=bond_enc, bond_dec=bond_dec)


def logp(smiles):
    return float(Crippen.MolLogP(Chem.MolFromSmiles(smiles)))


def graph(vocab, smiles):
    return smiles_to_pyg_data(smiles, vocab.atom_enc, vocab.bond_enc, kekulize=True)


def invalid_graph(vocab, k=5):
    """A carbon with ``k`` single bonds: a graph that is not a molecule once k > 4. (Edge class 0 is "no
    bond", so the single bonds are class 1; a helper that forgot that built isolated atoms, which decode.)"""
    ref = graph(vocab, "CCO")
    x = torch.zeros(k + 1, ref.x.shape[1])
    x[:, 0] = 1.0                                              # all carbon
    rows = [0] * k + list(range(1, k + 1))
    cols = list(range(1, k + 1)) + [0] * k
    edge_attr = torch.zeros(2 * k, ref.edge_attr.shape[1])
    edge_attr[:, 1] = 1.0                                      # all single
    return Data(x=x, edge_index=torch.tensor([rows, cols]), edge_attr=edge_attr)


CTMC = dict(sample_steps=3, eta=0.0, omega=0.0, time_distortion="identity")
CPU = torch.device("cpu")


# ============================================================================ measuring one arm
def test_the_error_is_pooled_over_valid_generations_and_validity_counts_the_rest(harness, vocab):
    """Three of four generations are molecules (two of them the same): the error is the mean over those three,
    validity is 3/4 and uniqueness 2/3. A target with no valid molecule does not drag the error to zero."""
    ethanol, hexane = logp("CCO"), logp("CCCCCC")
    target = 1.0

    def draw(t, n):
        return [graph(vocab, "CCO"), graph(vocab, "CCCCCC"), invalid_graph(vocab), graph(vocab, "CCO")][:n]

    result = harness.evaluate(draw, [target], 4, vocab.atom_dec, vocab.bond_dec, log=lambda *_: None)
    expected = (2 * abs(ethanol - target) + abs(hexane - target)) / 3
    assert result["mae"] == pytest.approx(expected)
    assert result["validity"] == pytest.approx(3 / 4)
    assert result["uniqueness"] == pytest.approx(2 / 3)
    assert result["per_target_valid"] == [3] and result["per_target_mae"][0] == pytest.approx(expected)


def test_the_error_is_pooled_across_targets_not_averaged_per_target(harness, vocab):
    """The protocol pools every valid generation. A target that yielded three molecules counts three times,
    one that yielded a single molecule once: averaging per target would give the lone molecule a 3x vote."""
    ethanol, hexane = logp("CCO"), logp("CCCCCC")

    def draw(target, n):
        return [graph(vocab, "CCO")] * 3 if target == 0.0 else [graph(vocab, "CCCCCC"), invalid_graph(vocab),
                                                                  invalid_graph(vocab)]

    result = harness.evaluate(draw, [0.0, 5.0], 3, vocab.atom_dec, vocab.bond_dec, log=lambda *_: None)
    pooled = (3 * abs(ethanol - 0.0) + abs(hexane - 5.0)) / 4
    per_target = (abs(ethanol - 0.0) + abs(hexane - 5.0)) / 2
    assert abs(pooled - per_target) > 0.1, "premise: the two ways of averaging must differ here"
    assert result["mae"] == pytest.approx(pooled)


def test_a_target_with_no_valid_molecule_has_no_error_rather_than_a_zero_one(harness, vocab):
    def draw(t, n):
        return [invalid_graph(vocab)] * n if t < 0 else [graph(vocab, "CCO")] * n

    result = harness.evaluate(draw, [-1.0, 0.5], 2, vocab.atom_dec, vocab.bond_dec, log=lambda *_: None)
    assert result["per_target_mae"][0] is None and result["per_target_valid"] == [0, 2]
    assert result["mae"] == pytest.approx(abs(logp("CCO") - 0.5))      # only the target that had molecules
    nothing = harness.evaluate(lambda t, n: [invalid_graph(vocab)] * n, [0.0], 2, vocab.atom_dec,
                               vocab.bond_dec, log=lambda *_: None)
    assert nothing["mae"] is None and nothing["validity"] == 0.0


# ============================================================================ which split the targets come from
def test_targets_come_from_the_requested_split_and_from_no_other(harness, vocab, monkeypatch):
    fake = SimpleNamespace(val_smiles=["CCO", "CCCCCC"], test_smiles=["CCN", "CCCCCCCC"])
    monkeypatch.setattr(zinc_reference, "load_reference_split", lambda *a, **k: fake)
    from steer_zinc import evaluation_targets      # the harness put its examples directory on sys.path

    def draw(split):
        rows = evaluation_targets([SimpleNamespace(name="logp")], 2, 7, None, vocab.atom_enc, vocab.bond_enc,
                                  vocab.atom_dec, vocab.bond_dec, split=split)
        return sorted(row[0] for row in rows)

    assert draw("validation") == pytest.approx(sorted([logp("CCO"), logp("CCCCCC")]))
    assert draw("test") == pytest.approx(sorted([logp("CCN"), logp("CCCCCCCC")]))
    with pytest.raises(ValueError, match="split must be"):
        draw("train")


# ============================================================================ the conditional arm
def tiny_model(vocab, **kw):
    return DeFoGModel(num_node_classes=len(vocab.atoms), num_edge_classes=len(vocab.bonds) + 1, n_layers=2,
                      hidden_dim=16, hidden_mlp_dim=32, n_heads=2, dropout=0.0, max_nodes=12,
                      sample_steps=3, noise_type="uniform", node_counts=torch.tensor([0.0] * 5 + [1.0] * 6 + [0.0]),
                      **kw)


class RecordingSize:
    """Stands in for the learned size model: always draws 11 nodes, and remembers what it was asked. Eleven is
    outside the tiny model's own node-count marginal (5 to 10), so a draw that bypassed this one could never
    produce it, not even by chance."""
    cond_mean, cond_std = torch.tensor([2.0]), torch.tensor([4.0])
    property_name, property_from = "logp", "decoded"

    def __init__(self):
        self.seen = []

    def sample(self, num_samples, condition=None, device=None, generator=None):
        self.seen.append(condition.clone())
        return torch.full((num_samples,), 11, dtype=torch.long)


def test_the_size_model_gets_the_raw_target_and_the_conditional_model_the_standardised_one(harness, vocab):
    """One number, two meanings: the sampler would hand a single ``condition`` to both. The size model
    normalises its input itself; the model expects it standardised by the caller. Swapping them, or sending
    the same value to both, still runs, and conditions on the wrong target."""
    model = tiny_model(vocab, cond_dim=1, cond_drop_prob=0.1).eval()
    size = RecordingSize()
    seen = []
    embed = model._embed_condition

    def spy(cond_raw, *args, **kwargs):
        seen.append(None if cond_raw is None else cond_raw.clone())
        return embed(cond_raw, *args, **kwargs)

    model._embed_condition = spy
    stats = {"property": "logp", "property_from": "decoded", "mean": 2.0, "std": 4.0}
    graphs = harness.conditional_draw(model, stats, size, CTMC, 1.0, CPU)(6.0, 3)

    assert torch.equal(size.seen[0], torch.full((3, 1), 6.0)), "the size model must get the RAW target"
    assert len(seen) == 1 and seen[0] is not None
    assert torch.allclose(seen[0], torch.full((3, 1), 1.0)), "the model must get (6 - 2) / 4 = 1"
    assert [g.x.shape[0] for g in graphs] == [11, 11, 11], "the node counts must be the size model's draw"


def test_without_a_size_model_the_conditional_arm_still_conditions_and_uses_the_marginal(harness, vocab):
    model = tiny_model(vocab, cond_dim=1).eval()
    seen = []
    embed = model._embed_condition
    model._embed_condition = lambda c, *a, **k: (seen.append(c), embed(c, *a, **k))[1]
    stats = {"property": "logp", "property_from": "decoded", "mean": 0.0, "std": 2.0}
    graphs = harness.conditional_draw(model, stats, None, CTMC, 1.0, CPU)(4.0, 2)
    assert len(graphs) == 2 and torch.allclose(seen[0], torch.full((2, 1), 2.0))


@pytest.mark.parametrize("bad, message", [
    (dict(property="qed"), "logp"),
    (dict(property_from="source"), "decoded"),
    (dict(mean=2.4), "label conventions"),                     # 0.4 of a std away: a different convention
    (dict(std=1.5), "label conventions"),
])
def test_a_conditional_model_whose_labels_do_not_match_the_size_model_is_refused(harness, bad, message):
    good = {"property": "logp", "property_from": "decoded", "mean": 2.0, "std": 4.0}
    harness.check_conditional_pairing({**good, "mean": 2.02}, RecordingSize())            # 0.5% of a std: fine
    with pytest.raises(RuntimeError, match=message):
        harness.check_conditional_pairing({**good, **bad}, RecordingSize())


# ============================================================================ results on disk
PROTOCOL = dict(split="test", n_targets=100, n_per_target=10,
                sampling=dict(sample_steps=500, eta=25.0, omega=0.0, time_distortion="polydec"))


def write_run(directory, arm, seed, mae, **extra):
    record = {"arm": arm, "seed": seed, "mae": mae, "validity": 0.98, "uniqueness": 0.9,
              "guidance_weight": None, "beta": None, "size_model": True, "checkpoints": {}, **PROTOCOL, **extra}
    (Path(directory) / f"{arm}__seed{seed}.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_summary_is_a_mean_and_a_sample_sd_over_seeds_with_paired_gains(harness, tmp_path):
    for seed, (b, f, x) in zip((42, 43, 44), [(0.60, 0.40, 0.30), (0.70, 0.45, 0.33), (0.80, 0.50, 0.36)]):
        write_run(tmp_path, "base", seed, b)
        write_run(tmp_path, "film", seed, f, guidance_weight=2.0)
        write_run(tmp_path, "xattn", seed, x, guidance_weight=1.5)
    summary = harness.summarize(tmp_path)
    assert list(summary["arms"]) == ["base", "film", "xattn"]               # in the figure's order
    base = summary["arms"]["base"]
    assert base["mae_mean"] == pytest.approx(0.70) and base["mae_sd"] == pytest.approx(0.10)    # ddof = 1
    assert summary["arms"]["film"]["guidance_weight"] == 2.0
    gain = summary["paired"]["film - base"]
    assert gain["per_seed"] == pytest.approx([-0.20, -0.25, -0.30]) and gain["mean"] == pytest.approx(-0.25)
    assert gain["sd"] == pytest.approx(0.05)
    assert (tmp_path / "summary.json").exists()
    assert "0.7000 +- 0.1000" in harness.format_summary(summary)


def test_paired_gains_match_seeds_by_seed_not_by_position(harness, tmp_path):
    """An arm that lacks a seed must not shift the others: 43 is paired with 43, not with 42."""
    for seed, value in ((42, 0.60), (43, 0.70), (44, 0.80)):
        write_run(tmp_path, "base", seed, value)
    for seed, value in ((43, 0.50), (44, 0.55)):
        write_run(tmp_path, "film", seed, value)
    gain = harness.summarize(tmp_path)["paired"]["film - base"]
    assert gain["seeds"] == [43, 44] and gain["per_seed"] == pytest.approx([-0.20, -0.25])


def test_a_directory_of_results_made_under_different_protocols_is_not_summarised(harness, tmp_path):
    write_run(tmp_path, "base", 42, 0.6)
    write_run(tmp_path, "film", 42, 0.4, split="validation")
    with pytest.raises(RuntimeError, match="not all made under one protocol"):
        harness.summarize(tmp_path)


def test_two_results_for_one_arm_and_seed_are_refused(harness, tmp_path):
    write_run(tmp_path, "base", 42, 0.6)
    (tmp_path / "base__seed42__copy.json").write_text((tmp_path / "base__seed42.json").read_text())
    with pytest.raises(RuntimeError, match="two results for one seed"):
        harness.summarize(tmp_path)


def test_a_run_with_no_valid_molecule_summarises_as_nan_rather_than_crashing(harness, tmp_path):
    write_run(tmp_path, "base", 42, None, validity=0.0)
    write_run(tmp_path, "film", 42, None, validity=0.0)             # two such arms: their gain is nan - nan
    write_run(tmp_path, "xattn", 42, 0.3)
    summary = harness.summarize(tmp_path)
    assert np.isnan(summary["arms"]["base"]["mae_mean"]) and "nan" in harness.format_summary(summary)
    assert np.isnan(summary["paired"]["film - base"]["mean"]) and np.isnan(summary["paired"]["xattn - film"]["mean"])


@pytest.mark.parametrize("field", ["arm", "seed", "split", "n_targets", "n_per_target", "sampling",
                                   "guidance_weight", "beta", "size_model", "checkpoints"])
def test_a_result_made_under_any_other_configuration_is_a_different_experiment(harness, field):
    identity = dict(arm="film", seed=42, split="validation", n_targets=100, n_per_target=10,
                    sampling=dict(sample_steps=500, eta=25.0), guidance_weight=2.0, beta=None,
                    size_model=True, checkpoints={"film": "abc"})
    assert harness.same_experiment(identity, dict(identity)) == []
    changed = {**identity, field: {"arm": "xattn", "seed": 43, "split": "test", "n_targets": 50,
                                   "n_per_target": 5, "sampling": dict(sample_steps=100, eta=25.0),
                                   "guidance_weight": 1.5, "beta": 20.0, "size_model": False,
                                   "checkpoints": {"film": "def"}}[field]}
    assert harness.same_experiment(identity, changed) == [field]


# ============================================================================ the command line
def test_an_arm_that_is_missing_something_it_needs_is_refused_before_anything_loads(harness, capsys):
    with pytest.raises(SystemExit):
        harness.main(["run", "--out", "x", "--split", "validation", "--arms", "film", "--base", "b.ckpt",
                      "--film", "f.ckpt", "--size", "s.ckpt"])
    assert "--film-weight" in capsys.readouterr().err                      # no silent default weight
    with pytest.raises(SystemExit):
        harness.main(["run", "--out", "x", "--split", "validation", "--arms", "xattn_fk", "--base", "b.ckpt",
                      "--xattn", "x.ckpt", "--xattn-weight", "1.5", "--size", "s.ckpt"])
    assert "--head" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        harness.main(["run", "--out", "x", "--split", "validation", "--arms", "conditional",
                      "--conditional", "c.ckpt"])
    assert "--size" in capsys.readouterr().err                             # sizes are a choice made on purpose


def test_the_split_has_no_default(harness, capsys):
    with pytest.raises(SystemExit):
        harness.main(["run", "--out", "x", "--arms", "base", "--base", "b.ckpt"])
    assert "--split" in capsys.readouterr().err


# ============================================================================ all five arms, tiny and untrained
@pytest.fixture(scope="module")
def assets(tmp_path_factory, vocab):
    """Untrained checkpoints with the real ZINC dimensions, for every arm."""
    root = tmp_path_factory.mktemp("arms")
    torch.manual_seed(0)
    conditions = torch.randn(400)
    sizes = (7 + (conditions * 1.5).round().clamp(-2, 2)).long()
    size, _ = fit_size_model(conditions, sizes, min_size=int(sizes.min()), max_size=int(sizes.max()),
                             hidden=8, layers=1, epochs=2, seed=0, property_name="logp", property_from="decoded")
    # The parts of a real run are fit on the same ~220k molecules and agree to a few parts in ten thousand
    # of a std; here the size model's own statistics are the reference the others are built from.
    mean, std = float(size.cond_mean.reshape(-1)[0]), float(size.cond_std.reshape(-1)[0])
    base = tiny_model(vocab)
    conditional = tiny_model(vocab, cond_dim=1)
    film = AdaLNAdapter.for_base(base, cond_dim=1, hidden=16, cond_mean=[mean], cond_std=[std], xattn_tokens=0,
                                 cond_fourier=0)
    xattn = AdaLNAdapter.for_base(base, cond_dim=1, hidden=16, cond_mean=[mean], cond_std=[std], xattn_tokens=4,
                                  xattn_dim=8, xattn_heads=2, cond_fourier=2)
    head = PropertyHead(len(vocab.atoms), len(vocab.bonds) + 1, hid=16, layers=2, prop_mean=mean, prop_std=std)
    paths = {name: str(root / f"{name}.ckpt") for name in ("base", "conditional", "film", "xattn", "head", "size")}
    for name, obj in (("base", base), ("conditional", conditional), ("film", film), ("xattn", xattn),
                      ("head", head), ("size", size)):
        obj.save(paths[name])
    (root / "condition.json").write_text(json.dumps(
        {"property": "logp", "property_from": "decoded", "mean": mean, "std": std}), encoding="utf-8")
    paths["conditional"] = str(root / "conditional.ckpt")
    return SimpleNamespace(paths=paths, root=root)


def run_args(assets, out, **overrides):
    p = assets.paths
    args = ["run", "--out", str(out), "--split", "validation", "--seeds", "1", "--targets", "2", "--n", "4",
            "--steps", "3", "--eta", "0", "--device", "cpu", "--base", p["base"], "--conditional", p["conditional"],
            "--film", p["film"], "--film-weight", "1.5", "--xattn", p["xattn"], "--xattn-weight", "1.5",
            "--head", p["head"], "--size", p["size"]]
    for key, value in overrides.items():
        args[args.index(f"--{key}") + 1] = str(value)
    return args


@pytest.fixture
def small_split(monkeypatch):
    fake = SimpleNamespace(val_smiles="CCO CCN CCC CCCO CC(C)O CCOC".split(),
                           test_smiles="CCCN CCCC CC(C)N CNC".split())
    monkeypatch.setattr(zinc_reference, "load_reference_split", lambda *a, **k: fake)


def test_every_arm_runs_writes_its_result_and_is_never_reused_under_another_configuration(
        harness, assets, tmp_path, small_split, capsys):
    out = tmp_path / "results"
    assert harness.main(run_args(assets, out)) == 0
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == [f"{arm}__seed1.json" for arm in sorted(harness.ARMS)]

    records = {arm: json.loads((out / f"{arm}__seed1.json").read_text()) for arm in harness.ARMS}
    for arm, record in records.items():
        assert record["split"] == "validation" and record["n_targets"] == 2 and record["n_per_target"] == 4
        assert len(record["targets"]) == 2 and record["sampling"]["sample_steps"] == 3
        assert record["validity"] is not None and 0.0 <= record["validity"] <= 1.0
    # what each arm depends on is recorded, and only that
    assert {a: r["guidance_weight"] for a, r in records.items()} == {
        "base": None, "conditional": 1.0, "film": 1.5, "xattn": 1.5, "xattn_fk": 1.5}
    assert {a: r["beta"] for a, r in records.items()} == {
        "base": None, "conditional": None, "film": None, "xattn": None, "xattn_fk": 20.0}
    assert {a: r["size_model"] for a, r in records.items()} == {
        "base": False, "conditional": True, "film": True, "xattn": True, "xattn_fk": True}
    assert set(records["base"]["checkpoints"]) == {"base"}
    assert set(records["xattn_fk"]["checkpoints"]) == {"base", "xattn", "head", "size"}
    assert set(records["conditional"]["checkpoints"]) == {"conditional", "conditional_stats", "size"}
    # every arm saw the same targets
    assert len({tuple(r["targets"]) for r in records.values()}) == 1

    before = {p.name: p.stat().st_mtime_ns for p in out.glob("*.json")}
    capsys.readouterr()
    harness.main(run_args(assets, out))                                  # the same again: nothing is redone
    assert capsys.readouterr().out.count("done already") == len(harness.ARMS)
    assert before == {p.name: p.stat().st_mtime_ns for p in out.glob("*.json")}

    with pytest.raises(RuntimeError, match="another configuration"):
        harness.main(run_args(assets, out, **{"film-weight": 2.0}))      # a different weight is a different run
    assert harness.main(["summarize", str(out)]) == 0
    assert "| unconditional base |" in capsys.readouterr().out


def test_the_test_split_draws_other_targets_than_validation(harness, assets, tmp_path, small_split):
    validation, held_out = tmp_path / "val", tmp_path / "test"
    harness.main(run_args(assets, validation, **{"split": "validation"}) + ["--arms", "base"])
    harness.main(run_args(assets, held_out, **{"split": "test"}) + ["--arms", "base"])
    val = json.loads((validation / "base__seed1.json").read_text())
    test = json.loads((held_out / "base__seed1.json").read_text())
    assert val["split"] == "validation" and test["split"] == "test"
    assert set(val["targets"]).isdisjoint(test["targets"])
