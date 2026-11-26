"""The three guides, run one after the other on tiny data.

The base guide writes ``base.ckpt``; the adapter guide reads it and writes the adapter, the size model and the head; the steer
guide loads all four. Run in one folder at toy scale, the chain shows that each guide's output is what the next one loads,
and that the models the guides build fit together. The only change to a script is its scale: the recorded sizes and epochs
would take hours, so they are shrunk in the text of the script (and a test fails if one is no longer there to shrink).
Untrained tiny models decode to almost nothing, so what is asserted about the results is their shape: the files, the
lengths, the keys and the printed lines the pages describe.
"""
import contextlib
import io
import os
import re
from types import SimpleNamespace

import pytest

from _snippets import included                                      # tests/ has no __init__.py; pytest prepends this dir
from _tutorials import BY_KEY, TUTORIALS

pytestmark = pytest.mark.skipif(not all(t.script_path.is_file() for t in TUTORIALS),
                                reason="docs/ is not part of an installed wheel")

TOY_MOLECULES = """
    CCO CCCO CCCCO CCN CCCN CCOC CCCOC CC(C)O CC(C)N CC(C)C CC(=O)C CC(=O)O CC(=O)N NCCO NCCN OCCO
    CNC CCNCC CN(C)C C1CCCCC1 C1CCCC1 C1CCNCC1 C1CCOCC1 OC1CCCCC1 c1ccccc1 Cc1ccccc1 CCc1ccccc1 Oc1ccccc1
    Nc1ccccc1 COc1ccccc1 c1ccncc1 Cc1ccncc1 OCc1ccccc1 CCS CCCl CCF CCBr CCCCC CCCCCC CCCCCCC CCOCC CCCOCC
    CCC(=O)O CCC(=O)N CCCC(=O)C CNCC CCNC
""".split()

#: what a toy-scale run changes in the text of each script, with how many times the text occurs: the model's size, the epochs,
#: the sampling steps, the number of samples, the size of the cleanup
SHRINK = {
    "base": {
        "max_epochs=20": ("max_epochs=2", 1), "n_layers=4": ("n_layers=2", 1), "hidden_dim=64": ("hidden_dim=16", 1),
        "hidden_mlp_dim=128": ("hidden_mlp_dim=32", 1), "n_heads=4": ("n_heads=2", 1), "rrwp_steps=20": ("rrwp_steps=4", 1),
        "sample_steps=500": ("sample_steps=3", 1), "sampler.sample(100)": ("sampler.sample(8)", 1),
        "rollout_size=128": ("rollout_size=8", 1), "sample_steps=100": ("sample_steps=3", 1),
        "subsample_steps=12": ("subsample_steps=2", 1), "minibatch_size=16": ("minibatch_size=4", 1),
        "trainer.fit(25,": ("trainer.fit(3,", 1),
    },
    "adapter": {
        "hidden=256": ("hidden=32", 1), "xattn_tokens=64": ("xattn_tokens=4", 1), "xattn_dim=128": ("xattn_dim=16", 1),
        "xattn_heads=16": ("xattn_heads=2", 1), "max_epochs=80": ("max_epochs=2", 1), "sample_steps=500": ("sample_steps=3", 1),
        "sampler.sample(100)": ("sampler.sample(8)", 1), "hidden=512": ("hidden=16", 1), "epochs=200": ("epochs=3", 1),
        "hid=128": ("hid=16", 1), "epochs=60": ("epochs=2", 1),
    },
    "steer": {"sample_steps=500": ("sample_steps=3", 2)},
}


def shrunk(code: str, changes: dict) -> str:
    for old, (new, count) in changes.items():
        assert code.count(old) == count, f"the script has {code.count(old)} of `{old}`, this test shrinks {count}"
        code = code.replace(old, new)
    return code


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Run the three scripts in order in one folder; each key holds the script's namespace and what it printed."""
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    folder = tmp_path_factory.mktemp("chain")
    (folder / "molecules.smi").write_text("\n".join(TOY_MOLECULES) + "\n")
    ran = {"folder": folder}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        for key in ("base", "adapter", "steer"):
            tutorial = BY_KEY[key]
            namespace, out = {"__name__": "__main__"}, io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(shrunk(tutorial.script_text(), SHRINK[key]), str(tutorial.script_path), "exec"), namespace)
            ran[key] = (namespace, out.getvalue())
    return ran


def test_each_guide_writes_the_files_it_says(chain):
    written = {"base": ("base.ckpt", "base_rl1.ckpt", "base_clean.ckpt"),
               "adapter": ("logp_adapter.ckpt", "logp_head_size.ckpt", "logp_head.ckpt")}
    for key, names in written.items():
        for name in names:
            assert (chain["folder"] / name).is_file(), f"the {key} guide did not write {name}"


def test_the_steer_guide_runs_on_what_the_other_two_wrote(chain):
    namespace, _ = chain["steer"]
    for run in ("graphs", "graphs_sized", "graphs_fk"):
        assert len(namespace[run]) == 10, f"the steer guide's {run} is not a list of 10 graphs"
    assert namespace["fk_sampler"].trace, "the Feynman-Kac run recorded no checkpoint"


def test_the_base_guide_reports_the_best_epoch_and_the_metrics_the_page_names(chain):
    namespace, output = chain["base"]
    assert namespace["best"].best_epoch is not None and namespace["best"].best < float("inf")
    assert re.search(r"^\S+ \d+$", output, flags=re.M), "the script did not print the best loss and the best epoch"
    metrics = namespace["domain"].metrics(namespace["generated"], compute_kl=False)
    assert {"validity", "uniqueness", "novelty", "sanity"} <= set(metrics), f"the page names four metrics, got {sorted(metrics)}"


def test_the_cleanup_leaves_the_base_alone(chain):
    """The page says the cleanup does not touch base.ckpt: so base.ckpt is older than the first round's result."""
    folder = chain["folder"]
    assert (folder / "base.ckpt").stat().st_mtime_ns < (folder / "base_rl1.ckpt").stat().st_mtime_ns


def test_the_adapter_check_prints_a_line_per_target(chain):
    _, output = chain["adapter"]
    lines = [line for line in output.splitlines() if line.startswith("target ")]
    assert len(lines) == 2, f"the check should print one line for each of its two targets, got {lines}"


def test_the_size_model_reports_the_two_numbers_the_page_names(chain):
    namespace, _ = chain["adapter"]
    assert {"gain_nats", "shrink"} <= set(namespace["metrics"])


def test_the_head_check_prints_its_error_and_the_spread_of_the_labels(chain):
    """Two lines of the adapter guide have the shape of two numbers: the size model's (gain, shrink), then the head's (error,
    spread). The head's second number is the standard deviation of the labels, which the page says it is."""
    namespace, output = chain["adapter"]
    number = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf"
    pairs = re.findall(rf"^({number}) ({number})$", output, flags=re.M)
    assert len(pairs) == 2, f"expected the size model's two numbers and the head check's two, got {pairs}"
    gain, shrink = map(float, pairs[0])
    assert gain == pytest.approx(namespace["metrics"]["gain_nats"], rel=1e-5)
    assert shrink == pytest.approx(namespace["metrics"]["shrink"], rel=1e-5)
    error, spread = map(float, pairs[1])
    assert spread == pytest.approx(namespace["labels"].std().item(), rel=1e-5)
    assert error >= 0


def test_labels_are_the_property_of_the_decoded_graph_not_of_the_source_smiles(tmp_path, monkeypatch):
    """The warning on the adapter page, as a test: a graph keeps no formal charge, so the molecule it represents is not the
    one that was written down, and the label has to follow the graph."""
    pytest.importorskip("torch")
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import Crippen

    monkeypatch.chdir(tmp_path)
    (tmp_path / "molecules.smi").write_text("C[NH3+]\nCCO\nc1ccccc1\n")
    script = BY_KEY["adapter"].script_text()
    namespace = {"base": SimpleNamespace(max_nodes=60)}                  # the label step asks the base only for its size limit
    exec(included(script, "graphs") + included(script, "label"), namespace)
    domain = namespace["domain"]
    label_of = {Chem.MolToSmiles(domain.decode(graph)): graph.cond.item() for graph in namespace["labelled"]}

    methylamine = Chem.MolFromSmiles("CN")                               # what the graph of C[NH3+] represents
    assert label_of["CN"] == pytest.approx(Crippen.MolLogP(methylamine))
    source = Crippen.MolLogP(Chem.MolFromSmiles("C[NH3+]"))
    assert abs(label_of["CN"] - source) > 0.01, "the source and the decoded label would not differ here: this test proves nothing"
