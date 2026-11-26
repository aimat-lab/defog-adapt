"""What is specific to the "Steer toward a target" tutorial.

The structure of its page (pieces, order, the final script, the optional steps) is checked with the other guides in
test_tutorial_pages.py, its settings against the presets in test_tutorial_settings.py, and the three guides run in a chain in
test_tutorial_chain.py. Here its script is run on its own, on tiny checkpoints that are not the other guides' output, and
its judge is tested on molecules whose answers are known, because tiny untrained models decode to nothing.
"""
import contextlib
import io
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from _snippets import included                                      # tests/ has no __init__.py; pytest prepends this dir
from _tutorials import BY_KEY

TUTORIAL = BY_KEY["steer"]
pytestmark = pytest.mark.skipif(not TUTORIAL.script_path.is_file(), reason="docs/ is not part of an installed wheel")

#: the four files the script loads
CHECKPOINTS = {"base.ckpt", "logp_adapter.ckpt", "logp_head.ckpt", "logp_head_size.ckpt"}


def test_the_script_loads_exactly_the_four_files_of_the_guides_before_it():
    loaded = set(re.findall(r'\.load\("([^"]+)"\)', TUTORIAL.script_text()))
    assert loaded == CHECKPOINTS, f"the script loads {sorted(loaded)}"


# ======================================================================================== the script, run for real
def write_tiny_checkpoints(folder: Path) -> None:
    """The four files the script loads, as tiny untrained models for the ZINC vocabulary (written and read back by the
    library's own save and load, so that a change to a checkpoint format is caught too)."""
    import torch
    from defog_adapt.core import CrossAttentionAdapter, DeFoGModel, LearnedSizeDistribution, PropertyHead
    from defog_adapt.data import zinc_reference

    atoms, bonds = len(zinc_reference.ATOM_TYPES), len(zinc_reference.BOND_TYPES) + 1
    torch.manual_seed(0)
    base = DeFoGModel(num_node_classes=atoms, num_edge_classes=bonds, n_layers=2, hidden_dim=16, hidden_mlp_dim=32,
                      n_heads=2, dropout=0.0, max_nodes=12, sample_steps=3, noise_type="uniform",
                      node_counts=torch.tensor([0.0] * 5 + [1.0] * 6 + [0.0]))
    base.save(str(folder / "base.ckpt"))
    CrossAttentionAdapter.for_base(base, cond_dim=1, hidden=16, xattn_tokens=2, xattn_dim=16, xattn_heads=2,
                                   cond_mean=[2.0], cond_std=[1.0]).save(str(folder / "logp_adapter.ckpt"))
    PropertyHead(atoms, bonds, hid=16, layers=2, prop_mean=2.0, prop_std=1.0).save(str(folder / "logp_head.ckpt"))
    LearnedSizeDistribution(1, 5, 15, hidden=32, layers=2, cond_mean=[2.0], cond_std=[1.5],
                            marginal=torch.softmax(torch.zeros(11), 0)).save(str(folder / "logp_head_size.ckpt"))


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """The script's own namespace and its printed output, after running it in a folder with tiny checkpoints. The only change
    is toy scale: the recorded 500 steps would take minutes, so they are shrunk in the text (and the test fails if they
    are no longer there to shrink)."""
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")

    code = TUTORIAL.script_text()
    assert code.count("sample_steps=500") == 2, "the script no longer has the two `sample_steps=500` this test shrinks"
    code = code.replace("sample_steps=500", "sample_steps=3")
    folder = tmp_path_factory.mktemp("steer")
    write_tiny_checkpoints(folder)
    namespace, out = {"__name__": "__main__"}, io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.chdir(folder)
        exec(compile(code, str(TUTORIAL.script_path), "exec"), namespace)
    return namespace, out.getvalue()


def test_the_script_runs_unchanged_and_makes_the_three_runs(ran):
    namespace, output = ran
    for run in ("graphs", "graphs_sized", "graphs_fk"):
        assert len(namespace[run]) == 10, f"{run} is not a list of 10 graphs"
    for label in ("adapter only", "+ size model", "+ Feynman-Kac"):
        assert any(line.startswith(label) for line in output.splitlines()), f"the judge printed no line for '{label}'"


def test_the_trace_has_the_fields_the_page_documents(ran):
    namespace, _ = ran
    table = re.search(r"^\| Field \|.*?(?:\n\n|\Z)", TUTORIAL.page_text(), flags=re.S | re.M)
    assert table, "the page has no table headed 'Field' for the trace"
    documented = re.findall(r"^\| `(\w+)` \|", table.group(0), flags=re.M)
    assert {"ess_over_k", "logw_range", "resampled"} <= set(documented), f"the page's table documents {documented}"
    trace = namespace["fk_sampler"].trace
    assert trace, "the Feynman-Kac run recorded no checkpoint"
    for record in trace:
        assert set(documented) <= set(record), f"a trace record lacks a documented field: {sorted(record)} vs {documented}"


# ====================================================================================================== the judge
def test_the_judge_measures_validity_uniqueness_and_error_on_decoded_molecules():
    """Known answers: tiny untrained models decode to nothing, so the run above cannot tell a right judge from a wrong one."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import Crippen

    namespace = {"domain": SimpleNamespace(decode=lambda graph: graph)}      # a graph here is already its molecule, or None
    exec(included(TUTORIAL.script_text(), "judge_fn"), namespace)
    judge = namespace["judge"]
    ethanol, benzene = Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")
    target = 1.0

    result = judge([ethanol, ethanol, None, benzene], target)
    assert result["validity"] == pytest.approx(3 / 4)                        # one of four does not decode
    assert result["uniqueness"] == pytest.approx(2 / 3)                      # of the three valid, two are one molecule
    errors = [abs(Crippen.MolLogP(mol) - target) for mol in (ethanol, ethanol, benzene)]
    assert result["mean_abs_error"] == pytest.approx(sum(errors) / 3)        # absolute: ethanol is below the target, benzene above

    nothing = judge([None, None], target)
    assert nothing["validity"] == 0.0
    assert math.isnan(nothing["uniqueness"]) and math.isnan(nothing["mean_abs_error"]), \
        "with no valid molecule there is no uniqueness and no error to report, and a number would claim there is"
