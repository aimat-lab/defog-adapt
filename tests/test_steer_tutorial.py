"""The tutorial on the "Steer toward a target" page.

The page walks through one script, ``docs/guides/steer_toward_a_target.py``, in pieces: each piece is a section of that file,
included by marker, and the whole file is the last thing on the page. So the page and its code can drift apart in a few
ways, and each has a test here. A piece that no longer runs: the script is run, unchanged, on tiny checkpoints. A piece
that is missing, shown twice or out of order. Code in the script that no step shows. Code pasted into the page, which would
be a second copy. A final script that is not the whole file, or not last. An optional step that stops saying it is optional.
And the judge, which tiny untrained models cannot exercise (they decode to nothing), is tested on molecules whose answers
are known.
"""
import contextlib
import io
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from _snippets import INCLUDE, MARKER, included, sections    # tests/ has no __init__.py; pytest prepends this dir

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "guides" / "steer-toward-a-target.md"
SCRIPT = ROOT / "docs" / "guides" / "steer_toward_a_target.py"
NAME = "docs/guides/steer_toward_a_target.py"
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="docs/ is not part of an installed wheel")

#: the four files the script loads, as its header says
CHECKPOINTS = {"base.ckpt", "logp_adapter.ckpt", "logp_head.ckpt", "logp_head_size.ckpt"}
#: steps 1 to 4 build the proposal, which is required; every layer after it is optional (as in the figure at the top)
REQUIRED_STEPS = {1, 2, 3, 4}


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


# =============================================================================================== the page and the script
def test_the_page_only_includes_code_it_never_pastes_it():
    """One copy of the code: every python block on the page is a single include line, nothing typed."""
    blocks = re.findall(r"^```python\n(.*?)^```", page_text(), flags=re.S | re.M)
    assert len(blocks) >= 10, f"the page shows {len(blocks)} python blocks: has its fence syntax changed?"
    for body in blocks:
        assert INCLUDE.fullmatch(body.strip()), f"a python block on the page holds typed code, not an include:\n{body}"


def test_every_section_of_the_script_is_on_the_page_once_and_in_file_order():
    on_page = [section for name, section in INCLUDE.findall(page_text()) if name == NAME and section]
    in_script = sections(script_text())
    assert len(in_script) >= 10, f"the script has {len(in_script)} sections: has its marker syntax changed?"
    assert on_page == in_script, f"the page includes {on_page}, the script has {in_script} (in this order, each once)"


def test_the_whole_script_is_the_last_thing_on_the_page_and_is_there_once():
    page = page_text()
    whole = [m for m in INCLUDE.finditer(page) if m.group(1) == NAME and not m.group(2)]
    assert len(whole) == 1, f"the whole script is included {len(whole)} times, it should be once"
    assert page[:whole[0].start()].rstrip().endswith("```python"), "the whole script is not in a python block"
    assert page[whole[0].end():].strip() == "```", "something follows the final script on the page"
    headings = re.findall(r"^#+ (.*)$", page, flags=re.M)
    assert headings[-1] == "The final script", f"the last heading is {headings[-1]!r}, not 'The final script'"


def test_no_code_in_the_script_goes_unshown():
    """Every line of code lies in a section, so the steps show everything that the final script runs."""
    inside = False
    for number, line in enumerate(script_text().splitlines(), 1):
        marker = MARKER.match(line)
        if marker:
            inside = marker.group(1) == "start"
        elif not inside and line.strip() and not line.lstrip().startswith("#"):
            pytest.fail(f"{SCRIPT.name}:{number}: `{line}` is outside every section, so no step shows it")


def test_the_steps_are_numbered_and_the_optional_ones_say_so():
    steps = re.findall(r"^## (\d+)\. (.*)$", page_text(), flags=re.M)
    assert [int(number) for number, _ in steps] == list(range(1, 9)), f"the steps are {steps}"
    for number, title in steps:
        said_optional = title.endswith("(optional)")
        assert said_optional == (int(number) not in REQUIRED_STEPS), \
            f"step {number} '{title}': required steps are {sorted(REQUIRED_STEPS)}, and an optional one says '(optional)'"


def test_the_script_loads_exactly_the_files_its_header_names():
    text = script_text()
    loaded = set(re.findall(r'\.load\("([^"]+)"\)', text))
    assert loaded == CHECKPOINTS, f"the script loads {sorted(loaded)}"
    header = text.split("# --8<--")[0]
    for name in CHECKPOINTS:
        assert name in header, f"the header of the script does not name {name}"


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
    """The script's own namespace and its printed output, after running it, unchanged, in a folder with tiny checkpoints."""
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    from defog_adapt.presets import ADAPTER_SAMPLING

    folder = tmp_path_factory.mktemp("steer")
    write_tiny_checkpoints(folder)
    namespace, out = {"__name__": "__main__"}, io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.chdir(folder)
        patch.setitem(ADAPTER_SAMPLING, "sample_steps", 3)           # toy scale: the recorded 500 steps would be minutes
        exec(compile(script_text(), str(SCRIPT), "exec"), namespace)
    return namespace, out.getvalue()


def test_the_script_runs_unchanged_and_makes_the_three_runs(ran):
    namespace, output = ran
    for run in ("graphs", "graphs_sized", "graphs_fk"):
        assert len(namespace[run]) == 10, f"{run} is not a list of 10 graphs"
    for label in ("adapter only", "+ size model", "+ Feynman-Kac"):
        assert any(line.startswith(label) for line in output.splitlines()), f"the judge printed no line for '{label}'"


def test_the_trace_has_the_fields_the_page_documents(ran):
    namespace, _ = ran
    documented = re.findall(r"^\| `(\w+)` \|", page_text(), flags=re.M)
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
    exec(included(script_text(), "judge_fn"), namespace)
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
