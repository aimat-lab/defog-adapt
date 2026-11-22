"""The README's results figure (docs/assets/results_logp.py) is drawn from a data file, not from numbers in the script.

The point of that: a chart cannot drift from the summary it shows, and the numbers quoted around it (the README's
alt text, docs/results.md) can be checked against the same file. These tests cover the script's contract with
synthetic numbers (clearly not results); the last one checks the committed data file against the text around it.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
pytestmark = pytest.mark.skipif(not (ASSETS / "results_logp.py").exists(),
                                reason="docs/ is not part of an installed wheel")

matplotlib = pytest.importorskip("matplotlib")

ARMS = ("base", "conditional", "film", "xattn", "xattn_fk")


@pytest.fixture(scope="module")
def figure():
    name = "results_logp_under_test"
    spec = importlib.util.spec_from_file_location(name, ASSETS / "results_logp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def synthetic(**missing):
    """Five arms with made-up numbers: they exercise the drawing, and mean nothing."""
    arms = {arm: {"seeds": [1, 2, 3], "mae_mean": 1.0 - 0.15 * k, "mae_sd": 0.02}
            for k, arm in enumerate(ARMS) if arm not in missing}
    return {"protocol": {"split": "test", "n_targets": 100, "n_per_target": 10, "sampling": {"sample_steps": 500}},
            "arms": arms}


def test_the_figure_is_drawn_from_its_data_file(figure, tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(synthetic()), encoding="utf-8")
    out = tmp_path / "figure.png"
    figure.main(data, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 10_000


@pytest.mark.parametrize("arm", ARMS)
def test_a_data_file_that_lacks_an_arm_is_refused(figure, tmp_path, arm):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(synthetic(**{arm: True})), encoding="utf-8")
    with pytest.raises(SystemExit, match=arm):
        figure.main(data, tmp_path / "figure.png")


def test_the_rows_are_the_references_first_and_the_accent_is_feynman_kac(figure):
    assert [arm for arm, _ in figure.ROWS] == list(ARMS)
    assert figure.N_REFERENCES == 2 and figure.HEADLINE == "xattn_fk"


# ============================================================================ the text around the figure
def committed():
    return json.loads((ASSETS / "results_logp.json").read_text(encoding="utf-8"))


def test_the_committed_data_file_has_all_five_arms_and_no_absolute_path():
    data = committed()
    assert [arm for arm in ARMS if arm in data["arms"]] == list(ARMS)
    assert data["protocol"]["split"] == "test" and len(data["arms"]["base"]["seeds"]) == 3
    # Absolute paths only: the specific private names are not listed in a published file (the repository's own scan, whose
    # patterns live outside it, covers this data file too).
    text = (ASSETS / "results_logp.json").read_text(encoding="utf-8")
    assert "/home/" not in text and "/media/" not in text


def test_the_readme_alt_text_quotes_exactly_the_means_of_the_data_file():
    """The alt text is read instead of the picture by anyone who cannot see it, so its numbers must be the figure's."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    alt = re.search(r'results_logp\.png" alt="([^"]*)"', readme).group(1)
    quoted = re.findall(r"\d\.\d{3}", alt)
    expected = [f"{committed()['arms'][arm]['mae_mean']:.3f}" for arm in ARMS]
    assert sorted(quoted) == sorted(expected), (quoted, expected)


def test_the_readme_quotes_the_feynman_kac_numbers_of_the_data_file_in_the_caption_and_in_the_table_row():
    """Two places quote the diversity cost; checked separately, or one of them could drift while the other still matches."""
    arms, readme = committed()["arms"], (ROOT / "README.md").read_text(encoding="utf-8")
    uniqueness = f"uniqueness {arms['xattn_fk']['uniqueness_mean']:.2f}"
    results = readme[readme.index("Results at a glance"):]
    caption = results[:results.index("\n## ")]
    assert uniqueness in caption, f"the caption under the figure does not say {uniqueness!r}"
    row = next(line for line in readme.splitlines() if line.startswith("| **Feynman-Kac steering**"))
    assert uniqueness in row and f"logP error {arms['xattn']['mae_mean']:.3f} to {arms['xattn_fk']['mae_mean']:.3f}" in row


def section_nine():
    text = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    return text[text.index("## 9."):]


def test_the_results_table_in_the_docs_is_the_data_file():
    section = section_nine()
    for arm in ARMS:
        a = committed()["arms"][arm]
        row = next(line for line in section.splitlines() if line.startswith(f"| {a['label']} |"))
        for cell in (f"**{a['mae_mean']:.4f}** ± {a['mae_sd']:.4f}", f"{a['validity_mean']:.3f}", f"{a['uniqueness_mean']:.3f}"):
            assert cell in row, f"{a['label']}: {cell!r} is not in {row!r}"


def test_the_caveat_that_names_the_end_to_end_gap_quotes_the_data_file():
    arms = committed()["arms"]
    caveats = (ROOT / "docs" / "caveats.md").read_text(encoding="utf-8")
    bullet = caveats[caveats.index("**A model trained end to end is more accurate"):]
    bullet = bullet[:bullet.index("\n* **")]
    for arm in ("conditional", "xattn_fk", "xattn"):
        assert f"{arms[arm]['mae_mean']:.4f}" in bullet, (arm, bullet)
