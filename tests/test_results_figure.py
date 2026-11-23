"""The README's results figure (docs/assets/results_logp.py) is drawn from a data file, not from numbers in the script.

The point of that: a chart cannot drift from the summary it shows, and the numbers quoted around it (the README's
alt text, docs/results.md) can be checked against the same file. These tests cover the script's contract with
synthetic numbers (clearly not results); the last one checks the committed data file against the text around it.
"""
import importlib.util
import json
import re
import statistics
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


def hex_to_rgb(color: str):
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def read_png(path):
    import matplotlib.image as mpimg
    return mpimg.imread(path)[:, :, :3]


def close(a, b, tolerance=0.012) -> bool:
    return max(abs(x - y) for x, y in zip(a, b)) < tolerance


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_the_figure_is_drawn_from_its_data_file_in_each_theme(figure, tmp_path, theme):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(synthetic()), encoding="utf-8")
    out = tmp_path / "figure.png"
    figure.main(data, out, theme)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 10_000
    assert close(read_png(out)[0, 0], hex_to_rgb(figure.THEMES[theme]["surface"])), "the theme did not reach the picture"


def test_an_unknown_theme_is_refused(figure, tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(synthetic()), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown theme"):
        figure.main(data, tmp_path / "figure.png", "sepia")


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


def recorded_spread() -> str:
    """The difference between two trainings of one adapter recipe, as docs/caveats.md records it."""
    caveats = (ROOT / "docs" / "caveats.md").read_text(encoding="utf-8")
    return re.search(r"Two seeds of the same adapter recipe differ by (\d\.\d{3}) in logP error", caveats).group(1)


def gap_to_end_to_end(arm: str) -> float:
    arms = committed()["arms"]
    return arms[arm]["mae_mean"] - arms["conditional"]["mae_mean"]


def paired(a: str, b: str):
    """Mean and sample sd of the per-seed difference a - b; the seeds share their target sets across approaches."""
    arms = committed()["arms"]
    diff = [x - y for x, y in zip(arms[a]["mae_per_seed"], arms[b]["mae_per_seed"])]
    return statistics.mean(diff), statistics.stdev(diff)


def readme_caption() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = readme[readme.index("Results at a glance"):]
    return results[:results.index("\n## ")]


def test_the_caveat_that_names_the_end_to_end_gap_quotes_the_data_file_and_the_side_of_the_spread_each_gap_is_on():
    arms, spread = committed()["arms"], recorded_spread()
    caveats = (ROOT / "docs" / "caveats.md").read_text(encoding="utf-8")
    bullet = caveats[caveats.index("**The model trained end to end had the lowest error"):]
    bullet = bullet[:bullet.index("\n* **")]
    for arm in ("conditional", "xattn_fk", "xattn"):
        assert f"{arms[arm]['mae_mean']:.4f}" in bullet, (arm, bullet)
    alone, steered = f"{gap_to_end_to_end('xattn'):.3f}", f"{gap_to_end_to_end('xattn_fk'):.3f}"
    assert float(steered) < float(spread) < float(alone), "the bullet's 'more than' and 'less than' would be wrong"
    assert f"a gap of {alone}, more than the {spread} recorded" in bullet
    assert f"a gap of {steered}, less than that" in bullet


def test_the_readme_caption_quotes_the_gap_to_feynman_kac_and_the_spread_it_is_measured_against():
    gap, spread = f"{gap_to_end_to_end('xattn_fk'):.3f}", recorded_spread()
    assert float(gap) < float(spread), "the caption's 'smaller than' would be wrong: the sentence needs rewriting"
    caption = " ".join(readme_caption().split())
    assert f"the {gap} between the end-to-end model and the steered adapter" in caption
    assert f"smaller than the {spread} recorded between two trainings" in caption


def test_the_every_seed_claims_of_the_text_hold_in_the_data_file():
    """'Below every other approach on every sampling seed', 'beats FiLM on all three seeds', 'improves ... on all three'."""
    arms = committed()["arms"]
    for i, seed in enumerate(arms["base"]["seeds"]):
        mae = {arm: arms[arm]["mae_per_seed"][i] for arm in ARMS}
        assert mae["conditional"] < min(v for arm, v in mae.items() if arm != "conditional"), seed
        assert mae["xattn"] < mae["film"] and mae["xattn_fk"] < mae["xattn"], seed


def test_the_reading_bullets_of_section_nine_quote_the_paired_differences_and_shares_of_the_data_file():
    arms, section = committed()["arms"], section_nine()
    reading = section[section.index("**Reading it.**"):section.index("### Protocol")]
    for a, b in (("xattn_fk", "conditional"), ("xattn", "conditional"), ("film", "xattn"), ("xattn_fk", "xattn")):
        mean, sd = paired(a, b)
        assert f"{abs(mean):.4f} ± {sd:.4f}" in reading, f"{a} against {b}: {abs(mean):.4f} ± {sd:.4f} is not in the bullets"
    base, e2e = arms["base"]["mae_mean"], arms["conditional"]["mae_mean"]
    share = {arm: f"{100 * (base - arms[arm]['mae_mean']) / (base - e2e):.0f}" for arm in ("film", "xattn", "xattn_fk")}
    quoted = {"xattn": re.search(r"cross-attention adapter closes about (\d+)%", reading).group(1),
              "xattn_fk": re.search(r"Feynman-Kac about (\d+)%", reading).group(1),
              "film": re.search(r"training budget, about (\d+)%", reading).group(1)}
    assert quoted == share


def test_no_text_calls_the_end_to_end_model_the_most_accurate_while_its_lead_over_feynman_kac_is_inside_the_spread():
    """The sentence the first version of this section carried; the section's own caveats say it is not established."""
    gap = gap_to_end_to_end("xattn_fk")
    if gap >= float(recorded_spread()):
        pytest.skip("the lead is larger than the recorded spread, so the plain wording would be supported")
    for name in ("README.md", "docs/results.md", "docs/caveats.md"):
        text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
        for claim in ("most accurate", "more accurate than any"):
            assert claim not in text, f"{name} says {claim!r}, but the lead over Feynman-Kac ({gap:.3f}) is inside the spread"


@pytest.mark.parametrize("theme, name", [("light", "results_logp.png"), ("dark", "results_logp_dark.png")])
def test_each_committed_figure_has_its_theme_and_bars_that_end_where_the_data_file_says(figure, theme, name):
    """Measured on the pixels of the committed file: a figure that was not redrawn after the data (or the palette)
    changed, or one committed under the other theme's name, is caught here rather than by a reader."""
    image, arms, colors = read_png(ASSETS / name), committed()["arms"], figure.THEMES[theme]
    assert close(image[0, 0], hex_to_rgb(colors["surface"])), f"{name} does not have the {theme} surface"
    top = max(arms[arm]["mae_mean"] + (arms[arm]["mae_sd"] or 0.0) for arm, _ in figure.ROWS)
    scale = figure.MAX_BAR / top
    centres = figure.layout()[1]
    for (arm, _), y in zip(figure.ROWS, centres):
        color = hex_to_rgb(colors["accent"] if arm == figure.HEADLINE else colors["context"])
        row = image[int(round((y + 7) * 2))]      # 7 px below the bar's centre: under the whisker's caps, above the rounded corner
        solid = [x for x in range(row.shape[0]) if close(row[x], color)]
        assert solid, f"{name}: no {arm} bar found"
        expected = (figure.BAR_X0 + arms[arm]["mae_mean"] * scale) * 2     # the file is written at twice the layout's pixels
        assert abs(max(solid) - expected) <= 3, f"{name}: the {arm} bar ends at {max(solid)} px, the data says {expected:.0f}"


def test_the_readme_serves_the_dark_figure_to_dark_pages_and_the_light_one_otherwise():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme[readme.index("Results at a glance"):]
    block = block[block.index("<picture>"):block.index("</picture>")]
    assert 'media="(prefers-color-scheme: dark)" srcset="docs/assets/results_logp_dark.png"' in block
    assert block.index("<source") < block.index('<img src="docs/assets/results_logp.png"'), "the fallback image must follow the source"
    for name in ("results_logp.png", "results_logp_dark.png"):
        assert (ASSETS / name).exists(), name
