"""The documentation site stays buildable, short and portable.

``mkdocs.yml`` publishes the high-level pages of ``docs/`` and leaves the detailed notes (adapters, steering, RL,
caveats, results, migration) in the repository. Four things go wrong quietly with a site like that: a page is added
and never reaches the nav, a detailed note is published by a typo in the exclusion list, the configuration grows a
feature the successor of Material for MkDocs cannot read, and the workflow gains a way to publish by accident.

Most checks read files, so they run anywhere. The build itself runs only where the docs dependency group is installed
(``pip install --group docs``); CI builds the site in its own workflow.
"""
import importlib.util
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10: pytest itself depends on the backport
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "mkdocs.yml"
WORKFLOW_FILE = ROOT / ".github" / "workflows" / "docs.yml"

#: the detailed notes: readable in the repository, not published on the site
DETAILED_NOTES = {"adapters.md", "steering.md", "rl.md", "caveats.md", "results.md", "migration.md"}

pytestmark = pytest.mark.skipif(not CONFIG_FILE.exists(), reason="mkdocs.yml is not part of an installed wheel")


def config():
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))      # safe_load: a python tag would be refused


def nav_entries(items):
    """Every (title, target) in the nav, however deeply it is nested."""
    for item in items:
        if isinstance(item, str):
            yield None, item
            continue
        for title, value in item.items():
            if isinstance(value, list):
                yield from nav_entries(value)
            else:
                yield title, value


def is_url(target):
    return re.match(r"^[a-z][a-z0-9+.-]*://", target) is not None


LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")           # a link or an image, in Markdown


def nav_pages():
    """The pages in the nav, relative to docs/ (external links are not pages)."""
    return {target for _, target in nav_entries(config()["nav"]) if not is_url(target)}


def excluded_files():
    """What ``exclude_docs`` removes by an exact, anchored path (``/results.md``), relative to docs/.

    Patterns with a wildcard are left to the build itself; this reads only the entries that name one file.
    """
    entries = [line.strip() for line in config()["exclude_docs"].splitlines()]
    return {e.lstrip("/") for e in entries if e.startswith("/") and not re.search(r"[*?\[]", e)}


def markdown_pages():
    return {p.relative_to(ROOT / "docs").as_posix() for p in (ROOT / "docs").rglob("*.md")}


def built_path(site, relative):
    """Where the build puts a docs file: a page ``a/b.md`` becomes ``a/b/index.html``, anything else stays as it is."""
    if relative == "index.md":
        return site / "index.html"
    return site / (relative[:-3] + "/index.html") if relative.endswith(".md") else site / relative


# ============================================================ the pages and the nav
def test_every_published_page_is_in_the_nav_and_every_nav_page_exists():
    in_nav = nav_pages()
    published = markdown_pages() - excluded_files()
    assert not in_nav - markdown_pages(), f"the nav names pages that do not exist: {sorted(in_nav - markdown_pages())}"
    assert not published - in_nav, f"pages that would be published but are not in the nav: {sorted(published - in_nav)}"
    assert not in_nav & excluded_files(), f"the nav lists pages that exclude_docs removes: {sorted(in_nav & excluded_files())}"
    assert len(in_nav) >= 10, "the nav was parsed into too few pages: has its format changed?"


def test_the_detailed_notes_stay_in_the_repository_but_off_the_site():
    excluded = excluded_files()
    assert DETAILED_NOTES <= excluded, f"published by accident: {sorted(DETAILED_NOTES - excluded)}"
    for name in excluded:
        assert (ROOT / "docs" / name).is_file(), \
            f"exclude_docs names {name}, which does not exist: a typo here would publish the real file"


def test_the_nav_keeps_the_changelog_as_a_link_not_a_page():
    """CHANGELOG.md links to files outside the site, so a page that included it would not build."""
    external = [target for _, target in nav_entries(config()["nav"]) if is_url(target)]
    assert any(t.endswith("/CHANGELOG.md") for t in external)


# ============================================================ portability
def test_the_configuration_asks_for_nothing_the_successor_of_material_cannot_read():
    """Zensical reads mkdocs.yml but supports no hooks and no plugins beyond search, and it has no `gh-deploy`."""
    cfg = config()
    assert "hooks" not in cfg, "hooks are not supported by Zensical"
    plugins = [p if isinstance(p, str) else next(iter(p)) for p in cfg.get("plugins", [])]
    assert set(plugins) <= {"search"}, f"plugins beyond search are not portable: {plugins}"
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        assert "gh-deploy" not in workflow.read_text(encoding="utf-8"), \
            f"{workflow.name} publishes with gh-deploy: use the Pages artifact flow"


def test_the_docs_dependencies_are_light_and_stay_below_mkdocs_2():
    group = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["dependency-groups"]["docs"]
    names = {re.split(r"[<>=!~ ]", requirement)[0] for requirement in group}
    assert {"mkdocs", "mkdocs-material"} <= names
    assert "torch" not in names, "the site is hand-written pages: it must not need PyTorch"
    mkdocs = next(r for r in group if re.split(r"[<>=!~ ]", r)[0] == "mkdocs")
    assert "<2" in mkdocs.replace(" ", ""), "MkDocs 2.0 will not run Material sites"


# ============================================================ the workflow
def workflow():
    data = yaml.safe_load(WORKFLOW_FILE.read_text(encoding="utf-8"))
    data["triggers"] = data.get(True, data.get("on"))             # YAML reads a bare `on` as True
    return data


@pytest.mark.skipif(not WORKFLOW_FILE.exists(), reason=".github/ is not part of an sdist")
def test_the_workflow_builds_strictly_on_every_pull_request():
    wf = workflow()
    assert set(wf["triggers"]) == {"push", "pull_request", "workflow_dispatch"}
    steps = " ".join(step.get("run", "") for step in wf["jobs"]["build"]["steps"])
    assert "mkdocs build --strict" in steps, "a warning must fail the build, or a broken link ships"
    assert "tests/test_docs_site.py" in steps and "--noconftest" in steps, \
        "the site's own checks (what it publishes, the nav, the exclusions) must run in CI beside the build"
    assert str(wf["env"]["NO_MKDOCS_2_WARNING"]) == "1"


@pytest.mark.skipif(not WORKFLOW_FILE.exists(), reason=".github/ is not part of an sdist")
def test_the_workflow_cannot_publish_by_accident():
    """Publishing needs a public repository, and the project's rule is that nothing is published without a go-ahead."""
    wf = workflow()
    assert wf["permissions"] == {"contents": "read"}, "the workflow as a whole may only read"
    assert "pages" not in wf["jobs"]["build"].get("permissions", {}), "the build runs on pull requests: it may not publish"
    assert wf["triggers"]["workflow_dispatch"]["inputs"]["deploy"]["default"] is False
    deploy = wf["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    assert deploy["permissions"]["pages"] == "write" and deploy["permissions"]["id-token"] == "write"
    assert deploy["environment"]["name"] == "github-pages"
    assert any("actions/deploy-pages" in step.get("uses", "") for step in deploy["steps"])
    condition = " ".join(str(deploy["if"]).split())
    assert "github.ref == 'refs/heads/main'" in condition, "publish from main only"
    assert "inputs.deploy" in condition and "vars.DOCS_DEPLOY == 'true'" in condition, \
        "publishing must be switched on deliberately, by a ticked box or a repository variable"
    assert "pull_request" not in condition


# ============================================================ the results figure on the Home page
# The Home page shows the README's results figure with the README's caption. The figure is drawn from a data file
# (docs/assets/results_logp.json, which is not part of the site), and the README's numbers are checked against it by
# tests/test_results_figure.py. These tests do the same for the Home page, and keep the two sentences that must travel
# with the figure: each approach was trained once, and a gap smaller than the recorded spread is not an ordering.
RESULTS_DATA = ROOT / "docs" / "assets" / "results_logp.json"
ARMS = ("base", "conditional", "film", "xattn", "xattn_fk")
BLOB = "https://github.com/aimat-lab/defog-adapt/blob/main/"
needs_results = pytest.mark.skipif(not RESULTS_DATA.exists() or not (ROOT / "README.md").exists(),
                                   reason="the results data file and the README are not part of an installed wheel")


def squash(text):
    return " ".join(text.split())


def readme_figure():
    """The README's alt text and the caption under its results figure, with the caption's links made absolute."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    alt = re.search(r'results_logp\.png" alt="([^"]*)"', readme).group(1)
    section = readme[readme.index("Results at a glance"):]
    section = section[:section.index("\n## ")]
    caption = section.split("</p>", 1)[1].replace("](docs/", f"]({BLOB}docs/")
    return squash(alt), squash(caption)


def home_figure():
    """The alt text of the figure on the Home page, and the text under it up to the next heading."""
    home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    image = re.search(r"!\[([^\]]*)\]\(assets/results_logp\.png\)", home)
    assert image, "the Home page does not show the results figure (assets/results_logp.png)"
    return squash(image.group(1)), squash(re.split(r"\n## ", home[image.end():], maxsplit=1)[0])


def results_arms():
    return json.loads(RESULTS_DATA.read_text(encoding="utf-8"))["arms"]


def recorded_spread():
    """The difference between two trainings of one adapter recipe, as docs/caveats.md records it."""
    caveats = (ROOT / "docs" / "caveats.md").read_text(encoding="utf-8")
    return re.search(r"Two seeds of the same adapter recipe differ by (\d\.\d{3}) in logP error", caveats).group(1)


@needs_results
def test_the_home_page_shows_the_figure_with_the_readme_alt_text_and_caption():
    readme_alt, readme_caption = readme_figure()
    home_alt, home_text = home_figure()
    assert home_alt == readme_alt, "the alt text of the figure on the Home page is not the README's"
    assert readme_caption in home_text, "the text under the figure on the Home page is not the README's caption"


@needs_results
def test_the_numbers_the_home_page_quotes_are_the_data_files_and_its_two_hedging_sentences_are_there():
    arms, (alt, caption) = results_arms(), home_figure()
    assert sorted(re.findall(r"\d\.\d{3}", alt)) == sorted(f"{arms[arm]['mae_mean']:.3f}" for arm in ARMS)
    assert f"uniqueness {arms['xattn_fk']['uniqueness_mean']:.2f}" in caption
    gap, spread = f"{arms['xattn_fk']['mae_mean'] - arms['conditional']['mae_mean']:.3f}", recorded_spread()
    assert float(gap) < float(spread), "the caption's 'smaller than' would be wrong: the sentence needs rewriting"
    # the two sentences that must travel with the figure: without them the whiskers read as more certainty than there is
    assert "Each approach was trained once" in caption
    assert (f"the {gap} between the end-to-end model and the steered adapter is smaller than the {spread} recorded "
            f"between two trainings of one adapter recipe, so this comparison does not order those two") in caption


@needs_results
def test_no_site_page_calls_the_end_to_end_model_the_most_accurate_while_its_lead_is_inside_the_spread():
    """The wording the first version of the results text carried; the notes' own caveats say it is not established."""
    arms = results_arms()
    gap = arms["xattn_fk"]["mae_mean"] - arms["conditional"]["mae_mean"]
    if gap >= float(recorded_spread()):
        pytest.skip("the lead is larger than the recorded spread, so the plain wording would be supported")
    for page in sorted(nav_pages()):
        text = squash((ROOT / "docs" / page).read_text(encoding="utf-8"))
        for claim in ("most accurate", "more accurate than any"):
            assert claim not in text, f"{page} says {claim!r}, but the lead over Feynman-Kac ({gap:.3f}) is inside the spread"


# ============================================================ the build
def mkdocs_installed():
    return importlib.util.find_spec("mkdocs") is not None and importlib.util.find_spec("material") is not None


@pytest.mark.skipif(not mkdocs_installed(), reason="the docs dependency group is not installed (pip install --group docs)")
def test_the_site_builds_strictly_and_publishes_only_what_it_should(tmp_path):
    site = tmp_path / "site"
    run = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site)],
                         cwd=ROOT, env={**os.environ, "NO_MKDOCS_2_WARNING": "1"}, capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, (run.stdout + run.stderr)[-3000:]

    for page in nav_pages():
        assert built_path(site, page).is_file(), f"{page} is in the nav but was not built"
    for relative in excluded_files():
        assert not built_path(site, relative).exists(), f"{relative} is excluded but was published"
    for stylesheet in config().get("extra_css", []):
        assert (site / stylesheet).is_file(), f"{stylesheet} is listed in extra_css but was not published"
    assert not list(site.rglob("*.py")), "a script was published"
    assert not list(site.rglob("trajectory.json")), "the banner's data was published"

    # Every link and image on a page leads somewhere in the built site. MkDocs only logs a link to an excluded file at
    # info level, so `--strict` passes and the link is dead on the site: a page that points at one of the detailed
    # notes, or at the figure that is left out, is caught here.
    for page in sorted(nav_pages()):
        for target in LINK.findall((ROOT / "docs" / page).read_text(encoding="utf-8")):
            if is_url(target) or target.startswith(("#", "mailto:")):
                continue
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(page), target.partition("#")[0]))
            assert built_path(site, relative).exists(), \
                f"{page} links to {target}, which is not in the built site (is it excluded?)"

    # Nothing goes out that no page uses: a figure or a data file that lands in docs/ would otherwise be published
    # with the site, and the build itself says nothing about it.
    used = "\n".join((ROOT / "docs" / page).read_text(encoding="utf-8") for page in nav_pages())
    used += "\n" + "\n".join(config().get("extra_css", []) + config().get("extra_javascript", []))
    for asset in sorted((ROOT / "docs").rglob("*")):
        relative = asset.relative_to(ROOT / "docs").as_posix()
        if asset.is_file() and asset.suffix != ".md" and (site / relative).exists():
            assert relative in used, f"docs/{relative} is published but no page uses it: use it, or add it to exclude_docs"

    pages = {p: p.read_text(encoding="utf-8") for p in site.rglob("*.html")}
    assert not [p.relative_to(site) for p, html in pages.items() if "--8<--" in html], \
        "a snippet marker reached a page: an include did not resolve"
    # the code that the pages include from the README and the notes arrived (identifiers survive the highlighting)
    expected = {"getting-started": "seed_everything", "guides/train-an-adapter": "AdapterModule",
                "guides/steer-toward-a-target": "FeynmanKacSampler", "about/citation-and-licence": "teufel2026defogadapt"}
    for page, identifier in expected.items():
        assert identifier in pages[site / page / "index.html"], f"{page} is missing its included code ({identifier})"
