"""The release metadata agrees with itself.

A version appears in four places that nothing else links: ``pyproject.toml`` (the source, read back by
``defog_adapt.__version__``), ``CITATION.cff``, the top entry of ``CHANGELOG.md``, and, for a released version, its date. A
mismatch shows only after the upload, when it can no longer be changed, because a file name on PyPI is used up for good. So these
tests read them against each other, and the project's documentation link against the site the docs are built for.
"""
import re
from pathlib import Path

import pytest

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10: pytest itself depends on the backport
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ["pyproject.toml", "CITATION.cff", "CHANGELOG.md", "mkdocs.yml", "README.md"]

pytestmark = pytest.mark.skipif(not all((ROOT / name).is_file() for name in NEEDED), reason="needs the source tree")


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def citation(key: str) -> str:
    match = re.search(rf'^{re.escape(key)}:\s*"([^"]+)"', (ROOT / "CITATION.cff").read_text(encoding="utf-8"), flags=re.M)
    assert match, f"CITATION.cff has no `{key}`"
    return match.group(1)


def changelog_top() -> tuple:
    """(version, what stands in the brackets) of the first entry, which reads `## 0.1.0 (2026-11-26)` or `## 0.2.0 (unreleased)`."""
    match = re.search(r"^## (\S+) \(([^)]+)\)\s*$", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), flags=re.M)
    assert match, "the changelog has no entry of the form `## <version> (<date or unreleased>)`"
    return match.group(1), match.group(2)


def test_the_version_is_the_same_in_pyproject_the_citation_and_the_top_changelog_entry():
    version = pyproject()["project"]["version"]
    assert citation("version") == version, "CITATION.cff has another version than pyproject.toml"
    assert changelog_top()[0] == version, "the top entry of CHANGELOG.md is for another version than pyproject.toml"


def test_a_released_version_has_the_same_date_in_the_changelog_and_the_citation():
    version = pyproject()["project"]["version"]
    _, when = changelog_top()
    if re.search(r"\.dev\d*$", version):
        assert when == "unreleased", "a development version is not released, so its entry says so"
    else:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", when), f"the entry of the released {version} says `{when}`, not its date"
        assert citation("date-released") == when, "CITATION.cff says another day of release than the changelog"


def test_the_documentation_link_is_the_site_the_docs_are_built_for():
    site = re.search(r"^site_url:\s*(\S+)", (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), flags=re.M)
    assert site, "mkdocs.yml has no site_url"
    urls = pyproject()["project"]["urls"]
    assert urls["Documentation"] == site.group(1), "the package points at another site than the one mkdocs builds"
    assert site.group(1) in (ROOT / "README.md").read_text(encoding="utf-8"), "the README does not link to the documentation site"


def test_every_project_link_is_https():
    urls = pyproject()["project"]["urls"]
    assert {"Homepage", "Documentation", "Repository", "Changelog", "Issues"} <= set(urls)
    assert all(url.startswith("https://") for url in urls.values()), urls
