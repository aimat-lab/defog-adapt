"""The documentation stays true to the code it describes.

Prose cannot be tested, but three kinds of documentation rot can: a code block that no longer parses, an
import that names something that does not exist any more, and a link or anchor that points nowhere. Each of
those has happened to documentation of a package that was extracted from a larger one, so each is checked.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = sorted([ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CLAUDE.md", ROOT / "CONTRIBUTING.md",
                    ROOT / "SECURITY.md",
                    *(ROOT / "docs").glob("*.md"), *(ROOT / "examples").glob("*.md")])
DOC_FILES = [p for p in DOC_FILES if p.exists()]

pytestmark = pytest.mark.skipif(not (ROOT / "docs").is_dir(), reason="docs/ is not part of an installed wheel")


def python_blocks(path):
    for match in re.finditer(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S):
        yield match.group(1), path.read_text(encoding="utf-8")[: match.start()].count("\n") + 1


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped (hyphens kept), spaces to hyphens."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path):
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*)$", path.read_text(encoding="utf-8"), flags=re.M)}


def test_the_documentation_exists():
    names = {p.name for p in DOC_FILES}
    for required in ("README.md", "adapters.md", "steering.md", "rl.md", "results.md", "caveats.md", "migration.md"):
        assert required in names, f"missing documentation file {required}"


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_python_block_parses(path):
    for code, line in python_blocks(path):
        try:
            ast.parse(code)
        except SyntaxError as err:
            pytest.fail(f"{path.name}:{line}: a python block does not parse: {err}")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_defog_adapt_import_in_a_block_names_something_real(path):
    pytest.importorskip("rdkit")                      # defog_adapt.domains needs it
    for code, line in python_blocks(path):
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("defog_adapt"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), \
                        f"{path.name}:{line}: `from {node.module} import {alias.name}` does not exist"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("defog_adapt"):
                        importlib.import_module(alias.name)


LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_relative_link_and_anchor_resolves(path):
    text = path.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if re.match(r"^[a-z]+:", target):
            continue                                  # an external URL
        file_part, _, anchor = target.partition("#")
        dest = path if not file_part else (path.parent / file_part).resolve()
        assert dest.exists(), f"{path.name}: the link ({target}) points at a file that does not exist"
        if anchor and dest.suffix == ".md":
            assert anchor in anchors(dest), \
                f"{path.name}: the link ({target}) has no heading '{anchor}' in {dest.name}"


def test_the_anchor_slugger_agrees_with_github_on_the_headings_this_repo_links_to():
    for heading, expected in [
        ("6. RL on adapters", "6-rl-on-adapters"),
        ("File format, compatibility and trust", "file-format-compatibility-and-trust"),
        ("What the condition means: the label convention", "what-the-condition-means-the-label-convention"),
        ("The adapter base is a re-packaged checkpoint", "the-adapter-base-is-a-re-packaged-checkpoint"),
        ("Features with thin evidence", "features-with-thin-evidence"),
        # An emoji is dropped but the space after it stays, so the anchor starts with a hyphen. Checked against
        # GitHub's own rendering of a README with emoji headings: "## 🚀 Overview" is served as "#-overview".
        ("📊 Results at a glance", "-results-at-a-glance"),
    ]:
        assert slug(heading) == expected
