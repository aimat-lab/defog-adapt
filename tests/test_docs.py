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
                    *(ROOT / "docs").rglob("*.md"), *(ROOT / "examples").glob("*.md")])
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


# ============================================================ the documentation site
# The pages under docs/ that the site publishes (see mkdocs.yml) link to the detailed notes by absolute GitHub URL,
# include code from the README and the notes, and show commands. Each of those can rot without a relative link
# breaking, so each is checked here. tests/test_docs_site.py checks the site's own configuration and build.
REPO = "https://github.com/aimat-lab/defog-adapt"
GITHUB_LINK = re.compile(re.escape(REPO) + r"/(?:blob|tree)/main/([^)\s#]+)(#[^)\s]*)?")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_github_link_to_this_repository_points_at_something_that_exists(path):
    for target, anchor in GITHUB_LINK.findall(path.read_text(encoding="utf-8")):
        dest = ROOT / target
        assert dest.exists(), f"{path.name}: the GitHub link to {target} points at nothing in this repository"
        if anchor and dest.suffix == ".md":
            assert anchor[1:] in anchors(dest), \
                f"{path.name}: the GitHub link to {target}{anchor} has no heading '{anchor[1:]}' in {dest.name}"


SNIPPET = re.compile(r'^--8<-- "([^":]+)(?::([\w-]+))?"[ \t]*$', flags=re.M)


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_snippet_include_names_a_file_and_a_section_that_exist(path):
    """The site includes code from the README and the notes (``--8<-- "README.md:quickstart"``), so there is one copy."""
    for name, section in SNIPPET.findall(path.read_text(encoding="utf-8")):
        source = ROOT / name
        assert source.is_file(), f"{path.name}: the snippet include names {name}, which does not exist"
        if section:
            text = source.read_text(encoding="utf-8")
            for edge in ("start", "end"):
                assert text.count(f"--8<-- [{edge}:{section}]") == 1, \
                    f"{path.name}: {name} must have exactly one '--8<-- [{edge}:{section}]' marker"
            assert text.index(f"[start:{section}]") < text.index(f"[end:{section}]"), \
                f"{path.name}: in {name} the section '{section}' ends before it starts"


#: options every PyComex recipe has, which are not written in the recipe itself
PYCOMEX_OPTIONS = {"__TESTING__"}


def example_commands(text):
    """Each ``python examples/x.py ...`` command in a bash block, with its backslash continuations joined."""
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            match = re.match(r"\s*python\s+(examples/\S+\.py)\s*(.*)", line)
            if match:
                yield match.group(1), " ".join(match.group(2).split(" #")[0].split())


def declares(script_text, flag):
    """Whether a recipe declares ``--flag``: as an argparse option string, or as a PyComex parameter (``FLAG: str = ..``).

    Looking for the bare word would pass any one-letter flag, because that letter is in every script.
    """
    return (f'"--{flag}"' in script_text or f"'--{flag}'" in script_text
            or re.search(rf"^{re.escape(flag)}\s*[:=]", script_text, flags=re.M) is not None)


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_example_command_names_a_real_script_and_flags_it_knows(path):
    for script, arguments in example_commands(path.read_text(encoding="utf-8")):
        source = ROOT / script
        assert source.is_file(), f"{path.name}: the command runs {script}, which does not exist"
        text = source.read_text(encoding="utf-8")
        for flag in re.findall(r"(?<![\w-])--([\w-]+)", arguments):
            assert flag in PYCOMEX_OPTIONS or declares(text, flag), \
                f"{path.name}: `python {script}` is given --{flag}, which {source.name} does not declare"


def test_a_flag_is_found_only_where_a_recipe_declares_it():
    argparse_script = 'ap.add_argument("--size", default=None)\nap.add_argument("--n", type=int)\n'
    pycomex_script = 'BASE_CKPT: str = "x"\nPROPERTY = "logp"\nunrelated = 1  # n size\n'
    assert declares(argparse_script, "size") and declares(argparse_script, "n")
    assert not declares(argparse_script, "nn") and not declares(argparse_script, "unrelated")
    assert declares(pycomex_script, "BASE_CKPT") and declares(pycomex_script, "PROPERTY")
    assert not declares(pycomex_script, "unrelated_flag") and not declares(pycomex_script, "n")


def test_the_command_checker_sees_the_commands_it_is_meant_to_check():
    """A parser that finds nothing would pass every file, so pin what it finds in a made-up block."""
    block = '```bash\npython examples/a.py --one 1 \\\n    --two "x"   # a comment --three\npip install z\n```'
    assert list(example_commands(block)) == [("examples/a.py", '--one 1 --two "x"')]


RECIPES = ROOT / "docs" / "reference" / "recipes.md"


@pytest.mark.skipif(not RECIPES.exists(), reason="the site pages are not part of an installed wheel")
def test_the_recipes_page_lists_every_recipe_and_only_real_ones():
    """A recipe without a row is one nobody finds; a row without a recipe sends the reader to a missing page."""
    text = RECIPES.read_text(encoding="utf-8")
    listed = set(re.findall(r"\[`(\w+\.py)`\]\(" + re.escape(REPO) + r"/blob/main/examples/\1\)", text))
    scripts = {p.name for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_")}
    assert listed == scripts, (f"missing from docs/reference/recipes.md: {sorted(scripts - listed)}; "
                               f"listed there but not in examples/: {sorted(listed - scripts)}")


MAIN_CLASSES = ROOT / "docs" / "reference" / "main-classes.md"


def main_classes_rows():
    """(module, name) for each row of the tables in the main-classes page; the heading of a table names its module."""
    module = None
    for line in MAIN_CLASSES.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):                    # every heading starts afresh: one without a module names none
            heading = re.match(r"^## .*\(`(defog_adapt[\w.]*)`\)", line)
            module = heading.group(1) if heading else None
        row = re.match(r"^\| `(\w+)` \|", line)
        if row:
            yield module, row.group(1)


@pytest.mark.skipif(not MAIN_CLASSES.exists(), reason="the site pages are not part of an installed wheel")
def test_every_name_in_the_main_classes_page_exists_where_its_heading_says():
    pytest.importorskip("rdkit")                      # defog_adapt.domains needs it
    rows = list(main_classes_rows())
    assert len(rows) >= 40, f"the page was parsed into only {len(rows)} rows: has its table format changed?"
    for module, name in rows:
        assert module, f"{name} is in a table under a heading that names no module"
        assert hasattr(importlib.import_module(module), name), f"main-classes.md lists {module}.{name}, which is gone"


@pytest.mark.skipif(not MAIN_CLASSES.exists(), reason="the site pages are not part of an installed wheel")
def test_the_main_classes_page_lists_every_preset():
    """The presets are the recorded recipes, so a new one without a row is a recipe nobody can find."""
    from defog_adapt import presets
    listed = {name for module, name in main_classes_rows() if module == "defog_adapt.presets"}
    assert listed == set(presets.__all__)


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
