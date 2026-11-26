"""The README as the PyPI page shows it.

PyPI draws the long description of a package on a page of its own, where a path in the repository (a link to
docs/concepts/steering.md or to LICENSE, an image under docs/assets/) leads nowhere. ``pyproject.toml`` therefore builds the
description from README.md with hatch-fancy-pypi-readme, which turns each relative path into an absolute URL on GitHub; the
README in the repository stays as it is, because GitHub and a clone want its relative paths. These tests build the wheel, as
``python -m build`` does, and read the description back out of the metadata, so that they check what a reader of the PyPI page
gets and not what the rules say.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RAW = "https://raw.githubusercontent.com/aimat-lab/defog-adapt/main/"       # the file itself: what an image needs
BLOB = "https://github.com/aimat-lab/defog-adapt/blob/main/"               # the page GitHub draws for it: what a link needs

pytestmark = pytest.mark.skipif(not (ROOT / "pyproject.toml").is_file() or not README.is_file(),
                                reason="needs the source tree: pyproject.toml and README.md")

FENCED = re.compile(r"^(`{3,}|~{3,}).*?^\1[ \t]*$", flags=re.S | re.M)
INLINE = re.compile(r"`[^`\n]+`")


def without_code(text: str) -> str:
    """The prose: a path in a code sample is not a link, and no rule may touch it."""
    return INLINE.sub("``", FENCED.sub("", text))


def code_in(text: str) -> list:
    return [m.group(0) for m in FENCED.finditer(text)] + INLINE.findall(FENCED.sub("", text))


def image_targets(text: str) -> list:
    prose = without_code(text)
    return re.findall(r'!\[[^\]]*\]\(([^)\s]+)', prose) + re.findall(r'\b(?:src|srcset)="([^"]*)"', prose)


def link_targets(text: str) -> list:
    prose = without_code(text)
    return (re.findall(r'(?<!!)\[[^\]]*\]\(([^)\s]+)', prose)             # [text](target)
            + re.findall(r'^\[[^\]\n]+\]:[ \t]*(\S+)', prose, flags=re.M)     # [label]: target
            + re.findall(r'\bhref="([^"]*)"', prose))


def relative(target: str) -> bool:
    return not target.startswith(("https://", "mailto:", "#"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The headers and the long description in the metadata of the wheel that the build produces."""
    pytest.importorskip("hatchling")
    pytest.importorskip("hatch_fancy_pypi_readme")          # the dev group has both; the minimum-versions job has neither
    folder = tmp_path_factory.mktemp("wheel")
    done = subprocess.run([sys.executable, "-m", "hatchling", "build", "--target", "wheel", "--directory", str(folder)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    (wheel,) = folder.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))).decode("utf-8")
    headers, _, description = metadata.partition("\n\n")
    return headers, description


def test_no_target_in_the_description_is_a_path_in_the_repository(built):
    _, description = built
    readme = README.read_text(encoding="utf-8")
    assert any(relative(t) for t in link_targets(readme) + image_targets(readme)), \
        "the README has no relative path left, so this test would have nothing to catch: is it still the file the rules are for?"
    unresolved = [t for t in link_targets(description) + image_targets(description) if relative(t)]
    assert not unresolved, f"PyPI cannot resolve these paths: add a rule for them to pyproject.toml, or write them absolute: {unresolved}"


def test_each_image_of_the_repository_is_the_file_itself_and_exists(built):
    _, description = built
    own = [t for t in image_targets(description) if t.startswith((RAW, BLOB))]
    assert own, "the README shows images from docs/assets: none of them points into the repository"
    for url in own:
        assert url.startswith(RAW), f"an image needs the raw file, a blob URL is an html page: {url}"
        assert (ROOT / url[len(RAW):]).is_file(), f"{url} points at a file the repository does not have"


def test_each_link_into_the_repository_is_a_page_of_it_that_exists(built):
    _, description = built
    own = [t for t in link_targets(description) if t.startswith((RAW, BLOB))]
    assert own, "the README links to docs/, examples/ and LICENSE: none of them points into the repository"
    for url in own:
        assert url.startswith(BLOB), f"a link needs the page GitHub draws, the raw file would download as text: {url}"
        assert (ROOT / url[len(BLOB):].split("#")[0]).exists(), f"{url} points at a file the repository does not have"


def test_only_the_targets_differ_from_the_readme(built):
    _, description = built
    readme = README.read_text(encoding="utf-8")

    def bare(text: str) -> list:
        """Without the two prefixes the rules add, and nothing else: what is left must be the README, line for line."""
        return text.replace(RAW, "").replace(BLOB, "").splitlines()

    mine, original = bare(description), bare(readme)
    differing = [(number, was, now) for number, (was, now) in enumerate(zip(original, mine), 1) if was != now]
    if differing:
        number, was, now = differing[0]
        pytest.fail(f"the build changed more than the targets: README line {number} is {was!r}, the description has {now!r}")
    assert len(mine) == len(original), "the build added or removed lines"
    assert code_in(description) == code_in(readme), "a code sample was rewritten: a path in code is not a link"


def test_the_description_is_declared_as_markdown(built):
    headers, _ = built
    assert "Description-Content-Type: text/markdown" in headers.splitlines()
