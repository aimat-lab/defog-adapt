"""The diagrams on the documentation pages.

Six things go wrong quietly with them. A picture without alt text is nothing to a reader who cannot see it. An SVG that keeps
live text depends on the reader having the font, so its labels would be set in whatever the browser has instead. An SVG that
loads something from outside breaks offline and tells a third party who reads the page. A figure drawn in black on a
transparent canvas vanishes on the dark theme of the site. A diagram that no page shows is dead weight, and a page that shows
a missing one is a hole. And a script whose picture was never rebuilt, or a picture without its script, can only be found by
redrawing. So: every image has alt text, every diagram on a page has a caption, every published SVG is self-contained, sized
for a page and on a white card, every diagram is shown, and every script has its SVG and is named in publish.py (the one place
that rebuilds them).
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DIAGRAMS = DOCS / "assets" / "diagrams"
SRC = DIAGRAMS / "src"
pytestmark = pytest.mark.skipif(not DIAGRAMS.is_dir(), reason="docs/ is not part of an installed wheel")

ANY_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
IMAGE_LINE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)[ \t]*$", flags=re.M)        # an image on a line of its own
PAGES = sorted(DOCS.rglob("*.md"))


def diagram_images():
    """(page, alt text, target, the first line after the image) for every diagram that a page shows."""
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for match in IMAGE_LINE.finditer(text):
            if "assets/diagrams/" in match.group(2):
                yield page, match.group(1), match.group(2), text[match.end():].lstrip("\n").split("\n", 1)[0]


def svg_root(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    root = re.search(r"<svg\b([^>]*)>", text, flags=re.S).group(1)
    return {key: value for key, value in re.findall(r'\b(width|height|viewBox)="([^"]*)"', root)}


def test_the_parser_finds_the_diagrams_it_is_meant_to_check():
    """A parser that finds nothing would pass every test below, so pin that it finds the four."""
    assert len(list(diagram_images())) >= 4, "the pages show fewer than four diagrams: has the image syntax changed?"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_image_on_a_page_has_alt_text(page):
    for alt, target in ANY_IMAGE.findall(page.read_text(encoding="utf-8")):
        assert alt.strip(), f"{page.name}: the image {target} has no alt text"


def test_every_diagram_on_a_page_is_followed_by_an_italic_caption():
    for page, _alt, target, after in diagram_images():
        assert len(after) > 20 and after.startswith("*") and after.endswith("*") and not after.startswith("**"), \
            f"{page.name}: the diagram {target} has no italic caption under it (got {after[:40]!r})"


def test_every_diagram_is_shown_and_every_shown_diagram_exists():
    shown = {Path(target).name for _page, _alt, target, _after in diagram_images()}
    published = {p.name for p in DIAGRAMS.glob("*.svg")}
    assert shown == published, f"shown but missing: {sorted(shown - published)}; published but shown nowhere: {sorted(published - shown)}"
    for page, _alt, target, _after in diagram_images():
        assert (page.parent / target.partition("#")[0]).resolve().is_file(), f"{page.name}: {target} does not exist"


@pytest.mark.parametrize("svg", sorted(DIAGRAMS.glob("*.svg")), ids=lambda p: p.name)
def test_a_published_diagram_is_self_contained_and_sized_for_a_page(svg):
    text = svg.read_text(encoding="utf-8")
    assert "<text" not in text, "live text: the picture needs the reader's fonts; rebuild it with src/publish.py"
    assert "<script" not in text and not re.search(r"\bon\w+=\"", text), "an SVG on a page must carry no script"
    assert not re.search(r'(?:href|src)="(?:https?:)?//', text) and "url(http" not in text and "@import" not in text, \
        "the diagram loads something from outside"
    root = svg_root(svg)
    canvas = float(root["viewBox"].split()[2])
    assert float(root["width"]) >= 1.9 * canvas, \
        f"the SVG is {root['width']} wide on a canvas of {canvas}: its labels would be small on a page (src/publish.py enlarges it)"


@pytest.mark.parametrize("svg", sorted(DIAGRAMS.glob("*.svg")), ids=lambda p: p.name)
def test_a_published_diagram_has_a_white_card_over_its_whole_canvas(svg):
    """The tool's SVG is transparent and the figures are drawn in black: on the dark theme of the site they would vanish."""
    text = svg.read_text(encoding="utf-8")
    element = re.search(r'<(?:path|rect)\b[^>]*\bid="background"[^>]*>', text, flags=re.S)
    assert element, "no background card: the figure is black on a transparent canvas and would vanish on a dark page"
    assert re.search(r'\bfill="#(?:ffffff|fff)"', element.group(0), flags=re.I), "the background card is not white"
    width, height = (float(v) for v in svg_root(svg)["viewBox"].split()[2:4])
    d = re.search(r'\bd="([^"]*)"', element.group(0))
    if d:                                               # path data: pairs of coordinates after the commands
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d.group(1))]
        assert (max(numbers[0::2]), max(numbers[1::2])) == (width, height), "the background card does not cover the canvas"
    else:
        assert (float(re.search(r'\bwidth="([\d.]+)"', element.group(0)).group(1)),
                float(re.search(r'\bheight="([\d.]+)"', element.group(0)).group(1))) == (width, height)


def scripts_in_publish():
    tree = ast.parse((SRC / "publish.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SCRIPTS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("publish.py has no SCRIPTS list")


def test_every_diagram_script_has_its_svg_and_publish_py_names_them_all():
    drawn = {}
    for script in sorted(SRC.glob("*.py")):
        found = re.search(r"^NAME = '([^']+)'", script.read_text(encoding="utf-8"), flags=re.M)
        if found:                                       # vocab.py and publish.py are not diagrams
            drawn[script.stem] = found.group(1)
    assert set(scripts_in_publish()) == set(drawn), \
        f"publish.py builds {sorted(scripts_in_publish())}, the scripts are {sorted(drawn)}"
    published = {p.stem for p in DIAGRAMS.glob("*.svg")}
    assert published == set(drawn.values()), \
        f"SVGs without a script: {sorted(published - set(drawn.values()))}; scripts without an SVG: {sorted(set(drawn.values()) - published)}"
