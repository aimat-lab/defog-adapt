"""The structure of the tutorial pages: the steer guide, the base guide and the adapter guide.

Each page walks through one script in pieces: a piece is a section of that file, included by marker, and the whole file is
the last thing on the page. So page and code can drift apart in a few ways, and each has a test here, run on every guide.
A piece that is missing, shown twice or out of order. Code in the script that no step shows. Code pasted into the page,
which would be a second copy. A final script that is not the whole file, or not last. A step that is numbered wrongly, or an
optional step that stops saying it is optional. A header that names other files than the script uses.
"""
import re

import pytest

from _snippets import INCLUDE, MARKER, sections                     # tests/ has no __init__.py; pytest prepends this dir
from _tutorials import TUTORIALS

pytestmark = pytest.mark.skipif(not all(t.script_path.is_file() for t in TUTORIALS),
                                reason="docs/ is not part of an installed wheel")
EACH = pytest.mark.parametrize("tutorial", TUTORIALS, ids=lambda t: t.key)


@EACH
def test_the_page_only_includes_code_it_never_pastes_it(tutorial):
    """One copy of the code: every python block on the page is a single include line, nothing typed."""
    blocks = re.findall(r"^```python\n(.*?)^```", tutorial.page_text(), flags=re.S | re.M)
    assert len(blocks) >= tutorial.steps, f"{tutorial.page} shows {len(blocks)} python blocks: has its fence syntax changed?"
    for body in blocks:
        assert INCLUDE.fullmatch(body.strip()), f"{tutorial.page}: a python block holds typed code, not an include:\n{body}"


@EACH
def test_every_section_of_the_script_is_on_the_page_once_and_in_file_order(tutorial):
    on_page = [section for name, section in INCLUDE.findall(tutorial.page_text())
               if name == tutorial.include_name and section]
    in_script = sections(tutorial.script_text())
    assert len(in_script) >= tutorial.steps, f"{tutorial.script} has {len(in_script)} sections: has its marker syntax changed?"
    assert on_page == in_script, f"{tutorial.page} includes {on_page}, the script has {in_script} (in this order, each once)"


@EACH
def test_the_whole_script_is_the_last_thing_on_the_page_and_is_there_once(tutorial):
    page = tutorial.page_text()
    whole = [m for m in INCLUDE.finditer(page) if m.group(1) == tutorial.include_name and not m.group(2)]
    assert len(whole) == 1, f"{tutorial.page} includes the whole script {len(whole)} times, it should be once"
    assert page[:whole[0].start()].rstrip().endswith("```python"), "the whole script is not in a python block"
    assert page[whole[0].end():].strip() == "```", f"something follows the final script on {tutorial.page}"
    headings = re.findall(r"^#+ (.*)$", page, flags=re.M)
    assert headings[-1] == "The final script", f"the last heading is {headings[-1]!r}, not 'The final script'"


@EACH
def test_no_code_in_the_script_goes_unshown(tutorial):
    """Every line of code lies in a section, so the steps show everything that the final script runs."""
    inside = False
    for number, line in enumerate(tutorial.script_text().splitlines(), 1):
        marker = MARKER.match(line)
        if marker:
            inside = marker.group(1) == "start"
        elif not inside and line.strip() and not line.lstrip().startswith("#"):
            pytest.fail(f"{tutorial.script}:{number}: `{line}` is outside every section, so no step shows it")


@EACH
def test_the_steps_are_numbered_and_the_optional_ones_say_so(tutorial):
    steps = re.findall(r"^## (\d+)\. (.*)$", tutorial.page_text(), flags=re.M)
    assert [int(number) for number, _ in steps] == list(range(1, tutorial.steps + 1)), f"the steps are {steps}"
    for number, title in steps:
        said_optional = title.endswith("(optional)")
        assert said_optional == (int(number) > tutorial.required), \
            f"{tutorial.page} step {number} '{title}': steps 1 to {tutorial.required} are required, a later one says '(optional)'"


@EACH
def test_the_header_names_the_files_the_script_uses_and_no_others(tutorial):
    text = tutorial.script_text()
    header = text.split("# --8<--")[0]
    named = set(re.findall(r"[\w]+\.(?:ckpt|smi)", header))
    assert named == set(tutorial.files), f"the header of {tutorial.script} names {sorted(named)}, expected {sorted(tutorial.files)}"
    for name in tutorial.files:
        stem = name.rsplit(".", 1)[0]
        assert stem in text.replace(header, ""), f"{tutorial.script} names {name} in its header but never uses {stem}"
