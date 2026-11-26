"""How the tutorials' code looks in the code window of the page.

The code of a tutorial is read in a code window, so it is written for one. No line is wider than the window (88
characters, which leaves room to spare at the width the site has, so that no scrollbar appears), and a call with four or
more arguments puts each argument on a line of its own, indented, with its closing bracket on a line of its own. Both rules
are checked on every script in docs/guides, so a new tutorial gets them without anyone remembering to ask.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "docs" / "guides").glob("*.py"))
MAX_WIDTH = 88
ONE_PER_LINE_FROM = 4          # a call with this many arguments or more is written one per line
pytestmark = pytest.mark.skipif(not SCRIPTS, reason="docs/ is not part of an installed wheel")


def crowded_calls(source: str) -> list:
    """(line, number of arguments) of every call with four or more arguments that is not written one per line: each
    argument starts its own line, the first one below the opening bracket, and the closing bracket has a line to itself."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        arguments = [(a.lineno, a.end_lineno) for a in node.args]
        arguments += [(k.lineno, k.value.end_lineno) for k in node.keywords]
        if len(arguments) < ONE_PER_LINE_FROM:
            continue
        starts = [start for start, _ in arguments]
        one_each = len(set(starts)) == len(starts)
        opened_alone = min(starts) > node.func.end_lineno
        closed_alone = node.end_lineno > max(end for _, end in arguments)
        if not (one_each and opened_alone and closed_alone):
            found.append((node.lineno, len(arguments)))
    return found


def test_the_checker_tells_a_wrapped_call_from_a_crowded_one():
    """The checker first, on calls whose answers are known: a test that cannot flag a bad call proves nothing."""
    assert crowded_calls("f(\n    1,\n    2,\n    3,\n    4,\n)\n") == []
    assert crowded_calls("f(1, 2, 3)\n") == [], "three arguments may stay on one line"
    assert crowded_calls("f(1, 2, 3, 4)\n") == [(1, 4)], "four on one line"
    assert crowded_calls("f(\n    1, 2,\n    3, 4,\n)\n") == [(1, 4)], "two to a line"
    assert crowded_calls("f(1,\n    2,\n    3,\n    4,\n)\n") == [(1, 4)], "the first one beside the bracket"
    assert crowded_calls("f(\n    1,\n    2,\n    3,\n    4)\n") == [(1, 4)], "the closing bracket beside the last one"
    assert crowded_calls("f(\n    1,\n    2,\n    3,\n    g(\n        1,\n        2,\n    ))\n") == [(1, 4)]


def test_there_are_tutorial_scripts_to_check():
    assert SCRIPTS, "docs/guides has no tutorial script"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_line_is_wider_than_the_code_window(script):
    for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
        assert len(line) <= MAX_WIDTH, f"{script.name}:{number} is {len(line)} characters wide, the limit is {MAX_WIDTH}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_call_with_four_or_more_arguments_has_one_per_line(script):
    crowded = crowded_calls(script.read_text(encoding="utf-8"))
    assert not crowded, (f"{script.name}: calls with {ONE_PER_LINE_FROM} or more arguments that are not one per line "
                         f"(line, arguments): {crowded}")
