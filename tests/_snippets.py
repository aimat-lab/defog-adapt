"""What the site's ``--8<--`` includes put on a page, for the documentation tests.

A page includes code by name (``--8<-- "docs/guides/steer_toward_a_target.py:domain"``): one section of a file, between its
``--8<-- [start:domain]`` and ``--8<-- [end:domain]`` marker lines, or the whole file. The site never shows the marker lines
themselves. These helpers do the same, so that a test can read the code a page really shows.
"""
import re

#: an include line on a page: the file, and a section of it if one is named
INCLUDE = re.compile(r'^--8<-- "([^":]+)(?::([\w-]+))?"[ \t]*$', flags=re.M)
#: a marker line inside the included file (it may sit in a comment): ``# --8<-- [start:name]``
MARKER = re.compile(r"^.*--8<--\s*\[(start|end):([\w-]+)\]\s*$")


def included(source: str, section: str = None) -> str:
    """The text an include puts on the page: one section of ``source``, or all of it. Marker lines never appear."""
    lines = source.splitlines()
    if section is None:
        return "\n".join(line for line in lines if not MARKER.match(line)) + "\n"
    out, inside = [], False
    for line in lines:
        marker = MARKER.match(line)
        if marker:
            if marker.group(2) == section:
                inside = marker.group(1) == "start"
            continue
        if inside:
            out.append(line)
    return "\n".join(out) + "\n"


def sections(source: str) -> list:
    """The names of the sections of ``source``, in the order of their start markers."""
    return [m.group(2) for m in map(MARKER.match, source.splitlines()) if m and m.group(1) == "start"]


def expand_includes(code: str, root) -> str:
    """``code`` with every include line replaced by the text it includes (files are read from ``root``)."""
    def replace(match):
        return included((root / match.group(1)).read_text(encoding="utf-8"), match.group(2)).rstrip("\n")
    return INCLUDE.sub(replace, code)
