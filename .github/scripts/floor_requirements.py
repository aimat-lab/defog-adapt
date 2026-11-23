#!/usr/bin/env python
"""Print the LOWEST supported version of every dependency as a pinned requirements file.

The floors job installs exactly these. They are generated from ``pyproject.toml`` rather than written out
in the workflow so that the two cannot drift: the floors in pyproject are only worth anything if the suite
is actually run against them, and a second copy of the list would be a place for them to disagree.

    python .github/scripts/floor_requirements.py              # core + the chem extra
    python .github/scripts/floor_requirements.py core chem    # chosen groups

Lower bounds (``>=X``) become ``==X``. Anything without a lower bound is printed unpinned.
"""
import re
import sys
from pathlib import Path

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10: the backport (`pip install tomli`)
    import tomli as tomllib

DEFAULT_GROUPS = ("core", "chem")


def floors(pyproject: dict, groups=DEFAULT_GROUPS):
    project = pyproject["project"]
    requirements = []
    for group in groups:
        requirements += project["dependencies"] if group == "core" else \
            project["optional-dependencies"][group]
    pins = []
    for requirement in requirements:
        if requirement.startswith("defog-adapt"):        # a self-reference to other extras
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([A-Za-z0-9_.!+\-]+)", requirement)
        pins.append(f"{match.group(1)}=={match.group(2)}" if match else re.split(r"[<>=!~ ]", requirement)[0])
    return pins


def main(argv):
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    print("\n".join(floors(pyproject, tuple(argv) or DEFAULT_GROUPS)))


if __name__ == "__main__":
    main(sys.argv[1:])
