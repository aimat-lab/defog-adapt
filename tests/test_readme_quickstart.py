"""The README's quickstart must keep running.

It is the first thing a new user copies, and a snippet nobody executes rots. ``tests/test_docs.py`` only checks that
it parses and that its imports exist; this runs it, exactly as printed in the README, in a fresh interpreter, at toy
scale (the quickstart's ``EPOCHS`` and ``N`` constants are shrunk so that it takes seconds instead of half a
minute).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")                                # the quickstart is molecular: the `chem` extra

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

pytestmark = pytest.mark.skipif(not README.exists(), reason="README.md is not part of an installed wheel")


def quickstart_code() -> str:
    text = README.read_text(encoding="utf-8")
    heading = re.search(r"^## .*\bQuickstart\b.*$", text, flags=re.M)          # the heading starts with an emoji
    assert heading, "the README has no '## Quickstart' heading"
    section = text[heading.end():].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, flags=re.S)
    assert match, "the README has no python block under '## Quickstart'"
    return match.group(1)


def at_toy_scale(code: str, epochs: int, n: int) -> str:
    """The quickstart with its two size constants shrunk; fails loudly if either is no longer there to shrink."""
    code, n_epochs = re.subn(r"^EPOCHS = \d+", f"EPOCHS = {epochs}", code, flags=re.M)
    code, n_samples = re.subn(r"^N = \d+", f"N = {n}", code, flags=re.M)
    assert (n_epochs, n_samples) == (1, 1), \
        "the quickstart no longer has its 'EPOCHS = ...' and 'N = ...' lines, which this test shrinks to toy scale"
    return code


def run_quickstart(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    script = tmp_path / "quickstart.py"
    script.write_text(code, encoding="utf-8")
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=900)


def test_the_readme_quickstart_runs_and_prints_one_line_per_target(tmp_path):
    run = run_quickstart(at_toy_scale(quickstart_code(), epochs=2, n=8), tmp_path)
    assert run.returncode == 0, run.stderr[-1500:]
    targets = [line.split(":")[0] for line in run.stdout.splitlines() if line.startswith("target logP")]
    assert targets == ["target logP 0.0", "target logP 2.0"], run.stdout


def test_the_quickstart_test_notices_a_broken_snippet(tmp_path):
    """A test that cannot fail is not a test: feed the same runner a snippet that uses a name the library lacks."""
    broken = quickstart_code().replace("AdaptedSampler(", "NoSuchSampler(")
    assert "NoSuchSampler(" in broken
    assert run_quickstart(at_toy_scale(broken, epochs=1, n=4), tmp_path).returncode != 0
