"""Parity gate: ``defog_adapt`` must reproduce the research repo's outputs.

Opt-in. It needs real checkpoints, which are not part of the package, so it is skipped unless
``DEFOG_ADAPT_CKPTS`` points at a directory containing the files listed in ``cases.CKPT_FILES``
(and the files must be the very ones the golden output was generated from; their hashes are
checked first).

    DEFOG_ADAPT_CKPTS=/path/to/ckpts python -m pytest tests/parity -m parity

What is compared (see ``cases.py``): the base model's logits on a seeded state; sampling from the
base; sampling through an adapter; Feynman-Kac sampling with a learned head and size model; two
RL iterations of the adapter trainer; inpainting with and without time-travel; Feynman-Kac with
a spectrum energy. Sampled molecules and counts must match exactly. Floats are compared to
1e-5 relative; the bit-exact tensor hashes are compared only when torch's version is the same as
when the golden output was generated, because kernels may legitimately differ across versions.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

import cases  # tests/parity is on sys.path (pytest's rootdir-relative import mode)

pytestmark = pytest.mark.parity

GOLDEN = Path(__file__).parent / "golden" / "golden.json"
CKPTS = os.environ.get("DEFOG_ADAPT_CKPTS")
PKG = os.environ.get("DEFOG_ADAPT_PARITY_PKG", "defog_adapt")
JOBS = int(os.environ.get("DEFOG_ADAPT_PARITY_JOBS", "7"))


def compare(golden, new, path="", strict_bits=False, problems=None):
    """Collect every difference between two JSON-like structures."""
    problems = [] if problems is None else problems
    if isinstance(golden, dict):
        if not isinstance(new, dict) or set(golden) != set(new):
            problems.append(f"{path}: keys differ "
                            f"({sorted(golden)} vs {sorted(new) if isinstance(new, dict) else new!r})")
            return problems
        for k in golden:
            if k == "sha256" and not strict_bits:
                continue
            compare(golden[k], new[k], f"{path}.{k}", strict_bits, problems)
    elif isinstance(golden, list):
        if not isinstance(new, list) or len(golden) != len(new):
            problems.append(f"{path}: length {len(golden)} vs "
                            f"{len(new) if isinstance(new, list) else new!r}")
            return problems
        for i, (g, n) in enumerate(zip(golden, new)):
            compare(g, n, f"{path}[{i}]", strict_bits, problems)
    elif isinstance(golden, float) and not isinstance(golden, bool):
        if not isinstance(new, (int, float)) or not math.isclose(
                golden, new, rel_tol=1e-5, abs_tol=1e-6):
            problems.append(f"{path}: {golden!r} vs {new!r}")
    elif golden != new:
        problems.append(f"{path}: {golden!r} vs {new!r}")
    return problems


@pytest.fixture(scope="module")
def run():
    if not CKPTS:
        pytest.skip("set DEFOG_ADAPT_CKPTS to run the parity gate")
    if not GOLDEN.exists():
        pytest.skip(f"no golden output at {GOLDEN}")
    golden = json.loads(GOLDEN.read_text())
    ctx = cases.Ctx(PKG, CKPTS)
    env = cases.environment(ctx)
    if env["ckpt_sha256"] != golden["environment"]["ckpt_sha256"]:
        pytest.fail("the checkpoint files differ from the ones the golden output was generated "
                    f"from:\n  golden: {golden['environment']['ckpt_sha256']}\n  here:   "
                    f"{env['ckpt_sha256']}", pytrace=False)
    new = cases.run_parallel(PKG, CKPTS, jobs=JOBS, log=lambda s: print(s, flush=True))
    strict = env["torch"] == golden["environment"]["torch"]
    return golden["cases"], new, strict


@pytest.mark.parametrize("name", list(cases.CASES))
def test_case_matches_golden(run, name):
    golden, new, strict = run
    problems = compare(golden[name], new[name], name, strict_bits=strict)
    shown = "\n  ".join(problems[:15])
    more = f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else ""
    assert not problems, f"{len(problems)} difference(s) from the golden output:\n  {shown}{more}"
