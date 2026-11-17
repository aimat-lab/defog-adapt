#!/usr/bin/env python
"""Create ``golden/golden.json`` by running the parity cases against a package.

This is run ONCE, against the research repo's ``defog`` package at the commit the port was
taken from, in the environment that package was validated in. The result is committed; the
parity tests then require ``defog_adapt`` to reproduce it.

    PYTHONDONTWRITEBYTECODE=1 /path/to/research/.venv/bin/python \
        tests/parity/generate_golden.py --pkg defog --ckpts /path/to/research/ckpts

It can also be pointed at ``defog_adapt`` (``--pkg defog_adapt``) to write a comparison file
somewhere else, which is useful when a parity test fails and the difference needs inspecting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pkg", required=True, help="package to run: 'defog' or 'defog_adapt'")
    ap.add_argument("--ckpts", required=True, help="directory holding the checkpoint files")
    ap.add_argument("--out", default=str(HERE / "golden" / "golden.json"))
    ap.add_argument("--only", default=None, help="comma-separated subset of cases")
    ap.add_argument("--jobs", type=int, default=7, help="cases run in parallel processes")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else None
    ctx = cases.Ctx(args.pkg, args.ckpts)
    print(f"running parity cases against {args.pkg!r}", flush=True)
    result = {"environment": cases.environment(ctx),
              "cases": cases.run_parallel(args.pkg, args.ckpts, names, jobs=args.jobs,
                                          log=lambda s: print(s, flush=True))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
