# CLAUDE.md

Guidance for working in this repository.

## What this is

`defog-adapt` (import `defog_adapt`) is a standalone, PyPI-targeted port of the `defog/` package of a
research fork of DeFoG (discrete flow matching for graph generation). It keeps **only features with
a recorded positive result**; null, negative, superseded and untried features were left out on
purpose. Do not re-add a feature without evidence that it works.

## Layout

- `src/defog_adapt/core/` base model, adapters, samplers, Feynman-Kac steering, RL, size models
- `src/defog_adapt/domains/molecule.py` RDKit decode/encode/metrics (needs the `chem` extra)
- `src/defog_adapt/data/` reference-split loaders (ZINC250k, MOSES) and vocabularies
- `tests/` CPU-only unit tests; `tests/parity/` opt-in comparison against golden outputs of the
  research repo
- `examples/` PyComex recipes (optional `recipes` extra); `docs/`

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[chem,data,dev]"   # CPU torch: see docs
.venv/bin/python -m pytest                                              # unit tests, no network
DEFOG_ADAPT_CKPTS=/path/to/ckpts .venv/bin/python -m pytest -m parity   # parity gate (local only)
```

## Rules

- **Parity gate.** After every change to a ported module, the parity tests must still match the
  golden outputs. Any drift must be understood, not tolerated.
- **Checkpoint compatibility.** Every config key present in existing checkpoints stays accepted;
  removed features accept only their neutral value and otherwise raise a clear error.
- **No network and no local data in unit tests.** Reference-split tests pass `download=False`.
- **Never commit** checkpoints, datasets, experiment output, cluster names, job IDs or local paths.
- Nothing is pushed or published (GitHub, TestPyPI, PyPI) without an explicit go-ahead.
- Behaviour notes that matter: molecules must be kekulized (the aromatic bond class is lossy);
  Feynman-Kac steering helps at low step counts only; adapter results were measured on a base that
  had itself been RL fine-tuned.
