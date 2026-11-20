# CLAUDE.md

Guidance for working in this repository.

## What this is

`defog-adapt` (import `defog_adapt`) is a standalone, PyPI-targeted port of the `defog/` package of a
research fork of DeFoG (discrete flow matching for graph generation). It keeps **only features with
a recorded positive result**; null, negative, superseded and untried features were left out on
purpose. Do not re-add a feature without evidence that it works, and say what the evidence is.

## Layout

- `src/defog_adapt/core/` base model, adapters, samplers, Feynman-Kac steering, energies, RL, size models
- `src/defog_adapt/domains/molecule.py` RDKit decode/encode/metrics (needs the `chem` extra)
- `src/defog_adapt/data/` reference-split loaders (ZINC250k, MOSES, ChEMBL) and vocabularies
- `src/defog_adapt/presets.py` the recorded recipes as plain data (every value has its evidence in a comment)
- `tests/` CPU-only unit tests; `tests/parity/` opt-in comparison against golden outputs of the research repo;
  `tests/test_recipes.py` opt-in smoke runs of every recipe
- `examples/` the quickstart, the PyComex recipes (optional `recipes` extra) and `steer_zinc.py`, the assembly
- `docs/` adapters, steering, rl, results, caveats, migration
- `mkdocs.yml`, `docs/{index,getting-started,concepts}.md`, `docs/{guides,reference,about}/` the documentation site
  (Material for MkDocs): the short, high-level version. The six detailed notes above are excluded from it on purpose,
  and code on its pages is included from the README and the notes by `--8<--` markers, never copied

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[chem]" --group dev   # CPU torch first: see the README
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -n 8                       # unit tests, no network, ~30 s
DEFOG_ADAPT_CKPTS=/path/to/ckpts .venv/bin/python -m pytest -m parity   # parity gate (local only)
DEFOG_ADAPT_HOME=/path/to/data DEFOG_ADAPT_CKPTS=/path/to/ckpts .venv/bin/python -m pytest tests/test_recipes.py
uv pip install --group docs && mkdocs build --strict                     # the documentation site: no PyTorch needed
```

Without `OMP_NUM_THREADS=2` the suite takes many times longer: tiny tensors across every core thrash. The
recipes need PyComex (`pip install pycomex`), which caps `numpy<2`, so run them in an environment of their own.

## Rules

- **Parity gate.** After every change to a ported module, the parity tests must still match the golden
  outputs bit for bit. Run it on a `git archive` snapshot, not the working tree (the library imports lazily).
  Any drift must be understood, not tolerated.
- **Checkpoint compatibility.** Every config key present in existing checkpoints stays accepted; removed
  features accept only their neutral value and otherwise raise a clear error.
- **Test your tests.** Every new test is checked against a deliberately broken implementation, **after**
  asserting that the unbroken one passes (a failing baseline makes every "caught" meaningless). Mutate the
  code under test, not the test. A test that cannot fail is not a test; this found more than one vacuous one.
- **Examples must run.** A recipe that is not run has already rotted. Keep `tests/test_recipes.py` green when
  the library API changes, and assert on a recipe's own output, not on PyComex's banner (it varies by version).
- **Documentation is tested.** `tests/test_docs.py` parses every code block, checks every `defog_adapt` import
  in one and every link and anchor. Every number in `docs/results.md` needs a source and its protocol.
- **The site stays the short version.** A new page goes in the nav of `mkdocs.yml`, detail stays in the notes, and a
  number on the site has the same source as in `docs/results.md`. `mkdocs build --strict` (group `docs`, no PyTorch) and
  `tests/test_docs_site.py` guard the nav, the exclusions, the includes and the config. Keep the config portable (no
  hooks, no plugin beyond search, publish through the Pages artifact flow, never `gh-deploy`): Material for MkDocs
  ends on 2027-05-05 and its successor reads the same file. Publishing is dormant until the repository is public.
- **No network and no local data in unit tests.** Reference-split tests pass `download=False`.
- **Never commit** checkpoints, datasets, experiment output, cluster or account names, job IDs, local paths, or
  the names of sibling projects. `tests/test_no_private_references.py` fails on the generic patterns (paths,
  scheduler commands, job numbers, addresses) everywhere. The SPECIFIC names (a published list of private names is
  itself a leak) are read from a file named by `DEFOG_ADAPT_PRIVATE_PATTERNS`, one regex per line, that the
  maintainer keeps OUTSIDE the repository; set it when running the suite locally. Before publishing a history,
  scan **every commit**, not just the tip: the intermediate commits of a port are where leaks hide.
- **Dependency floors are tested**, not guessed: the suite passes with every dependency at its minimum on
  Python 3.10. Change a floor only by running it.
- Nothing is pushed or published (GitHub, TestPyPI, PyPI) without an explicit go-ahead.

## Releasing

Nothing publishes by accident: the workflow in `.github/workflows/release.yml` needs a `v*` tag for the real index
and uses PyPI **Trusted Publishing** (no API token exists). One-time setup, on the maintainer's side:

1. On PyPI and TestPyPI, add a *trusted publisher* for the project `defog-adapt`: owner and repository of the GitHub
   remote, workflow `release.yml`, environments `pypi` and `testpypi` respectively.
2. In the GitHub repository settings, create the environments `testpypi` and `pypi` and give each a required reviewer.
3. To release: set `version` in `pyproject.toml` (and `CITATION.cff`), move the changelog entry out of "unreleased",
   run the gates (unit suite, floors, parity, `tests/test_recipes.py`), push, run the workflow manually (TestPyPI
   only), check the upload, then push the `v<version>` tag.

## Behaviour notes that matter

- Molecules must be **kekulized**: the aromatic bond class is lossy (118 of 120 hard validity failures on the
  aromatic MOSES base were kekulization errors).
- A graph stores **no formal charge**: the property of a source SMILES and of the decoded graph differ by up to
  1.6 logP. Every recorded result labelled by the decoded graph (`property_from="decoded"`), and the adapter, head
  and size model must agree.
- **Feynman-Kac `beta` is dimensionless only through `StandardisedEnergy`.** At `beta` 20 it is the largest lever
  measured, at a cost in diversity; a starved `beta` (2.5) made it look useless. Read `FeynmanKacSampler.trace`.
- **Do not read a re-packaged checkpoint's stored hyperparameters as its training configuration.** The adapter base
  is weight-identical to the two-round RL model and carries the E1 recipe; its file stores defaults.
- RL: `rollout_weight` must be the deployed guidance weight; one round for adapters; probe many fixed targets and
  always compute the true property beside the selection metric.
- Gains from RL and from Feynman-Kac do not add.
