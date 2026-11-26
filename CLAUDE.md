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
- `docs/` the documentation: Home and Getting started, `concepts/` (overview, adapters, steering, RL), `guides/`,
  `reference/` (main classes, recipes, migration) and `about/` (limits, results and caveats, contributing, citation)
- `mkdocs.yml` the documentation site (Material for MkDocs). Every page of `docs/` is in its nav, and code on its pages is
  included from the README, `docs/concepts/adapters.md` and the three tutorial scripts in `docs/guides/`
  (`train_an_unconditional_base.py`, `train_an_adapter.py`, `steer_toward_a_target.py`: the one copy of a tutorial's code,
  shown in pieces by section and in full at the end of its page) by `--8<--` markers, never copied

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
- **Documentation is tested.** `tests/test_docs.py` parses every code block (an include is expanded first), checks every
  `defog_adapt` import in one and every link and anchor. The tutorials are tested by `tests/test_tutorial_*.py` (and
  `test_steer_tutorial.py`): the structure of each page, every written-out setting against its preset and its table, the
  three scripts run one after the other on tiny data, and the style of the code (no line over 88 characters, a call with
  four or more arguments one per line). Every number on `docs/about/results-and-caveats.md` needs a source and its protocol.
- **The site is the documentation.** Every page of `docs/` is in the nav of `mkdocs.yml`, and a number on any page has
  the same source as on `docs/about/results-and-caveats.md`. `mkdocs build --strict` (group `docs`, no PyTorch) and
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
- **The PyPI page is the README, with its relative paths made absolute at build time** (hatch-fancy-pypi-readme, the rules
  in `pyproject.toml`); the README in the repository keeps them relative. `tests/test_pypi_readme.py` builds the wheel and
  checks the description it carries. A new kind of path in the README (a reference-style link, say) needs a rule.
- Commits and pushes to the working branch need no go-ahead. Releasing does: tags, PyPI uploads, making the repository
  public, merging a pull request.

## Releasing

Nothing publishes by accident. A pushed `v*` tag runs `.github/workflows/release.yml`, which builds the sdist and the
wheel, checks them with `twine check` and keeps them as the `dist` artifact of the run; a manual run of the workflow does
the same and nothing more, so it is the dry run. There is no TestPyPI step: only a PyPI account exists. To release (every
step that publishes needs the maintainer's go-ahead):

1. Set `version` in `pyproject.toml` and `CITATION.cff` (with its `date-released`) and date the top entry of
   `CHANGELOG.md`; `tests/test_release_metadata.py` checks that they agree.
2. Run the gates: the unit suite, the minimum-versions job in CI, the parity gate and `tests/test_recipes.py`.
3. Merge to `main`, wait for CI, then push the tag `v<version>` on that commit.
4. Upload what the tag's run built, so that PyPI gets the files CI checked: `gh run download <run-id> -n dist -D dist`,
   then `uv publish dist/*`, which reads the maintainer's PyPI token from `UV_PUBLISH_TOKEN`. (The first upload of a
   project needs an account-wide token, because a project-scoped one cannot exist yet: scope it to the project after.)
5. Look at the project page on PyPI, and install the release into a clean environment.

PyPI **Trusted Publishing** is the alternative to the token, and the job `publish-pypi` of the workflow is for it. It is
dormant: it runs only when the repository variable `PYPI_TRUSTED_PUBLISHING` is `true`. Set it up once, on the maintainer's
side: on PyPI add a *trusted publisher* for the project `defog-adapt` (this repository, workflow `release.yml`,
environment `pypi`), create the GitHub environment `pypi` with a required reviewer (possible for a public repository on
the free plan), and set the variable. From then on a tag push builds, waits for the reviewer and publishes, and the token
can be deleted.

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
