# Contributing to defog-adapt

Issues and pull requests are welcome. This page explains the one rule that is specific to this project, how to run
the checks, and what they guard.

## The scope rule: only features with a recorded positive result

`defog-adapt` deliberately contains **only features that have a recorded positive result**. Things that were tried
and came out null, negative or superseded are left out on purpose; the list, with the reasons, is in
[`docs/migration.md`](docs/migration.md#what-was-removed). So a new feature has to arrive with its evidence: the
protocol, the numbers and their run-to-run noise. Say in the pull request what the evidence is, and add the number
to [`docs/results.md`](docs/results.md) with its source. Bug fixes, documentation and tests need no such evidence.

## Setting up

```bash
git clone https://github.com/aimat-lab/defog-adapt && cd defog-adapt
uv venv --python 3.12 .venv && uv pip install -e ".[chem]" --group dev   # CPU PyTorch first if you have no GPU
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -n 8                       # the unit suite: no network, about 30 s
```

With plain pip (25.1 or newer), `pip install -e ".[chem]" --group dev` and `pytest` do the same. `chem` is the only
extra; the contributor tools are the `dev` dependency group. Keep `OMP_NUM_THREADS=2`: without it the suite takes
many times longer, because tiny tensors spread across every core thrash.

Two checks are opt-in because they need the research checkpoints and data. `tests/parity` runs real checkpoints
through the library and compares them with golden outputs from the research package; it is the release gate.
`tests/test_recipes.py` runs every recipe in `examples/` end to end (these also need PyComex, which caps numpy below
2, so give them an environment of their own and `pip install pycomex` there):

```bash
DEFOG_ADAPT_CKPTS=/path/to/ckpts python -m pytest -m parity
DEFOG_ADAPT_HOME=/path/to/data DEFOG_ADAPT_CKPTS=/path/to/ckpts python -m pytest tests/test_recipes.py
```

## Rules the checks enforce

- **Test your tests.** Check every new test against a deliberately broken implementation, after asserting that the
  unbroken one passes. Break the code under test, not the test. A test that cannot fail is not a test.
- **Examples must run.** A recipe or a README snippet that is not run has already rotted. The README quickstart is
  executed by `tests/test_readme_quickstart.py`; keep it, and `tests/test_recipes.py`, green when the API changes.
- **Documentation is tested.** `tests/test_docs.py` parses every code block, checks every `defog_adapt` import in one,
  and resolves every link and anchor. Every number in `docs/results.md` needs a source and its protocol.
- **Checkpoint compatibility.** Every config key present in existing checkpoints stays accepted; a removed feature
  accepts only its neutral value and otherwise raises a clear error.
- **Dependency floors are tested, not guessed.** The suite passes with every dependency at its minimum on Python
  3.10; change a floor only by running it.
- **No network and no local data in unit tests.**
- **Never commit** checkpoints, datasets, experiment output, cluster or account names, job IDs or local paths.
  `tests/test_no_private_references.py` fails on the generic patterns.

[`CLAUDE.md`](CLAUDE.md) has the code layout and the same conventions, written for anyone (or any tool) working in
the repository.

## Pull requests

Keep the pull request to one change. Run the unit suite, say what you changed and why, and update the documentation
and [`CHANGELOG.md`](CHANGELOG.md) where the change is visible to users. Security problems go to
[`SECURITY.md`](SECURITY.md), not to a public issue.
