# Contributing

Issues and pull requests are welcome. This is the short version;
[CONTRIBUTING.md](https://github.com/aimat-lab/defog-adapt/blob/main/CONTRIBUTING.md) has the details.

## The scope rule

defog-adapt contains **only features that have a recorded positive result**. What was tried and came out null,
negative or superseded was left out on purpose
([what was left out, and why](https://github.com/aimat-lab/defog-adapt/blob/main/docs/migration.md#what-was-removed)).
So a new feature has to arrive with its evidence: the protocol, the numbers and their run-to-run noise. Say in the
pull request what the evidence is. Bug fixes, documentation and tests need no such evidence.

## Run the checks

```bash
git clone https://github.com/aimat-lab/defog-adapt && cd defog-adapt
uv venv --python 3.12 .venv && uv pip install -e ".[chem]" --group dev   # CPU PyTorch first if you have no GPU
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -n 8                       # the unit suite: no network, about 30 s
```

Keep `OMP_NUM_THREADS=2`: without it the suite takes many times longer, because tiny tensors spread across every core
thrash.

## Build this site

```bash
uv pip install --group docs
mkdocs serve            # a live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs: a broken link, or a page missing from the nav, fails the build
```

The site is the short, high-level documentation. The detailed notes in `docs/` (adapters, steering, RL, caveats,
results and migration) stay in the repository and are not published here. Code on these pages is included from the
README and from those notes, so each snippet has one copy, and the tests check it.

## Pull requests

Keep a pull request to one change. Run the unit suite, say what you changed and why, and update the documentation and
the changelog where a user can see the change. Security problems go to
[SECURITY.md](https://github.com/aimat-lab/defog-adapt/blob/main/SECURITY.md), not to a public issue.
