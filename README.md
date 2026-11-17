# defog-adapt

Frozen-base **adapters**, **Feynman-Kac steering** and **RL fine-tuning** for
[DeFoG](https://arxiv.org/abs/2410.04263) discrete flow matching on graphs, with molecules as the
validated domain.

> **Status: pre-release, under construction.** This README is a placeholder; it is rewritten
> when the package is documented (see `CHANGELOG.md`). Nothing here is published yet.

This package keeps only what has a recorded positive result. Features that were tried and failed,
came out null, or were never tried are not included.

## Relationship to DeFoG

The base model follows the upstream DeFoG implementation (Qin, Madeira, Thanou, Frossard; ICML 2025,
MIT licensed). Everything built on top of it here was developed in a research fork. See `NOTICE` for
the full credits and the papers behind each method.

The import name is `defog_adapt`. It does not clash with the unrelated `defog` package on PyPI.

## License

MIT, see `LICENSE`. The upstream copyright notice is retained.
