# Changelog

## 0.1.0 (unreleased)

First release: a standalone port of the `defog/` package of a research fork of
[DeFoG](https://arxiv.org/abs/2410.04263), keeping only features with a recorded positive result.
Import name `defog_adapt`. Code only: no model weights ship.

### Included

- **Frozen-base adapters**: `AdaLNAdapter` (gated FiLM, node-to-condition cross-attention, Fourier bands on
  the condition, an encoder for high-dimensional conditions), `AdapterModule`, `AdapterComposition`
  (product or mean of probability-space marginals), `AdaptedSampler`.
- **Learned node counts**: `LearnedSizeDistribution`, `ComposedSizeDistribution`, `fit_size_model`.
- **Feynman-Kac steering**: `FeynmanKacSampler` (with `trace`), `StandardisedEnergy`, `MoleculePropertyEnergy`,
  `LearnedPropertyEnergy`, `PropertyHead`, `fit_property_head`.
- **GDPO fine-tuning**: `GDPOTrainer`, `AdapterGDPOTrainer` and the rewards that were used with them
  (`SanityReward`, `PropertyMatchReward`, `HeadPropertyMatchReward`, `WeightedSanityPropertyReward`).
- **Inpainting** (`InpaintingSampler`, `SubgraphConstraint`) and a **spectrum energy** (`SpectrumEnergy`,
  `Band`): thin evidence, see `docs/caveats.md`.
- **Molecules**: encode and decode, validity and distribution metrics, `morgan_condition`, kekulized
  representations, and reference-split loaders for ZINC250k, MOSES and ChEMBL.
- `defog_adapt.presets`: the recorded recipes as plain data.
- `examples/`: a CPU quickstart, PyComex recipes (E1 base training, structural RL, adapter training, adapter
  RL), the property-head script and `steer_zinc.py`, the assembly.
- `docs/`: adapters, steering, RL, results, caveats, migration.
- `docs/assets/banner/`: the animated README banner (one recorded sampler run, base alone and then with a logP
  adapter, from the same noise), its recorded trajectory and the scripts that made it.
- The README opens with installation and a self-contained quickstart that runs on a CPU after `pip install` (and is
  executed by `tests/test_readme_quickstart.py`), and has a results figure (`docs/assets/results_logp.py`), a
  BibTeX block and badges. `CONTRIBUTING.md` and `SECURITY.md` are new.
- One extra, `chem` (RDKit). pandas, psutil and nvidia-ml-py are core dependencies; the contributor tools are a
  `dev` dependency group; PyComex, which the recipes need, is installed separately because it caps numpy below 2.

### Changed relative to the research fork

- Import name `defog` to `defog_adapt`; experiment helpers now live in `defog_adapt.domains.molecule`.
- Trainer defaults are the recorded recipes, and `AdapterGDPOTrainer(rollout_weight=...)` is **required**.
- Only probability-space guidance blending remains (rate-space blending was numerically broken above weight 1).
- `MoleculeDomain.encode` encodes an aromatic core in Kekule form when the vocabulary is kekulized.
- The library no longer selects a matplotlib backend at import time or leaks a pyplot figure per epoch.
- Reference data is read from `$DEFOG_ADAPT_HOME` or a per-user cache and is never downloaded silently.
- `defog_adapt.__version__` is read from the installed metadata.

### Removed relative to the research fork

Features with a null, negative, superseded or untried result: the other RL estimators (RAM, DAM, NFT), credit
assignment, distribution-fidelity penalties, the guidance family and its samplers, autoguidance, the refinement
sampler, the spectrum condition encoder, interior adapters, Feynman-Kac rejuvenation and beta schedules, and the
GuacaMol reference loader. The complete list, with reasons, is in `docs/migration.md`. Checkpoints that need a
removed feature raise an error that says so instead of loading a different model.

### Verified

- A golden-output parity test runs real checkpoints through the sampling, adapter, Feynman-Kac and RL code and
  matches the research package bit for bit.
- The unit suite passes with every dependency at its tested minimum (Python 3.10, torch 2.1, PyG 2.3, Lightning
  2.4, numpy 1.23) and on current releases.
- Every recipe runs end to end in its smoke configuration, on PyComex 0.13.1 and 0.29.
- A wheel and sdist build, pass `twine check`, install into a clean environment and import.
