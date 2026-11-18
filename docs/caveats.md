# Caveats: what the evidence does and does not support

`defog-adapt` contains only features that have a recorded positive result. That is a narrower claim
than "these features work", and this page says where the gap is. Read it before relying on a number in
[results.md](results.md).

## Provenance

* **Nothing was retrained for 0.1.** The code was extracted from a research fork, and the release gate
  is *equivalence*: a golden-output parity test that runs real checkpoints through the real sampling,
  Feynman-Kac and RL code paths and matches the research package bit for bit, plus the unit tests.
  The numbers in `results.md` are the research fork's, measured with its harness, and have not been
  reproduced end to end here. The one exception is the steering re-measurement at the end of
  `results.md`, which ran this repository's `examples/steer_zinc.py` on the same kind of protocol.
* **No model weights ship with 0.1.** The recorded results depend on checkpoints you have to train
  with the recipes in `examples/` (or obtain elsewhere).
* **The evaluation harness is not included.** The conditional numbers follow the FreeGress ZINC250k
  protocol (100 targets drawn from real molecules, 10 generations each, pooled error to the target)
  and were produced by a separate script. `examples/steer_zinc.py --evaluate` follows the same protocol
  shape, so numbers agree within run-to-run noise, but it is not bit-reproducible against the original.
  FCD, NSPDK and scaffold similarity are computed by tools this package does not depend on.
* **The recipes were validated by smoke runs.** Each runs end to end on CPU against the real reference
  data and checkpoints (`tests/test_recipes.py`), which shows the pipeline executes and its checks pass.
  It does not show that a full-scale run reproduces the recorded result.

## The adapter base is a re-packaged checkpoint

Every adapter result was measured on one ZINC base: the E1-recipe model (`examples/training__e1.py`,
`defog_adapt.presets.E1_MODEL`) after two rounds of structural RL (`examples/rl_sanity__zinc.py`), so the
pipeline in `examples/` reproduces the lineage the adapters sat on.

The trap is in reading the file they were trained against. It was re-packaged, and the hyperparameters
stored in it (identity time distortion, edge-loss weight 1, 100 sampling steps, no learning-rate schedule)
are defaults filled in at export, **not** the configuration it was trained with: its weights are identical,
tensor for tensor, to the RL model, which carries the E1 recipe (`tests/test_presets.py` pins both facts when
the checkpoints are available). So a loaded model's stored sampling defaults are not to be trusted on a
checkpoint of unknown history. Pass the sampling arguments explicitly, as every recipe here does, and check a
checkpoint's provenance before reading its hyperparameters as its recipe.

The base-model results (the E1 rows) used the same recipe directly.

## Noise, and how much a number can be trusted

* **A 100-target error is good to about 0.011.** That is the spread of an identical configuration
  re-run with different seeds, measured three times. Differences below it are not differences.
* **Two seeds of the same adapter recipe differ by 0.058 in logP error** (0.3068 against 0.3646). That
  is larger than most of the capacity differences between architecture variants, so no capacity trend
  should be read from single seeds. Only the step from FiLM-only to cross-attention (-0.19) and the epoch
  count (-0.07) clear it.
* **Seed duplicates change conclusions.** A weighted-reward RL run on QED gave -26% on one seed and
  -9.8% on a duplicate of the same configuration. The direction was safe (12 of 12 level-arm deltas
  negative); the magnitude was not.
* **TPSA and SAScore adapters are single-seed.** They were measured on the same protocol but without
  seed duplicates, and the head-reward RL on SAScore regressed in its second round.
* **Targets matter.** A 100-target error mixes "does it steer" with "were the targets worth steering
  to": about a third of real-molecule targets sit near the model's own mean, where a constant predictor
  is unbeatable and any generative sampler loses on its own spread. Judge the low and high thirds. QED
  is the sharpest case: its raw error looks poor against a do-nothing baseline measured on the wrong
  target set, and is a +26% skill at both ends against the right one.

## Where steering does not work (yet)

* **The low end of logP.** The lowest third of targets moved least (2% with the first adapter, against
  22% and 26% for the others). Part of that is the base being handed the wrong atom count (a learned
  size model recovered 8% on that third, p = 0.035 uncorrected, 0.105 under Bonferroni), and the rest is
  what the frozen base can reach. An adapter does not widen it.
* **QED is the property where a low error is easiest to earn without steering.** The range is narrow
  and saturated, so about a third of real-molecule targets sit where a constant predictor beats any
  sampler. Judge the low and high thirds: an early QED adapter had +26% skill over doing nothing at both
  ends and -174% in the middle, and read as "worse than doing nothing" only when compared against a
  baseline measured on the wrong target set.
* **High guidance weights break validity.** Past about 2 to 2.5 validity falls away fast (logP at
  w = 4: 0.466), driven by disconnected fragments. The optimum is per property and per adapter; see
  `presets.ADAPTER_WEIGHT`.

## Feynman-Kac steering

* **It costs diversity.** At beta = 20 logP uniqueness is about 0.76, and at beta = 80 about 0.54. That
  is the steered distribution being genuinely narrower, not a fault: cloned particles have hundreds of
  stochastic steps left to diverge, and no impoverishment cure helped (rejuvenation was strictly
  dominated on both axes; lower-variance resampling moved nothing consistently).
* **beta matters far more than the particle count.** Going from 10 to 64 particles moved logP error from
  0.2299 to 0.2129, an asymptote; the earlier jump from beta = 2.5 to 20 was worth 0.083. Do not spend
  compute on particles before tuning beta.
* **It looked useless until the beta was fixed.** At beta = 2.5 the particle system was starved (resampling
  on 16% of checkpoints), and an earlier reading, that Feynman-Kac was only useful at low step counts,
  was an artefact of it. The step-count dependence was **not re-measured at beta = 20**; the only
  measurement (gain growing from +0.0001 at 500 steps to +0.0284 at 50) is at the starved beta.
  Read `FeynmanKacSampler.trace`: a run that never resamples is the unguided sampler with extra passes.
* **The gains do not add.** Policy-gradient fine-tuning and Feynman-Kac correct the same error. On top of
  FK the RL increment was not significant (logP p = 0.66 and 0.32, QED p = 0.69 for both seeds).
* **Drawing 64 and keeping the 10 closest is a different measurement**, and nearly as good: the no-FK
  control at 64 particles scores 0.0408 on that order statistic against FK's 0.0347.

## Reinforcement learning

* **One round is the useful depth for adapters.** A second round of head-reward RL on QED was a
  significant regression on the 100-target protocol while its own internal evaluation claimed the
  opposite; a second round of weighted-reward RL bought nothing. (For the *unconditional* structural RL,
  two rounds is the recorded depth and a third moved nothing.)
* **The internal evaluation overstated QED.** For the RL-only effect it read -15.3% where the 100-target
  protocol measured -10.7%, and on logP it was accurate (-9.5% against -10.3%). It scored two fixed
  targets, and best-of-five snapshot selection overfits two points. The recipe now probes nine fixed
  targets and always computes the true property beside the selected one.
* **A learned head as a reward is only as good as the head.** The logP head scored 0.019 of a standard
  deviation, the QED head 0.155; the QED head's ranking of candidate snapshots inverted against RDKit
  once, and truth-based selection beat head-based selection by 0.0084 (p = 0.043).
* **RL can unlearn what nothing measures.** In a fingerprint-adapter experiment (not part of this package),
  optimising Tanimoto similarity plus connectivity erased the adapter's inference of molecule size: a
  correlation of +0.74 between target size and generated size fell to about zero. The failure modes you
  measure are the ones that are protected, so guard anything you rely on.
* **Reward hacking is a representation question as much as a reward one.** The structural sanity reward
  raised validity on an aromatic MOSES base while FCD doubled on four of four seeds; on the kekulized
  representation the same reward does not. A sanity gain with a degraded FCD is a failed run.

## Representation

* **Molecules must be kekulized.** An aromatic bond class is a promise about a whole ring system that
  RDKit checks by kekulizing, and a per-edge model cannot keep it: on the aromatic MOSES base 118 of 120
  hard validity failures were kekulization errors. The kekulized representations are the recorded
  results; `aromatic_v1` is kept only so that older checkpoints load.
* **A checkpoint decodes correctly only under the vocabulary it was trained with.** Decoding with the
  wrong one does not raise: it yields plausible molecules made of the wrong elements. Use
  `defog_adapt.data.vocabulary.check_model`, which every recipe does.
* **The graph carries no formal charge or stereochemistry.** A third of ZINC is charged, and that is
  exactly what makes a molecule low-logP, so the property of a source SMILES and of the molecule the
  graph represents differ by up to 1.6 logP at the 5th percentile. Every recorded result labelled by the
  *decoded* graph. A head, an adapter and a size model must agree on this or they pull against each other
  (a head accurate to 0.083 against decoded labels reads 0.461 against source labels).

## Features with thin evidence

These are included deliberately, and each deserves less trust than the rest.

| feature | what is recorded | what is not |
|---|---|---|
| **Inpainting** (`InpaintingSampler`, `SubgraphConstraint`) | One demonstration on a benzene core: about 84% validity and 99% core preservation, on an unconditional model that this package does not ship. A core with no free valence gives about 0% valid, so check `MoleculeDomain.core_valence_report` first. | A benchmark, a comparison against another inpainting method, or a conditional use. |
| **Spectrum energy** (`SpectrumEnergy`, `Band`) | The energy and its one-sided band penalty are unit-tested, and a first steered run exposed and fixed a normalisation bug (three requests returned identical molecules because the weights were uniform). The forward predictor it was used with scored carbonyl present/absent AUC 0.93. | The predictor is **not shipped**, you supply one. There is no recorded end-to-end steering result you can reproduce from this package, and the encoder side (`SpectrumEncoder`) was dropped. |
| **Fingerprint conditioning** (`FingerprintEncoder`, `morgan_condition`) | A cross-attention adapter on 1024-bit Morgan counts lifted Tanimoto similarity by +0.118 to +0.157 over the unconditional baseline at its best weight (1.5). It does no better on trained-on targets than on unseen ones, so capacity is not the limit. | A comparison with a binary-fingerprint control (declined), and RL on it ratchets connectivity but not similarity. The ceiling is the frozen base. |
| **Legacy input-CFG model** (`DeFoGModel` with `cond_dim > 0`) | Kept for checkpoints trained that way. | It is superseded by adapters for every recorded result. |

## What was removed, and why

Features with a null, negative, superseded or untried result were left out rather than shipped with a
caveat. The reasons are recorded in [migration.md](migration.md#what-was-removed). In short: other RL
estimators (a four-seed benchmark was not significant, p = 0.15), distribution-fidelity penalties (one
term *rewarded* the hack it was meant to stop), exact guidance and the guided/refinement samplers
(superseded or null), rate-space guidance blending (numerically broken above weight 1), beta schedules
and rejuvenation (dominated), and interior adapters.

## Dependencies and platforms

* The lower bounds in `pyproject.toml` were **tested**: the full suite passes with every dependency at
  exactly its minimum on Python 3.10 (torch 2.1, PyG 2.3, Lightning 2.4, numpy 1.23, matplotlib 3.5.3).
  `pytorch_lightning` 2.0 to 2.3 cannot be imported in a current environment at all (they import
  `pkg_resources`, which setuptools 81 removed), which is why the floor is 2.4.
* PyComex, which the recipes need, caps `numpy<2`. It is therefore not a dependency of this package (it would
  hold every user on numpy 1.x): an environment with PyComex resolves to a numpy 1.x stack, and
  `defog-adapt[chem]` alone does not.
* The test suite and the recipes were run on **CPU and, for the steering assembly, one consumer GPU**.
  Multi-GPU and cluster behaviour (the chained-link training, `PerLinkTimer`) is covered by unit tests
  but was not exercised here.
* Checkpoints with interior adapters (`interior_ff` / `interior_attn`) or a spectrum encoder will not
  load: they raise an error that says so, rather than loading a different model.
