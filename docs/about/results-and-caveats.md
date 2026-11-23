# Results and caveats

What was measured, how far to trust it, and where the numbers stop. `defog-adapt` contains only features that have a
recorded positive result. That is a narrower claim than "these features work", and this page says where the gap is. Read
it before relying on a number anywhere on this site.

## The headline comparison

![Bar chart of the logP error of the five approaches in the table below, lower is better.](../assets/results_logp.png#only-light)
![Bar chart of the logP error of the five approaches in the table below, lower is better.](../assets/results_logp_dark.png#only-dark)

Five ways of steering a ZINC250k model toward a logP target. Every model was trained in this repository with the recipes
in `examples/`, and every number was measured with `examples/compare_arms__zinc.py`. Held-out **test split**, 100 targets
× 10 generations, mean ± sd over three sampling seeds (each approach was trained once). Lower error is better. The data
behind the figure and the table is `docs/assets/results_logp.json`.

| approach | what it is | setting | logP error | validity | uniqueness |
|---|---|---|---|---|---|
| unconditional base | the adapter base checkpoint as it is: nothing steers it | - | **1.2602** ± 0.0260 | 0.996 | 1.000 |
| end-to-end conditional | the E1 recipe trained end to end on logP, 300 epochs | guidance 1.0 | **0.1810** ± 0.0058 | 0.991 | 1.000 |
| FiLM adapter | frozen base + adapter with FiLM only, 80 epochs | guidance 2.0 | **0.4456** ± 0.0201 | 0.985 | 1.000 |
| cross-attention adapter | frozen base + adapter with node-to-condition cross-attention and Fourier bands, 80 epochs | guidance 2.0 | **0.2721** ± 0.0071 | 0.983 | 1.000 |
| + Feynman-Kac | the cross-attention adapter, with Feynman-Kac particle steering on top | guidance 2.0, beta 20, 10 particles | **0.2161** ± 0.0024 | 0.990 | 0.790 |

The unconditional base is the checkpoint described under
[The adapter base is a re-packaged checkpoint](#the-adapter-base-is-a-re-packaged-checkpoint), and it is the same frozen
model under all three adapter rows.

**Reading it.**

* The model trained end to end on logP has the lowest error here, below every other approach on every sampling seed.
  Its lead over the adapter alone, 0.0911 ± 0.0091 (paired per seed), is larger than the 0.058 recorded between two
  trainings of one adapter recipe (see [Noise](#noise)). Its lead over the adapter with Feynman-Kac, 0.0351 ± 0.0078, is
  smaller, so this comparison does not order those two (see *What these numbers do and do not show*).
* Measured from the unconditional base to the end-to-end model, the cross-attention adapter closes about 92% of the
  distance and Feynman-Kac about 97%; the FiLM adapter, at the same training budget, about 75%.
* The cross-attention adapter beats the FiLM adapter by 0.1736 ± 0.0198 (on all three seeds), and Feynman-Kac improves
  the same adapter by 0.0560 ± 0.0089 (on all three seeds).
* Validity is 0.983 to 0.991 for every conditional approach (0.996 for the base). Uniqueness is 1.000 everywhere except
  under Feynman-Kac, where it is 0.790: about one valid generation in five repeats another.
* What each cost, as wall-clock on one shared 24 GB GPU: the end-to-end model 36 h 23 min, the cross-attention adapter
  13 h 41 min, the FiLM adapter 12 h 04 min, and the property head with the size model 53 min. The base the adapters
  share was trained once, earlier, and is not counted.

### How it was measured

* **Targets.** 100 per seed, drawn from the real molecules of the held-out test split (their *decoded* logP). Seeds 42,
  43 and 44 each draw their own 100, and all five approaches see the same ones within a seed, so the differences between
  approaches are paired.
* **Error.** The pooled mean absolute difference between each valid generation's decoded logP and its target, over 10
  generations per target (Feynman-Kac: 10 particles). Validity is the fraction of generations that decode to a valid
  molecule; uniqueness is the fraction of the valid ones that are distinct, pooled over all generations.
* **Sampling.** 500 steps, eta 25, omega 0, polydec time distortion; Feynman-Kac with `beta` 20 and the `FK_STEERING`
  schedule, scored by a learned property head. The four conditional approaches draw node counts from the learned
  `P(n | logP)`; the unconditional base cannot, and draws from the dataset's marginal.
* **Settings were chosen on validation, once, before any test number was read.** Per approach, the guidance weight with
  the lowest validation error among those whose validity is at least 0.97 (seed 42, 100 targets × 10 generations; 1.0 to
  2.5 for the adapters, 1.0 to 2.0 for the end-to-end model). A candidate within the 0.011 run-to-run noise floor of the
  best loses to the recipe's own checkpoint, then to the lower weight. Feynman-Kac uses the cross-attention adapter's
  weight. The test split was then read once.

### How each approach was made

* **Unconditional base.** The adapter base checkpoint: the E1 recipe followed by two rounds of structural RL, sampled
  with nothing steering it.
* **End to end.** `training__e1.py --CONDITION "'logp'"`: the E1 recipe with every setting unchanged and a logP condition
  added (decoded-graph labels, condition dropped on 10% of training steps). 300 epochs, with the recorded batch of 256
  accumulated from two micro-batches of 128 (`MICRO_BATCH_SIZE`). The checkpoint with the lowest validation loss (epoch
  278) is the one measured. **No structural-RL clean-up followed**, unlike the base.
* **FiLM adapter.** `adapter_training__zinc.py --XATTN_TOKENS 0 --COND_FOURIER 0`: the FiLM-only architecture, 2.75 M
  parameters, 80 epochs.
* **Cross-attention adapter.** The same recipe with its defaults: node-to-condition cross-attention and three Fourier
  bands, 6.64 M parameters, 80 epochs. **+ Feynman-Kac** adds the particle steering on top of this adapter.
* **Property head and size model.** `fit_property_head__zinc.py` with its defaults, on the decoded-label convention of the
  adapters. Every model was trained once, with seed 42.

### What these numbers do and do not show

* **The sd is over sampling seeds, not over trainings.** Each approach was trained once, so the standard deviations say how
  much the *measurement* moves with the target set and the sampling noise, not how much the *model* would move. The
  research fork recorded 0.058 between two trainings of one adapter recipe (see [Noise](#noise)). A gap between two
  different trained models of about that size or less, such as Feynman-Kac against the end-to-end model (0.035), is not
  an ordering this comparison can establish; larger gaps (the adapters against the base, FiLM against cross-attention, the
  end-to-end model against the adapter alone) exceed it, though each still comes from one training. Feynman-Kac against
  the adapter it steers is a paired difference within one trained model and does not carry that noise.
* **The lineages differ.** The base, and so every adapter, had two structural-RL clean-up rounds after its E1 training; the
  end-to-end model had none. Its validity (0.991) is nonetheless as high as the adapters', but the comparison is not
  like for like in how the models were made.
* **The end-to-end model was not guided.** The library's guidance for such a model blends rate matrices, which breaks above
  weight 1 (on validation, weight 2.0 gave an error of 0.6892 and a validity of 0.331), so it was measured at weight 1.0. The
  adapters use probability-space blending and are measured at 2.0. A guided end-to-end model is not measured here, and could
  score lower.
* **The FiLM adapter here is an equal-budget baseline.** It was trained as long as the cross-attention adapter (80
  epochs), not for the 20 epochs of the research fork's comparison, so the gap between them here is the architecture alone;
  the research fork's step also contained the difference in training length.
* **One hardware and software setup**, one frozen base and one property. GPU sampling is not bit-reproducible, so a re-run
  agrees within the noise, not exactly.

### Reproducing it

Trained and measured on 24 GB consumer GPUs (RTX 4090), under Python 3.12, PyTorch 2.13.0 (CUDA 13.0), PyTorch Geometric
2.8.1, PyTorch Lightning 2.6.6, RDKit 2026.9.2 and numpy 2.5.5. The adapters, the head and the size model were trained at
commit `c6b64a0`, the end-to-end model at `7f961db`, and every measurement was made at `c44e6fc`. Between those commits the
code changed only by the `MICRO_BATCH_SIZE` option and the rename of `AdaLNAdapter` to `CrossAttentionAdapter`, which is
the same class.

```bash
# the unconditional base is the adapter base checkpoint (see below); everything else is trained once
python examples/training__e1.py --DATASET "'zinc'" --CONDITION "'logp'" --MICRO_BATCH_SIZE 128 \
    --CKPT_DIR "'runs/e2e'" --MAX_TIME_HOURS 23 --SKIP_FINAL_EVAL True
python examples/adapter_training__zinc.py --BASE_CKPT "'BASE'" --XATTN_TOKENS 0 --COND_FOURIER 0   # the FiLM adapter
python examples/adapter_training__zinc.py --BASE_CKPT "'BASE'"                                    # the cross-attention adapter
python examples/fit_property_head__zinc.py --base BASE --property logp --out HEADS/logp_head --with-size-model
# choose the guidance weights on --split validation, then read the test split once:
python examples/compare_arms__zinc.py run --split test --seeds 42 43 44 --out RESULTS \
    --base BASE --conditional runs/e2e/best_model.ckpt --conditional-weight 1.0 \
    --film FILM.ckpt --film-weight 2.0 --xattn XATTN.ckpt --xattn-weight 2.0 \
    --head HEADS/logp_head.ckpt --size HEADS/logp_head_size.ckpt
python examples/compare_arms__zinc.py summarize RESULTS
```

## What the wider evidence does and does not support

### Provenance

* **Most numbers were not retrained for 0.1.** The code was extracted from a research fork, and the release gate is
  *equivalence*: a golden-output parity test that runs real checkpoints through the real sampling, Feynman-Kac and RL code
  paths and matches the research package bit for bit, plus the unit tests. The numbers quoted on the concept pages, for
  adapters on four properties, learned node counts, Feynman-Kac steering, RL, inpainting and the spectrum energy, are the
  research fork's, measured with its own harness, and have not been reproduced end to end here. The exceptions are the
  headline comparison above and a smaller check of the steering assembly with `examples/steer_zinc.py --evaluate` (40
  validation targets, one seed), where the Feynman-Kac effect reproduced closely (-28.3% against -26.6% recorded). That check
  is a consistency check, not a reproduction of the table.
* **No model weights ship with 0.1.** The recorded results depend on checkpoints you train with the recipes in
  `examples/` (or obtain elsewhere).
* **The harness behind the older numbers is not included.** They follow the FreeGress ZINC250k protocol (100 targets drawn
  from real molecules, 10 generations each, pooled error to the target) and were produced by a separate script.
  `examples/steer_zinc.py --evaluate` follows the same protocol shape, so numbers agree within run-to-run noise, but it is
  not bit-reproducible against the original. The harness behind the headline comparison, `examples/compare_arms__zinc.py`,
  is included. FCD, NSPDK and scaffold similarity are computed by tools this package does not depend on.
* **The recipes were validated by smoke runs.** Each runs end to end on CPU against the real reference data and
  checkpoints (`tests/test_recipes.py`), which shows that the pipeline executes and its checks pass. It does not show that
  a full-scale run reproduces a recorded result; the headline comparison is a full-scale run of the logP path.

### The adapter base is a re-packaged checkpoint

Every adapter result was measured on one ZINC base: the E1-recipe model (`examples/training__e1.py`,
`defog_adapt.presets.E1_MODEL`) after two rounds of structural RL (`examples/rl_sanity__zinc.py`), so the pipeline in
`examples/` reproduces the lineage the adapters sat on.

The trap is in reading the file they were trained against. It was re-packaged, and the hyperparameters stored in it
(identity time distortion, edge-loss weight 1, 100 sampling steps, no learning-rate schedule) are defaults filled in at
export, **not** the configuration it was trained with: its weights are identical, tensor for tensor, to the RL model, which
carries the E1 recipe (`tests/test_presets.py` pins both facts when the checkpoints are available). So pass the sampling
arguments explicitly, as every recipe here does, and check a checkpoint's provenance before reading its hyperparameters
as its recipe.

### Noise

* **A 100-target error is good to about 0.011.** That is the spread of an identical configuration re-run with different
  seeds, measured three times. Differences below it are not differences.
* **Two seeds of the same adapter recipe differ by 0.058 in logP error** (0.3068 against 0.3646). That is larger than most
  of the capacity differences between architecture variants, so no capacity trend should be read from single seeds. Only
  the step from FiLM-only to cross-attention (-0.19) and the epoch count (-0.07) clear it.
* **Seed duplicates change conclusions.** A weighted-reward RL run on QED gave -26% on one seed and -9.8% on a duplicate of
  the same configuration. The direction was safe (12 of 12 level-arm deltas negative); the magnitude was not. The TPSA and
  SAScore adapters are single-seed.
* **Targets matter.** A 100-target error mixes "does it steer" with "were the targets worth steering to": about a third of
  real-molecule targets sit near the model's own mean, where a constant predictor is unbeatable and any generative sampler
  loses on its own spread. Judge the low and high thirds.

### Where steering does not work

* **The low end of logP.** The lowest third of targets moved least (2% with the first adapter, against 22% and 26% for the
  others). Part of that is the base being handed the wrong atom count (a learned size model recovered 8% on that third,
  p = 0.035 uncorrected, 0.105 under Bonferroni), and the rest is what the frozen base can reach. An adapter does not
  widen it.
* **QED is the property where a low error is easiest to earn without steering.** The range is narrow and saturated, so
  about a third of real-molecule targets sit where a constant predictor beats any sampler. An early QED adapter had +26%
  skill over doing nothing at both ends and -174% in the middle, and read as "worse than doing nothing" only when compared
  against a baseline measured on the wrong target set.
* **High guidance weights break validity.** Past about 2 to 2.5 validity falls away fast (logP at w = 4: 0.466), driven by
  disconnected fragments. The optimum is per property and per adapter; see `presets.ADAPTER_WEIGHT`.

### Feynman-Kac steering and RL

* **Feynman-Kac costs diversity.** In the research fork's measurements logP uniqueness was about 0.76 at beta 20 and about
  0.54 at beta 80. That is the steered distribution being genuinely narrower, not a fault: no impoverishment cure helped.
* **beta matters far more than the particle count.** Going from 10 to 64 particles moved logP error from 0.2299 to 0.2129;
  the earlier jump from beta 2.5 to 20 was worth 0.083. At beta 2.5 the particle system was starved (resampling on 16% of
  checkpoints) and Feynman-Kac looked useless, so read `FeynmanKacSampler.trace`: a run that never resamples is the
  unguided sampler with extra passes.
* **The gains do not add.** Policy-gradient fine-tuning and Feynman-Kac correct the same error. On top of Feynman-Kac the
  RL increment was not significant (logP p = 0.66 and 0.32, QED p = 0.69 for both seeds).
* **One round is the useful depth for adapters.** A second round of head-reward RL on QED was a significant regression on
  the 100-target protocol while its own internal evaluation claimed the opposite. For the unconditional structural RL, two
  rounds is the recorded depth and a third moved nothing.
* **A learned head as a reward is only as good as the head, and RL can unlearn what nothing measures.** The QED head's
  ranking of candidate snapshots inverted against RDKit once, and truth-based selection beat head-based selection by 0.0084
  (p = 0.043). In a fingerprint-adapter experiment (not part of this package), optimising Tanimoto similarity plus
  connectivity erased the adapter's inference of molecule size. Guard anything you rely on.

### Representation

* **A checkpoint decodes correctly only under the vocabulary it was trained with.** Decoding with the wrong one does not
  raise: it yields plausible molecules made of the wrong elements. Use `defog_adapt.data.vocabulary.check_model`, which
  every recipe does. Kekulization and the missing formal charge are on [Limits](limits.md).

### Features with thin evidence

These are included deliberately, and each deserves less trust than the rest.

| feature | what is recorded | what is not |
|---|---|---|
| **Inpainting** (`InpaintingSampler`, `SubgraphConstraint`) | One demonstration on a benzene core: about 84% validity and 99% core preservation, on an unconditional model that this package does not ship. A core with no free valence gives about 0% valid, so check `MoleculeDomain.core_valence_report` first. | A benchmark, a comparison against another inpainting method, or a conditional use. |
| **Spectrum energy** (`SpectrumEnergy`, `Band`) | The energy and its one-sided band penalty are unit-tested, and a first steered run exposed and fixed a normalisation bug (three requests returned identical molecules because the weights were uniform). The forward predictor it was used with scored carbonyl present/absent AUC 0.93. | The predictor is **not shipped**, you supply one. There is no recorded end-to-end steering result you can reproduce from this package, and the encoder side (`SpectrumEncoder`) was dropped. |
| **Fingerprint conditioning** (`FingerprintEncoder`, `morgan_condition`) | A cross-attention adapter on 1024-bit Morgan counts lifted Tanimoto similarity by +0.118 to +0.157 over the unconditional baseline at its best weight (1.5). It does no better on trained-on targets than on unseen ones, so capacity is not the limit. | A comparison with a binary-fingerprint control (declined), and RL on it ratchets connectivity but not similarity. The ceiling is the frozen base. |
| **Legacy input-CFG model** (`DeFoGModel` with `cond_dim > 0`) | Kept for checkpoints trained that way. | It is superseded by adapters for every recorded result. |

### What was removed, and why

Features with a null, negative, superseded or untried result were left out rather than shipped with a caveat. The reasons
are on [Migrating from the research fork](../reference/migration.md#what-was-removed).

### Dependencies and platforms

* The lower bounds in `pyproject.toml` were **tested**: the full suite passes with every dependency at exactly its minimum
  on Python 3.10 (torch 2.1, PyG 2.3, Lightning 2.4, numpy 1.23, matplotlib 3.5.3).
* The test suite and the recipes were run on CPU and on single consumer GPUs. Multi-GPU training was not exercised. The
  chained-link training (`PerLinkTimer`) was used for the end-to-end model of the headline comparison, which took two jobs.
* Checkpoints with interior adapters (`interior_ff` / `interior_attn`) or a spectrum encoder will not load: they raise an
  error that says so, rather than loading a different model.

## The earlier numbers

Most numbers on the [Adapters](../concepts/adapters.md), [Steering](../concepts/steering.md) and
[Reinforcement learning](../concepts/rl.md) pages, and the figures quoted above under Noise and under Feynman-Kac
steering and RL, come from the research fork. They used to be collected in two long notes, `docs/results.md` and
`docs/caveats.md`. Those notes are no longer part of the repository tree; their last version is in the repository
history, at commit `a227b62`, and their tables are not reproduced here. Read a number on this site that is not in the
headline comparison as the research fork's measurement under the protocol described above: 100 targets drawn from real
molecules, 10 generations each, pooled error to the target.
