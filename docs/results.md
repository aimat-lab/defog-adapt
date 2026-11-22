# Results

What was measured, under which protocol, and how much to trust it. **Read [caveats.md](caveats.md) with
this page**: it says what the numbers do not support, and a number quoted without its protocol is not a
result.

Every figure below was measured in the research fork this package was extracted from, with its own
harness, and has not been reproduced end to end here; the exceptions are sections 8 and 9. No weights ship
with 0.1, and `examples/` holds the recipes that produce them.

## Protocols

**E2 (conditional generation).** The FreeGress ZINC250k protocol: **100 target values** drawn from real
validation molecules, **10 generations per target**, and the error is the **pooled mean absolute
difference** between each valid generated molecule's property and its target, with the property read off
the *decoded* graph. Validity is the fraction of generations that decode to a valid molecule. All
conditional numbers (the fingerprint evaluation excepted, which has its own protocol below) use 500
sampling steps, eta 25, omega 0, polydec time distortion, probability-space guidance blending at each
property's own best weight (logP, TPSA, SAScore 1.5; QED 2.0), and a seed of 42. The run-to-run **noise floor of a 100-target error is about 0.011**, measured three times on an
identical configuration; differences smaller than that are not differences.

**Unconditional (E1).** Test-split sampling at 500 steps, eta 25, omega 0, n = 10,000 per seed.

"Validation" numbers were used to choose configurations; "test" numbers were read once for an
already-chosen configuration.

## 1. Unconditional base models

**ZINC250k**, kekulized, `E1_MODEL`, four seeds, test split:

| metric | mean ± sd |
|---|---|
| validity (relaxed) | 0.9854 ± 0.0049 |
| disconnected | 0.0615 ± 0.0219 |
| wonky rings (a ring outside 3 to 8) | 0.0356 ± 0.0129 |
| combined "sanity" (valid, connected, rings ok) | 0.9504 |

**MOSES: the representation is the result.** An AROMATIC bond class is a promise about a whole ring
system that RDKit checks by kekulizing, and a per-edge model cannot keep it. On the aromatic base 118 of
120 hard validity failures were kekulization errors and exactly one was a valence error. Dropping the
class (`kekulized_v2`: 7 atom types, 3 bond types, lossless on 1,579,663 training molecules) gave:

| | aromatic | kekulized (4 seeds) |
|---|---|---|
| validity | 0.8843 | **0.9907 ± 0.0035** |
| FCD | 1.478 | **1.126** |
| kekulization failures | 0.1084 | **0** |

Nothing migrated to another failure mode (valence errors 0.0020 to 0.0031). FCD scales roughly as 1/n,
so the two columns are comparable to each other and not to other tables.

## 2. Structural RL on the base (ZINC)

`examples/rl_sanity__zinc.py`, two rounds, one lineage, test split, n = 10,000:

| metric | E1 base (seed 42) | after two rounds |
|---|---|---|
| validity (relaxed) | 0.9929 | **0.9959** |
| disconnected | 0.0267 | **0.0180** |
| wonky rings | 0.0173 | **0.0133** |
| FCD (lower is better) | 1.401 | **1.346** |
| scaffold similarity (higher is better) | **0.6035** | 0.5464 |

The scaffold similarity cost is front-loaded: round one paid 0.0625 of it, and round two recovered some.
A third round moved nothing outside the noise (nine improvements, seven regressions). At n = 2,048
"disconnected" carries a spread of about 0.010, larger than most per-round effects, so use n = 10,000.
On this lineage the sanity gain did not come at the cost of FCD; the same reward on an aromatic MOSES
base did (validity 0.885 to 0.939 while FCD went 0.863 to 1.706, four of four seeds).

## 3. Adapters

### The architecture, on logP

Validation, E2, probability-space blending, each arm at its own best weight:

| arm | error | validity |
|---|---|---|
| FiLM adapter, 20 epochs (the original) | 0.5420 | 0.982 |
| cross-attention + Fourier bands, 20 epochs | 0.4299 | 0.980 |
| same, 40 epochs | 0.3596 | 0.973 |
| same, 60 epochs | 0.3501 | 0.977 |
| **64 tokens / width 128 / 16 heads, 80 epochs** | **0.3250** | 0.982 |

Attribution at 60 epochs: cross-attention alone -0.153 (80% of the effect), Fourier bands alone -0.024,
together -0.192. Epochs 20 to 40 bought -0.070 and 40 to 60 only -0.0095, so 60 to 80 is the saturated
region. The adapter now matches the full conditional model it was losing to. **Capacity is not a
trend you can read from single seeds**: two seeds of the identical recipe differ by 0.058 (0.3068 and
0.3646), which is larger than almost every difference in the capacity sweep (4 to 128 tokens, widths 64
to 256, 4 to 16 heads).

### Four properties

The best architecture, validity at least 0.97 throughout. "Adapter" is on its own; the last column adds a
learned size model and Feynman-Kac at the **starved beta of 2.5** (see section 5 for the corrected
reading), so it is a lower bound on what FK gives.

| property | best weight | adapter | + FK (beta 2.5) + size | adapter error / std |
|---|---|---|---|---|
| logP | 1.5 | 0.3250 | 0.2988 | 0.280 |
| QED | 2.0 | 0.0867 | 0.0666 | 0.667 |
| TPSA | 1.5 | 4.7045 | 4.5598 | 0.204 |
| SAScore | 1.5 | 0.4233 | 0.3691 | 0.756 |

Read the **error / standard deviation** column, not the raw error: TPSA is the best-steered property
despite the largest number. For scale, the published best on logP is 0.16 and on QED 0.04 (FreeGress, at
81.2% validity on logP), against validity of 0.97 to 0.99 here; these are not same-condition comparisons.
The first adapter's logP error was 0.5420.

### Probability-space blending

The classifier-free-guidance blend is done on the clean-graph marginals. Blending rate matrices instead
was numerically broken above weight 1, which is why "weight 1 always wins" was an artefact of the
blend space. (E2 logP, validation, FiLM adapter, 500 steps, eta 25, seed 42.)

| blend | weight | error | validity |
|---|---|---|---|
| rate (historical) | 1.0 | 0.6453 | 0.991 |
| rate | 2.0 | 5.5948 | 0.526 |
| probability | 1.0 | 0.6410 | 0.997 |
| **probability** | **2.0** | **0.5420** | 0.982 |
| probability | 3.0 | 0.5943 | 0.898 |
| probability | 4.0 | 0.6717 | 0.466 |

-16.0% (0.6453 to 0.5420), better on 99 of 99 targets (p = 5.7e-18), and about 6% faster. At weight 1 it
is a no-op, as the theory requires (paired p = 0.72). Only this blend is in the library.

### Joint properties

Two adapters compose by product or mean of their clean-graph marginals, and two learned size models by
product. FreeGress's "Total" is the unweighted mean of the per-property errors, so that is what is
reported. 100 targets, K = 10, composed sizes, FK.

| pair | mode | Total | validity | uniqueness |
|---|---|---|---|---|
| logP + QED | product | 0.1992 | 0.936 | 0.740 |
| **logP + QED** | **mean** | **0.1901** | **0.994** | 0.683 |
| logP + TPSA | product | **2.9817** | | |
| logP + TPSA | mean | 3.3544 | | |

The best mode is **pair-dependent**, so do not hard-code one. The published best joint row is a Total of
about 0.12 at 80.7% validity; ours is higher in absolute terms, but jointness cost 0.0074, which is 0.37 of
the 0.0200 it cost them, at 99.4% validity. Uniqueness fell while validity rose; it is a ratio over valid
molecules, so admitting more valid ones can lower it without collapse (distinct molecules per ten were
about unchanged, 6.93 to 6.79).

### Fingerprints

`FingerprintEncoder` and `morgan_condition`: a cross-attention adapter on 1024-bit Morgan counts. Tanimoto
similarity to the target at the best weight (1.5), 22 targets by 64 molecules:

| split | unconditional | steered | lift |
|---|---|---|---|
| held-out, original (6 targets) | 0.1287 | 0.2861 | **+0.157** |
| held-out, fresh (10 targets) | 0.1362 | 0.2540 | **+0.118** |
| trained-on (6 targets) | 0.1251 | 0.2470 | +0.122 |

It does no better on molecules it trained on than on ones it never saw (+0.122 against +0.118), so the
adapter is not overfitting and capacity is not what limits it; the frozen base is. Validity holds to
weight 1.5 and collapses to 0.64 to 0.70 by 3.0. The run-to-run noise of a single point is about 0.013.
Pinning the generated size to the target's heavy-atom count was worth another +0.03 to +0.05, more than
counts (+0.014) or a wider path, and the original adapter largely ignored the target's size
(correlation -0.13).

## 4. Learned size model

`P(n | target)` replaces the dataset's node-count marginal. Fit on the 219,568-molecule ZINC train split:

| property | correlation with size | gain over the marginal (nats) | shrink |
|---|---|---|---|
| logP | +0.398 | 0.098 | 0.910 |
| QED | -0.321 | 0.165 | 0.945 |
| TPSA | +0.457 | 0.127 | 0.886 |

The expected size runs from 20.2 to 26.7 heavy atoms across logP deciles against a marginal of 23.2. A
shrink near 0.9 means it is a **bias correction on the size draw, not a variance reduction**. On the
test split it was worth -0.0103 logP error and -0.0058 QED error, and on QED roughly half of the total
gain in the RL comparison (section 6). Composing per-property size models by product recovers 87% of what
a joint size model would; the product beat the mean (KL to the joint 0.0371 against 0.0660). A size model
that gains under 0.02 nats is not worth shipping.

## 5. Feynman-Kac steering

Validation, logP, 100 targets, K = 10, learned sizes:

| arm | validation | test |
|---|---|---|
| adapter + marginal sizes | | 0.3014 |
| adapter + learned sizes | 0.3132 | 0.2911 |
| + FK, beta 2.5 | 0.2988 | 0.3013 |
| **+ FK, beta 20** | **0.2299** | **0.2169** |

At beta 20 the gain is +0.0742 on test (6.7 times the noise floor), validity 0.992, **uniqueness 0.739**
on test (0.764 on validation).
The resample rate was 65% on test and 64% on validation, with the effective sample size at 0.432 of the
particles on both, so the mechanism transfers. The published best on logP is 0.16 at 81.2% validity.

**The first FK result was wrong, and the reason matters.** At beta 2.5 the particle system was starved:
logP resampled on only 16% of checkpoints (QED, where FK "appeared to work", on 54%), and at 500 steps FK
gave back exactly what the size model won. beta is dimensionless only because the energy is divided by the
property's variance (`StandardisedEnergy`); a raw energy gives it a different meaning for every property.

- **beta, not particles.** At beta 20, K = 10, 16, 32, 64 gave logP error 0.2299, 0.2213, 0.2151, 0.2129,
  an asymptote near 0.212 (QED at beta 8: 0.0539, 0.0561, 0.0520, 0.0507). Raising the particle count only
  removes finite-sample bias; to lower the asymptote, raise beta. The measured split: beta was worth
  0.083, a 6.4-fold increase in K worth 0.017.
- **The diversity cost.** logP uniqueness is 0.764 at beta 20 and 0.539 at beta 80, and the number of
  distinct molecules still rises with K (7.5 to 39.6) rather than saturating.
- **Two textbook cures did nothing.** Rejuvenation lost to plain beta 20 on both axes at once (jump 5 to
  50 ran 0.2587 / uniqueness 0.545 to 0.3199 / 0.748), and lower-variance resampling moved nothing
  consistently between logP and QED.
- **Draw 64 and keep the 10 closest** is a different, order-statistic measurement: the no-FK control at
  64 particles scores 0.0408 on it against FK's 0.0347, so FK is worth 0.006 of that 0.193 improvement.
- **At the starved beta 2.5**, FK's gain grew as steps fell (500: +0.0001, 250: +0.0017, 100: +0.0156,
  50: +0.0284; test, logP). That dependence was **not re-measured at beta 20**.

## 6. RL on adapters

Head reward, rollouts and scoring at the adapter's own best weight with the learned size draw, four seeds
by six hours each. E2, 100 validation targets by 10, target-paired:

| arm | logP | QED |
|---|---|---|
| pre-RL, marginal sizes | 0.3250 | 0.0867 |
| pre-RL, learned sizes | 0.3132 (p = 0.27) | 0.0782 (p = 0.012) |
| **RL, best seed** | **0.2807** (p = 0.019) | **0.0698** (p = 0.013) |
| RL, second seed | 0.2847 (p = 0.028) | 0.0700 (p = 0.008) |
| seeds differ? | p = 0.95, no | p = 0.89, no |

A total of **-13.6% on logP and -19.5% on QED**, with validity rising (0.982 to 0.989 and 0.979 to 0.996)
and uniqueness unchanged at 1.0; disconnected fragments on QED fell from 12.5% to 3.5%. Both pre-RL rows
reproduced their published numbers exactly, which is what makes the rest readable. The size model is
worth half of the QED gain and almost none of the logP gain, which is why the marginal-size control row is
not optional.

**RL does not add to Feynman-Kac** (validation):

| logP | error | QED | error |
|---|---|---|---|
| adapter + learned sizes | 0.3132 | | 0.0782 |
| + RL | 0.2807 | | 0.0698 |
| + FK, beta 20 | 0.2299 | | 0.0492 |
| + RL + FK | **0.2157** | | **0.0466** |

The best point estimates of all, but the RL increment on top of FK is not significant: logP -0.0084 and
-0.0146 (p = 0.66, 0.32), QED -0.0028 and -0.0010 (p = 0.69, 0.69), where the same RL was worth -0.0324 at
p = 0.019 without FK. Read it as redundancy, not failure: both correct the same error.

Earlier head-reward runs, on a different (interior-conditioned) adapter that this package does not
include, tightened the true RDKit property on all four properties (logP 0.58 to 0.50, TPSA 9.67 to 7.69,
QED 0.165 to 0.130 over two rounds, SAScore 0.545 to 0.444 from round one), the head tracking the truth
throughout. SAScore's second round regressed on every seed and the head said so, which is why round one was
kept; and a later 100-target check of a QED second round found a significant regression. One round is the
useful depth. See [caveats.md](caveats.md).

## 7. Inpainting and the spectrum energy

Thin evidence, included deliberately: see [caveats.md](caveats.md#features-with-thin-evidence). Inpainting
with a benzene core reached about 84% validity and about 99% core preservation on an unconditional model
this package does not ship. The spectrum energy has no end-to-end steering result that can be reproduced
from this package.

## 8. Re-measured here

Everything above is the research fork's. To check that the *assembly* in this repository, not just its
parts, reproduces the central steering result, `examples/steer_zinc.py --evaluate` was run on logP with real
checkpoints: the E1-lineage base, the 80-epoch cross-attention adapter, its head and its size model, at the
recorded sampling configuration (500 steps, eta 25, polydec, guidance weight 1.5) and, for the Feynman-Kac
arm, `beta` 20 with the `FK_STEERING` schedule and 10 particles. The three arms share **40 targets drawn from
the validation split** (a seeded permutation, the first 40 whose property decodes) and a seed of 42, so the
differences between them are paired.

| arm | logP error | validity | uniqueness | recorded error (100 targets) |
|---|---|---|---|---|
| adapter, marginal sizes | 0.2747 | 0.985 | 1.000 | 0.3250 |
| adapter, learned sizes | 0.2905 | 0.998 | 1.000 | 0.3132 |
| adapter, learned sizes, **Feynman-Kac** | **0.2084** | 0.990 | 0.838 | 0.2299 |

* **The Feynman-Kac effect reproduces closely.** Against the same arm without it the error fell by 0.0821
  (-28.3%); the recorded value is -0.0833 (-26.6%). Validity stayed at 0.99, and uniqueness fell as the
  diversity cost predicts (0.838 here against 0.764 recorded; this script pools uniqueness over all
  generations, and the target sets differ).
* **The absolute levels are lower than the recorded ones**, by 0.02 to 0.05. That is the target set: these are
  a different and smaller sample of real molecules than the 100 the original harness used, and the
  noise floor of a 100-target error (0.011) grows to roughly 0.017 at 40. It is a consistency check, not a
  reproduction of the table.
* **Learned against marginal sizes came out the other way** (+0.0158 here, -0.0127 recorded). Both are about
  one noise floor and neither is significant; the recorded one was not either (p = 0.27). Do not read a size
  effect for logP from either.
* It is one seed, on one consumer GPU (about 50 minutes per arm), and checks only the logP path.

```bash
python examples/steer_zinc.py --base BASE --property logp --adapter ADAPTER --head HEAD \
    --size SIZE --evaluate 40 --n 10 --steps 500 --eta 25 --seed 42 --method fk --beta 20
```


## 9. The five approaches, trained and measured here

Sections 1 to 7 are the research fork's numbers. This section is the second exception, beside section 8: every model in
it was trained with the recipes in `examples/` and measured with `examples/compare_arms__zinc.py`, in this repository.
It is the comparison behind the figure in the README, and the data behind that figure is
[`assets/results_logp.json`](assets/results_logp.json).

ZINC250k, logP, **held-out test split**, 100 targets × 10 generations, mean ± sd over three seeds. Lower error is better.

| approach | what it is | setting | logP error | validity | uniqueness |
|---|---|---|---|---|---|
| unconditional base | the adapter base checkpoint as it is: nothing steers it | - | **1.2602** ± 0.0260 | 0.996 | 1.000 |
| end-to-end conditional | the E1 recipe trained end to end on logP, 300 epochs | guidance 1.0 | **0.1810** ± 0.0058 | 0.991 | 1.000 |
| FiLM adapter | frozen base + adapter with FiLM only, 80 epochs | guidance 2.0 | **0.4456** ± 0.0201 | 0.985 | 1.000 |
| cross-attention adapter | frozen base + adapter with node-to-condition cross-attention and Fourier bands, 80 epochs | guidance 2.0 | **0.2721** ± 0.0071 | 0.983 | 1.000 |
| + Feynman-Kac | the cross-attention adapter, with Feynman-Kac particle steering on top | guidance 2.0, beta 20, 10 particles | **0.2161** ± 0.0024 | 0.990 | 0.790 |

The unconditional base is the checkpoint described in
[caveats.md](caveats.md#the-adapter-base-is-a-re-packaged-checkpoint), and it is the same frozen model under all three
adapter rows.

**Reading it.**

* The model trained end to end on logP is the most accurate, on every seed. Steering a frozen base comes close:
  Feynman-Kac is 0.0351 ± 0.0078 behind it and the adapter alone 0.0911 ± 0.0091 (paired per seed, behind on all three).
* Measured from the unconditional base to the end-to-end model, the cross-attention adapter closes about 92% of the
  distance and Feynman-Kac about 97%; the FiLM adapter, at the same training budget, about 76%.
* The cross-attention adapter beats the FiLM adapter by 0.1736 ± 0.0198 (on all three seeds), and Feynman-Kac improves
  the same adapter by 0.0560 ± 0.0089 (on all three seeds).
* Validity is 0.983 to 0.991 for every conditional approach (0.996 for the base). Uniqueness is 1.000 everywhere except
  under Feynman-Kac, where it is 0.790: about one valid generation in five repeats another.
* What each cost, as wall-clock on one shared 24 GB GPU: the end-to-end model 36 h 23 min, the cross-attention adapter
  13 h 41 min, the FiLM adapter 12 h 04 min, and the property head with the size model 53 min. The base the adapters
  share was trained once, earlier, and is not counted.

### Protocol

* **Targets.** 100 per seed, drawn from the real molecules of the held-out test split (their *decoded* logP). Seeds 42,
  43 and 44 each draw their own 100, and all five approaches see the same ones within a seed, so the differences between
  approaches are paired.
* **Error.** As in the other conditional sections: the pooled mean absolute difference between each valid generation's
  decoded logP and its target, over 10 generations per target (Feynman-Kac: 10 particles). Validity is the fraction of
  generations that decode to a valid molecule; uniqueness is the fraction of the valid ones that are distinct, pooled over
  all generations.
* **Sampling.** 500 steps, eta 25, omega 0, polydec time distortion; Feynman-Kac with `beta` 20 and the `FK_STEERING`
  schedule, scored by a learned property head.
* **Node counts.** The four conditional approaches draw them from the learned `P(n | logP)`; the unconditional base cannot,
  and draws from the dataset's marginal.
* **Settings were chosen on validation, once, before any test number was read.** The rule was written down in advance:
  per approach, sweep the guidance weight on the validation split (seed 42, 100 targets × 10 generations), take the lowest
  error among the weights whose validity is at least 0.97, and where another candidate is within the 0.011 run-to-run noise
  floor of the best, prefer the recipe's own checkpoint and then the lower weight. Feynman-Kac uses the cross-attention
  adapter's weight. For the end-to-end model two checkpoints were candidates: the one with the lowest validation loss (the
  recipe's own choice, epoch 278) and the final epoch (300). They tied within the noise floor and the recipe's own won.
  The test split was then read once.

| approach | checkpoint | guidance weight | logP error | validity | outcome |
|---|---|---|---|---|---|
| trained end to end | lowest validation loss (epoch 278) | 1.0 | 0.1869 | 0.992 | **chosen** |
| trained end to end | lowest validation loss (epoch 278) | 1.5 | 0.1897 | 0.948 | excluded: validity below 0.97 |
| trained end to end | lowest validation loss (epoch 278) | 2.0 | 0.6892 | 0.331 | excluded: validity below 0.97 |
| trained end to end | final epoch (300) | 1.0 | 0.1900 | 0.992 |  |
| trained end to end | final epoch (300) | 1.5 | 0.1928 | 0.933 | excluded: validity below 0.97 |
| trained end to end | final epoch (300) | 2.0 | 0.7260 | 0.305 | excluded: validity below 0.97 |
| FiLM adapter | - | 1.0 | 0.5550 | 0.997 |  |
| FiLM adapter | - | 1.5 | 0.4724 | 0.991 |  |
| FiLM adapter | - | 2.0 | 0.4410 | 0.983 | **chosen** |
| FiLM adapter | - | 2.5 | 0.4592 | 0.962 | excluded: validity below 0.97 |
| cross-attention adapter | - | 1.0 | 0.3603 | 0.991 |  |
| cross-attention adapter | - | 1.5 | 0.3057 | 0.992 |  |
| cross-attention adapter | - | 2.0 | 0.2762 | 0.987 | **chosen** |
| cross-attention adapter | - | 2.5 | 0.2711 | 0.961 | excluded: validity below 0.97 |
| unconditional base | - | - | 1.3688 | 0.998 | reference |
| + Feynman-Kac (beta 20) | - | 2.0 | 0.2095 | 0.995 | reference, at the cross-attention weight |

Weight 2.5 would have had the lowest raw error for the cross-attention adapter (0.2711), but its validity (0.961) fails the
rule, and the difference from 2.0 is inside the noise floor in any case.

### How each approach was made

* **Unconditional base.** The adapter base checkpoint: the E1 recipe followed by two rounds of structural RL, sampled with
  nothing steering it. It is the do-nothing reference.
* **End to end.** `training__e1.py --CONDITION "'logp'"`: the E1 recipe with every setting unchanged and a logP condition
  added (decoded-graph labels, standardised by the training graphs' mean 2.8251 and standard deviation 1.1581, with the
  condition dropped on 10% of training steps). 300 epochs. The recorded batch of 256 does not fit a 24 GB GPU (its first
  step needs more than 23 GB), so it is accumulated from two micro-batches of 128 (`MICRO_BATCH_SIZE`), with the EMA decay
  adjusted to stay at 0.9999 per optimizer step. The run was chained over two jobs, and the checkpoint with the lowest
  validation loss is the one measured. **No structural-RL clean-up followed**, unlike the base.
* **FiLM adapter.** `adapter_training__zinc.py --XATTN_TOKENS 0 --COND_FOURIER 0`: the FiLM-only architecture, 2.75 M
  parameters, trained as long as the cross-attention adapter (80 epochs, batch 24, learning rate 4e-4).
* **Cross-attention adapter.** The same recipe with its defaults: node-to-condition cross-attention and three Fourier bands,
  6.64 M parameters. **+ Feynman-Kac** adds the particle steering on top of this adapter.
* **Property head and size model.** `fit_property_head__zinc.py` with its defaults, on the decoded-label convention of the
  adapters: the head's held-out error is 0.0467 (correlation 0.998, 0.040 of the property's standard deviation) and the
  size model gains 0.0927 nats over the marginal (shrink 0.912).
* Every model was trained once, with seed 42.

### What these numbers do and do not show

* **The sd is over sampling seeds, not over trainings.** Each approach was trained once, so the standard deviations say how
  much the *measurement* moves with the target set and the sampling noise, not how much the *model* would move. Section 3
  recorded 0.058 between two trainings of one adapter recipe. A gap between two different trained models of about that size
  or less, such as Feynman-Kac against the end-to-end model (0.035), is not an ordering this section can establish; larger
  gaps (the adapters against the base, FiLM against cross-attention, the end-to-end model against the adapter alone) exceed
  it, though each still comes from one training. Feynman-Kac against the adapter it steers is a paired difference within one
  trained model and does not carry that noise.
* **The lineages differ.** The base, and so every adapter, had two structural-RL clean-up rounds after its E1 training; the
  end-to-end model had none. Its validity (0.991) is nonetheless as high as the adapters', but the comparison is not
  like for like in how the models were made.
* **The end-to-end model was not guided.** The library's guidance for such a model blends rate matrices, which breaks above
  weight 1 (on validation, weight 2.0 gave an error of 0.6892 and a validity of 0.331), so it was measured at weight 1.0. The
  adapters use probability-space blending and are measured at 2.0. A guided end-to-end model is not measured here, and could
  score lower.
* **The FiLM adapter here is an equal-budget baseline.** It was trained as long as the cross-attention adapter (80
  epochs), not for the 20 of section 3, so the gap between them here is the architecture alone; the step in section 3 also
  contains the difference in training length.
* **One hardware and software setup**, one frozen base and one property. GPU sampling is not bit-reproducible, so a re-run
  agrees within the noise, not exactly.

### Provenance and reproducing it

Trained and measured on 24 GB consumer GPUs (RTX 4090), under Python 3.12, PyTorch 2.13.0 (CUDA 13.0), PyTorch Geometric
2.8.1, PyTorch Lightning 2.6.6, RDKit 2026.9.2 and numpy 2.5.5. The adapters, the head and the size model were trained at
commit `c6b64a0`, the end-to-end model at `7f961db`, and every measurement was made at `c44e6fc`. Between them the code
changed by the `MICRO_BATCH_SIZE` option and by renaming `AdaLNAdapter` to `CrossAttentionAdapter`, which is the same
class (the parity gate matches the research outputs bit for bit after the rename).

```bash
# the unconditional base is the adapter base checkpoint (see caveats.md); everything else is trained once
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
