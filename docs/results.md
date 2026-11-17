# Results

What was measured, under which protocol, and how much to trust it. **Read [caveats.md](caveats.md) with
this page**: it says what the numbers do not support, and a number quoted without its protocol is not a
result.

Every figure below was measured in the research fork this package was extracted from, with its own
harness, and has not been reproduced end to end here; the exception is the last section. No weights ship
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
