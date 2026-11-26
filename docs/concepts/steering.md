# Steering at sampling time

Everything here acts on a trained model **without retraining it**: Feynman-Kac particle steering, learned
node-count draws, composing several properties, substructure inpainting, and a spectral energy. For the
adapters that supply the conditioning, see [Adapters](adapters.md); for the numbers, [Results and caveats](../about/results-and-caveats.md).

## Feynman-Kac steering

`FeynmanKacSampler` runs a population of `K` particles through the denoising process and, at checkpoints,
**resamples** them: low-weight particles are killed and high-weight ones cloned. The weight of a particle is
`exp(-beta * energy)`, where the energy scores the particle's **predicted clean graph** (decode it, measure
the property, compare with the target). Nothing is differentiated, so the energy can be any black box, and the
proposal can be the base model alone or an adapter composition. The ensemble corrects itself globally: a
particle that happens to head toward the target takes over, which attacks both failure modes of plug-in
guidance (undershooting, and reward hacking within a mode).

![Diagram of Feynman-Kac steering with six candidates that denoise side by side from noise to molecule. Nothing is resampled during the warm-up. At three checkpoints late in the run each candidate gets a weight from how far its predicted clean graph is from the target; low-weight candidates are dropped and high-weight ones are cloned.](../assets/diagrams/feynman-kac.svg)

*Feynman-Kac steering with six candidates. After the warm-up, the heavy candidates are cloned (cyan) and the light ones dropped (✕) at each checkpoint. Weights, parents and molecules are drawn for illustration, not taken from a run.*

```python
from defog_adapt.core import (AdapterComposition, ConditionBranch, FeynmanKacSampler,
                              LearnedPropertyEnergy, StandardisedEnergy)
from defog_adapt.presets import FK_STEERING

energy = StandardisedEnergy([(
    LearnedPropertyEnergy(head, target, domain, atom_encoder, bond_encoder, scale=head_std),
    head_std)])                                   # (energy, scale): divided by scale**2, see below
composition = AdapterComposition([ConditionBranch(adapter, torch.tensor([target]), 1.5)], base=base)
sampler = FeynmanKacSampler(base, energy, composition=composition, **FK_STEERING,
                            sample_steps=500, eta=25.0, omega=0.0, time_distortion="polydec")
graphs = sampler.sample(10, size_dist=size_model, condition=torch.full((10, 1), target))
print(sampler.trace)         # read this: did it engage?
```

`examples/steer_zinc.py` is this, assembled and runnable, with an evaluation mode.

### The energy: always standardise it

`MoleculePropertyEnergy` and `LearnedPropertyEnergy` return a squared error in the property's **own units**.
With a raw energy, `beta` means "per logP unit squared" for one property and "per QED unit squared" for another,
so a `beta` tuned on logP (standard deviation 1.16) pushes about **76 times less** on QED (0.14): the particle
weights stay close to uniform, the resampling gate never fires, and a `beta` sweep reads as "Feynman-Kac does
not help this property" when it never engaged. `StandardisedEnergy` divides by the **variance**, which makes the
energy a squared z-score and `beta` dimensionless, so one value carries from property to property. The recorded
`beta` values are in these units. It also sums several properties (`terms=[(e1, s1, w1), (e2, s2, w2)]`).

The invalid-graph sentinel is already stated in squared standard deviations by both energy classes and
pre-multiplied by `scale ** 2`, so standardising leaves it as the same number (25 by default: five standard
deviations off) for every property. Pass `scale=` to the energy as well as to `StandardisedEnergy`.

### Choosing `beta` (and reading `trace`)

`beta` is the lever; the particle count is not. At `beta` 20 going from 10 to 64 particles moved the logP error
from 0.2299 to 0.2129, an asymptote, while the jump from `beta` 2.5 to 20 was worth 0.083. More particles only
remove finite-sample bias; to lower the asymptote, raise `beta`, at a cost in diversity (logP uniqueness 0.76
at `beta` 20, 0.54 at 80). **Sweep it**, and read `sampler.trace` after each run: one record per checkpoint with

| field | meaning | a starved run looks like |
|---|---|---|
| `ess_over_k` | effective sample size as a fraction of the particles | near 1.0 at every checkpoint |
| `logw_range` | spread of the accumulated log-weights: the steering pressure | near 0 |
| `resampled` | whether it resampled | rarely `True` |

A run that never resamples is the unguided sampler with extra forward passes, and no error metric will tell you.
At `beta` 2.5 logP resampled on 16% of checkpoints (ESS/K 0.75) and Feynman-Kac looked useless; at 20 it
resampled on about 65% (ESS/K about 0.43) and it was the largest lever measured. The sampler also refuses an
`ess_frac` whose threshold `ess_frac * K` is at or below 1, because the effective sample size is never below 1
and it could never resample.

### The schedule, as recorded

`presets.FK_STEERING`: `beta=20`, `warmup_frac=0.6`, `ess_frac=0.5`, `resample_scheme="residual"`, and the
sampler's own `resample_interval` (steps // 8).

* **`warmup_frac`**: no resampling until this fraction of the trajectory. Early on the predicted clean graph is
  unreliable and size-biased: large graphs are transiently invalid or overshoot, so early resampling culls them
  and collapses the sizes toward small "reliable" molecules. Resample late.
* **`ess_frac`**: resample only when the effective sample size drops below this fraction of `K` (adaptive
  resampling preserves diversity). `None` resamples at every checkpoint.
* **`resample_scheme`**: `"residual"` (default), `"systematic"` or `"multinomial"`. All are unbiased and the
  measured differences were inside the noise on both properties, so the default is a choice on theory, not a
  result. The earliest recorded results used `"multinomial"`; pin it to reproduce them, and write the scheme into
  any result you keep.
* **Cost**: `K` forward passes per step, one extra forward per particle at each checkpoint, and an RDKit decode
  of `K` graphs per checkpoint.

### The ruler: a learned head

For a property with a closed form you can score with RDKit (`MoleculePropertyEnergy`). The recorded runs used a
**learned head** (`PropertyHead`, `LearnedPropertyEnergy`), which scored 4 to 18 times tighter than the adapters
it steered, works for properties with no closed form, and is what the particles are scored by because each one is
a *discrete* one-hot graph. Each predicted clean graph is decoded (the validity gate) and re-encoded in the head's
native format before scoring, because feeding the head the model's raw argmax graph (spurious diagonal edges,
off-support classes) makes it mispredict and steers the wrong way. Fit one with
`examples/fit_property_head__zinc.py`, **with the same label convention as the adapter** (see
[adapters.md](adapters.md#what-the-condition-means-the-label-convention)).

### What did not help

Textbook cures for particle impoverishment were tried and are not in the library: rejuvenation (resample-move)
lost to plain `beta` 20 on both error and diversity, beta schedules were dominated, and lower-variance
resampling moved nothing consistently. The reading is that there was no impoverishment to fix: cloned particles
have hundreds of stochastic steps left to diverge, and the lost diversity is the steered distribution being
genuinely narrower. Feynman-Kac and RL fine-tuning also **do not add**: they correct the same error.

## Learned node counts

Draw the graph size from `P(n | target)` instead of the dataset marginal: `fit_size_model(conditions, sizes, ...)`
returns a `LearnedSizeDistribution`, and you pass it as `size_dist=` with the target as `condition=` to any
sampler's `sample`. It is a **bias correction** on the size draw (not a narrowing: the shrink stays near 0.9),
worth -0.0103 logP and -0.0058 QED error on the test split. The fitter reports `gain_nats` over the marginal;
below 0.02 the property carries no information about size and the model is a second thing to keep in sync that
buys nothing. Fit it from the **same rows and label convention** as the head and the adapter, and give it the
base's `max_nodes` as its ceiling so it cannot put mass on a graph the base cannot build.

For several properties, `ComposedSizeDistribution([SizeBranch(model, condition_rows, weight), ...], mode="product")`
multiplies the per-property models over the node-count grid: the same log-ratio algebra the rate matrices use,
recovering 87% of what a joint model would. A composition is built for `n` rows and cannot serve another count.

## Several properties at once

Adapters compose (`AdapterComposition` with several `ConditionBranch`es, `mode="product"` or `"mean"`), size
models compose, and the energies add inside `StandardisedEnergy`. `examples/steer_zinc.py` takes one `--property`
group per property. The best composition mode is pair-dependent (mean won logP + QED, product won logP + TPSA).

## Inpainting: freeze a substructure, grow the rest

`InpaintingSampler` is RePaint-style: at every step the running state is projected onto the constraint at the
**current noise level** (the frozen core is re-noised to the same level as the free region, so the network's
input stays on-distribution), and after the loop the exact core is installed. The guarantee is that the core you
asked for is exactly the first `k` nodes of every sample, with exactly the bonds you specified among them.

```python
from defog_adapt.core import InpaintingSampler, SubgraphConstraint
from defog_adapt.domains import MoleculeDomain

domain = MoleculeDomain(atom_types, bond_decoder)
X, E = domain.encode("c1ccccc1")          # an aromatic SMILES is encoded in Kekule form for a kekulized model
print(domain.core_valence_report(X, E))   # check FIRST: a core with no free valence completes ~0% valid
sampler = InpaintingSampler(base, SubgraphConstraint(X, E), sample_steps=250, eta=50, omega=0.3,   # the recorded
                            time_distortion="polydec", resample=True, n_resample=4)                      # demonstration
graphs = sampler.sample(8, n_free=8)
```

`mode="clamp"` instead of the default `"repaint"` holds the core fixed at every step rather than re-noising it.
`resample=True` adds RePaint's time travel: it periodically jumps back to a noisier time (`jump_length`, by
default about 10% of the steps, every `jump_interval` steps) and re-denoises, `n_resample` times per jump point,
giving the free region repeated chances to harmonise with the core at the boundary. It costs extra forward passes
and helps mainly on hard cores. A core plus `n_free` that exceeds the model's `max_nodes` is refused. The evidence for this feature is thin: see
[Results and caveats](../about/results-and-caveats.md#features-with-thin-evidence).

## A spectral energy

`SpectrumEnergy` is a Feynman-Kac energy over a molecule's **predicted vibrational spectrum**: steer toward a
band here, no band there, or a whole measured spectrum. You supply the forward predictor
(`predictor: list[Mol] -> (n, len(grid)) array of nonnegative intensities`, called once per batch, never
differentiated), which this package does not ship.

```python
from defog_adapt.core import Band, SpectrumEnergy

energy = SpectrumEnergy(domain, predictor, grid, bands=[Band(1710, half_width=30, target=0.05)])
```

The requestable unit is a **band**: a window and how much of the molecule's absorption should fall in it, because
a forward predictor reproduces a whole spectrum only moderately well while placing a functional group's band
very well, and because a spectrum is not a vector of free parameters (peaks are strongly correlated). The penalty
is **one-sided**: "give me a carbonyl" means at least this much absorption there, so a molecule with a very
strong carbonyl has satisfied the request rather than overshot it (`mode="at_least"`, and `"at_most"` for an
absence request). Band shortfalls are normalised by the band's `scale`, for the same reason `StandardisedEnergy`
exists: an un-normalised squared error on a fraction of total absorption lands near 1e-3, the weights stay
uniform, and three different requests return byte-identical molecules. An unreachable request degrades
gracefully (the weights never concentrate, and the closest molecules found are returned). The evidence is thin:
see [Results and caveats](../about/results-and-caveats.md#features-with-thin-evidence).
