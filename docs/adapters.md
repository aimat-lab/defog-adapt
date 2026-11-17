# Adapters

A **frozen-base adapter** turns an unconditional DeFoG model into a conditional one without touching the
unconditional model. This page is the reference for what one is, how it is trained and composed, and the
ways it can be wrong without raising an error. Measured results are in [results.md](results.md).

## What it is

An adapter maps a condition `c` (and the flow time `t`) to per-layer `{scale, shift, gate}` modulation of
the base transformer's node (`X`), edge (`E`) and global (`y`) streams: gated FiLM, in the style of
adaLN-Zero. The gate heads are zero-initialised, so **an untrained adapter reproduces the base exactly,
bit for bit**.

Two additions to the FiLM trunk carry the recorded results:

* **Node-to-condition cross-attention** (`xattn_tokens`, `xattn_dim`, `xattn_heads`). Each atom queries the
  condition, instead of every atom receiving the same broadcast correction. Nodes only: edges and the global
  vector keep the FiLM path. This is the half that carries the effect.
* **Fourier bands on the condition** (`cond_fourier`). Without them the trunk sees a scalar property as one
  raw float while the flow time gets a 64-dimensional sinusoidal embedding, so the map from condition to
  modulation learns the smooth part long before the part that separates 3.5 from 4.2. The bandwidth is a
  ceiling, not a knob to raise: three bands keep neighbouring targets correlated (cosine 0.76 for a 0.1 logP
  step) while separating the ends (0.34 for 2.5); by six bands a 0.1 step has already decorrelated the
  features and the mapping is memorisation. Bands are for conditions of dimension 8 or less.

The training objective is the base's **own** denoising cross-entropy with the base frozen, so a trained
adapter is a conditional denoiser `p(x1 | x_t, c)` sitting on top of an unchanged unconditional model.
Three properties follow, and they are the reason adapters exist:

* **Swappable.** One frozen base, one small file per property (about 6.6M parameters for the recorded
  architecture, against 9.6M for the base).
* **Composable.** Two adapters multiply (or average) into a joint condition without a joint model.
* **The unconditional path survives**, so the base can be improved separately (for example by structural RL)
  and the same machinery still applies.

The price is that an adapter is tied to its base: to its vocabulary (class order) for certain, and in practice
to its weights.

## Building and training one

```python
from defog_adapt.core import AdaLNAdapter, AdapterModule, DeFoGModel
from defog_adapt.presets import ADAPTER_ARCHITECTURE

base = DeFoGModel.load("base.ckpt")                       # frozen below; must be unconditional
adapter = AdaLNAdapter.for_base(
    base, cond_dim=1, cond_mean=[mean], cond_std=[std],   # the property's statistics, in the label
    name="logp_adapter", cond_type="logp",                # convention you will deploy under
    **ADAPTER_ARCHITECTURE)
module = AdapterModule(base, adapter, cond_attr="cond", lr=4e-4)   # freezes the base, trains the adapter
trainer.fit(module, loader)        # each graph carries its RAW condition as a (1, 1) tensor in `.cond`
adapter.save("logp_adapter")       # writes logp_adapter.ckpt
```

`for_base` builds the recorded architecture by default (cross-attention 64 by 128 by 16, three Fourier bands
where they are legal) and stamps the adapter with a fingerprint of its base; `ADAPTER_ARCHITECTURE` states the
same thing explicitly. `AdapterModule` freezes the base, keeps it in eval mode (dropout in the base would make
the denoising target noisy in a way deployment does not), optimises exactly the adapter's parameters, and
**drops the base's weights from the Lightning checkpoint** (a second copy of the base is ballast, and the base
is supplied separately at load time). `examples/adapter_training__zinc.py` is the full recipe.

**The recorded recipe is one recipe.** 80 epochs, learning rate 4e-4, batch 24, cross-attention 64 tokens by
width 128 by 16 heads, three Fourier bands (`defog_adapt.presets.ADAPTER_TRAINING` and
`ADAPTER_ARCHITECTURE`). Twenty epochs was badly under-trained for this architecture (0.4299 against 0.3250
at 80), and changing one of the three in isolation does not reproduce it.

**Judge an architecture sweep against the seed noise.** Two seeds of the identical recipe differ by 0.058 in
logP error, which is larger than almost every difference between capacity settings. A sweep of single seeds
does not rank configurations.

## What the condition means: the label convention

A graph stores atoms and bonds but **not formal charge or stereochemistry**, so the property of the source
SMILES and the property of the molecule the graph actually represents differ, by up to 1.6 logP at the low
end (charged amines and carboxylates are what make a molecule low-logP). `decoded` labels the graph;
`source` labels the SMILES. Every recorded result used `decoded`, and the adapter, its property head, its
size model and any RL targets must all use the same one. A mismatch is invisible in the training loss:
training converges and the model is quietly wrong at the end of the range where the lost charge bites.

`examples/_common.label_graphs` implements both. Three checks guard the common mismatches, and they exist
because each one cost time:

```python
adapter.check_compatible(base)                 # dims and layer count match; warns on a different base
size_model.check_compatible(adapter, tol=...)  # same condition width, mean and std
# examples/steer_zinc.pair_check: a head's mean within 1% of a standard deviation of the adapter's
```

Use a scale-free tolerance (1% of a standard deviation, not an absolute 1e-4): parts fit on slightly different
subsets differ by a few parts in ten thousand of a standard deviation, which is noise, while the failure you
are guarding against (source against decoded labels) is about a third of one.

`check_compatible(base)` also compares the fingerprint the adapter stored (the sum of one fixed weight tensor)
with the base's and **warns**, rather than raises, on a mismatch. It is a cheap guard against hot-swapping an
adapter onto an unrelated base with matching dimensions; it uses a loose tolerance and does not distinguish a
base from its own fine-tune, so it is not a substitute for knowing which base an adapter was trained on.

## Scalar and vector conditions

* **A scalar property** (logP, TPSA, QED, SAScore): raw value in, z-scored by the adapter's own `cond_mean`
  and `cond_std`, Fourier bands on.
* **A high-dimensional condition** such as a fingerprint: pass a `cond_encoder` (the library ships
  `FingerprintEncoder`, a residual MLP, registered as kind `"mlp"`) and leave Fourier bands off. The adapter
  refuses `cond_fourier` together with a `cond_encoder`: Fourier features are a result about low-dimensional
  inputs. `defog_adapt.domains.molecule.morgan_condition` builds the 1024-bit count vector the recorded
  fingerprint adapter used (`log1p` of the counts).

## Sampling with an adapter

```python
from defog_adapt.core import AdapterComposition, AdaptedSampler, ConditionBranch
import torch

composition = AdapterComposition(
    [ConditionBranch(adapter, torch.tensor([4.0]), weight=1.5)],   # target 4.0 at guidance weight 1.5
    base=base, mode="product")
graphs = AdaptedSampler(base, composition, sample_steps=500, eta=25.0, omega=0.0,
                        time_distortion="polydec").sample(10, size_dist=..., condition=...)
```

* **The blend is in probability space.** Classifier-free guidance combines the conditional and unconditional
  clean-graph marginals and the sampler draws once from the guided distribution. (Blending rate matrices
  instead drew a clean graph from each, built a rate matrix from each and took their geometric mean; above
  weight 1 it was numerically broken, so "weight 1 always wins" was an artefact of it.)
* **`weight` is per branch and per property.** The measured optima are 1.5 for logP, TPSA and SAScore and 2.0
  for QED (`presets.ADAPTER_WEIGHT`). Validity falls away past about 2 to 2.5, driven by disconnected
  fragments. Reporting or fine-tuning at another property's weight is how a real effect gets missed.
  `ConditionBranch.weight` defaults to 2.0.
* **Several branches** combine as `mode="product"` (sum the branches' log-ratios to the unconditional model)
  or `mode="mean"` (average them). The best mode is pair-dependent (mean won logP + QED, product won logP +
  TPSA), so do not hard-code one.
* **Draw the node count from `P(n | target)`**, not the dataset marginal, when you have a learned size
  model: pass `size_dist=` and `condition=` to `sample`. A size model composes across branches by product
  (`ComposedSizeDistribution`), and a composition built for `n` rows cannot serve a different count.
  Forgetting to pass them does not raise: it silently falls back to the marginal and scores a different
  policy than the one you meant.
* **For Feynman-Kac steering**, pass the composition to `FeynmanKacSampler(..., composition=composition)`.
  See [steering.md](steering.md).

## File format, compatibility and trust

`adapter.save(path)` writes `{"state_dict": ..., "config": ...}` to `path + ".ckpt"`, and
`AdaLNAdapter.load(path)` rebuilds the architecture from the config and loads the weights (including the
`cond_mean` and `cond_std` buffers). Loading is strict, and **a config that asks for something this package
does not have raises instead of loading a different model**:

* checkpoints with `interior_ff` or `interior_attn` enabled (interior conditioning, superseded by node
  cross-attention) raise, naming the key; a config that merely records them as `False` loads normally;
* checkpoints with the spectrum condition encoder (`spectrum_cnn`) raise;
* an unknown `cond_encoder` kind raises, listing the known ones.

Load such adapters with the research package they were trained in.

**Checkpoints are pickles.** `DeFoGModel.load`, `AdaLNAdapter.load`, `PropertyHead.load` and
`LearnedSizeDistribution.load` call `torch.load(..., weights_only=False)`, because the files carry
configuration dictionaries and hyperparameters. Unpickling executes code, so **only load checkpoints you
trust**, exactly as you would for any `.pt` file from the internet.

## Where adapters stop

An adapter does not widen what the frozen base can reach: the low end of logP, where steering was weakest, is
partly the base's reach and partly a wrong atom count that a learned size model recovers. See
[caveats.md](caveats.md).
