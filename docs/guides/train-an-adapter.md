# Train an adapter

This guide trains one adapter, for one property, on top of a base model: logP, on the molecules and the base of
[Train an unconditional base](train-an-unconditional-base.md). It then fits the two small models that Feynman-Kac steering
can use in [Steer toward a target](steer-toward-a-target.md): a size model and a property head. For what an adapter is,
see [Concepts](../concepts/index.md#adapters). Like the base guide, it is small enough to run on one GPU, and
[Going further](#going-further) says what the recorded recipe does at scale.

## What you need

**Required**

- **A trained, unconditional base**, `base.ckpt`: the one from [Train an unconditional base](train-an-unconditional-base.md),
  or any base trained on the same vocabulary. An adapter is tied to the base it is trained on.
- **The molecules**, `molecules.smi`: the file the base was trained on, one SMILES per line.
- **RDKit**: install `defog-adapt[chem]`. A **GPU** is used if there is one.

**Optional**

- **A size model** and **a property head**, fitted at the end from the same labelled graphs. The adapter steers without
  them; they come into play when you steer toward a target.

All the code is one script. It is shown here in pieces, and in full at the end of the page.

## 1. Turn the molecules into graphs

The graphs and the vocabulary are those of the base guide, and they must be the same ones: the adapter reads the same
class numbers as the base. `build_graphs` skips what the vocabulary cannot encode, and `domain` turns a graph back into a
molecule, which the next step needs.

```python
--8<-- "docs/guides/train_an_adapter.py:graphs"
```

## 2. Load the frozen base

The base is never trained here. It is loaded to be frozen, and to tell the adapter its own sizes.

```python
--8<-- "docs/guides/train_an_adapter.py:base"
```

## 3. Label the graphs

An adapter learns from graphs that each carry the value of the property they should be steered to: a *label*.

!!! warning "Decide how the label is computed, before you train anything"
    A graph stores atoms and bonds but no formal charge, so the property of a source SMILES and of the molecule the
    graph represents can differ, by up to 1.6 logP at the low end. Every recorded result labels by the **decoded graph**
    (`property_from="decoded"`). The adapter, the property head and the size model must all use the same convention. If
    they do not, they disagree about what the target means, and the disagreement is largest at the low end of logP.

So the script decodes each graph and measures the molecule it gets. A graph is kept only if it decodes and the base can
build a graph of that size.

```python
--8<-- "docs/guides/train_an_adapter.py:label"
```

`labelled` holds the kept graphs, each with its label in `.cond` as a one-by-one tensor, and `labels` is the column of all
of them. The adapter standardises the target with the mean and standard deviation of `labels` in the next step, and the
size model and the head are fitted from the same labels.

## 4. Build the adapter

`for_base` reads the sizes it needs from the base and stamps the adapter with a fingerprint of that base, so that using the
adapter with the wrong base fails when the sampler is built. `cond_mean` and `cond_std` are the property's statistics in the
label convention you will deploy under: the adapter standardises the raw target with them, so you give it logP and not a
z-score. `name` and `cond_type` are written into the file. The seven settings after them are the recorded architecture
(`ADAPTER_ARCHITECTURE` in `defog_adapt.presets`), written out here.

```python
--8<-- "docs/guides/train_an_adapter.py:build"
```

| Setting | What it does | Recorded |
|---|---|---|
| `hidden` | The width of the adapter's own network, which turns the target and the flow time into a scale, a shift and a gate for each layer of the base. | 256 |
| `time_conditioned` | The adapter also reads the flow time. | `True` |
| `streams` | What it modulates in the base: the nodes (`X`), the edges (`E`) and the global vector (`y`). | `X`, `E`, `y` |
| `cond_fourier` | Fourier bands on the target, which make nearby targets such as 3.5 and 4.2 easier to tell apart. | 3 |
| `xattn_tokens` | Cross-attention: how many tokens the target is turned into, for each atom to look at. | 64 |
| `xattn_dim` | The width of those tokens. | 128 |
| `xattn_heads` | The attention heads of the cross-attention. | 16 |

The architecture does not shrink with the base. It is the recorded one, and it is cheap next to training a base.

## 5. Train it

`AdapterModule` freezes the base, keeps it in evaluation mode (dropout in the base would make the denoising target noisy in
a way that deployment does not), trains exactly the adapter's parameters, and leaves the base's weights out of the
checkpoint. The loss is the base's own denoising loss, so the trained adapter is a conditional denoiser sitting on top of
an unchanged unconditional model. Its four training settings are the recorded ones (`ADAPTER_TRAINING`); the Lightning
`Trainer` applies the gradient clipping.

```python
--8<-- "docs/guides/train_an_adapter.py:train"
```

| Setting | What it does | Recorded |
|---|---|---|
| `max_epochs` | How many passes over the labelled graphs. Twenty was badly under-trained for this architecture; sixty to eighty is saturated. | 80 |
| `batch_size` | Graphs per batch. | 24 |
| `lr` | The learning rate. | 4e-4 |
| `gradient_clip_val` | The largest gradient norm. | 1 |

`adapter.save("logp_adapter")` writes `logp_adapter.ckpt`. It holds the adapter only, and the base is supplied separately
when you load it.

## 6. Check it (optional)

Sample at a low and at a high target with the adapter alone, and compare the mean logP of the valid molecules with the
target. The targets are the 10th and 90th percentiles of your labels, so they lie inside the range the adapter has seen.
The weight and the four sampling settings are the ones explained in [Steer toward a
target](steer-toward-a-target.md#3-tie-the-adapter-to-your-target).

```python
--8<-- "docs/guides/train_an_adapter.py:check"
```

The script prints one line per target: the target, how many of the 100 molecules are valid, and the mean logP of those. If
the mean follows the target, the adapter steers. A base that makes few valid molecules, as a small base does, makes the
means noisy.

## 7. Fit a size model (optional)

Without a size model, the number of atoms of a generated molecule is drawn from the dataset's overall distribution,
whatever the target. A size model draws it from `P(n | target)`, a bias correction on the size draw. It sees only a property
value and an atom count, never a molecule, so it takes seconds on a CPU. `max_size=base.max_nodes` keeps it from putting
mass on a graph that the base cannot build. Fit it from the same labels as the adapter, and check that the pair agrees as
[Adapters](../concepts/adapters.md#what-the-condition-means-the-label-convention) describes.

```python
--8<-- "docs/guides/train_an_adapter.py:size"
```

| Setting | What it does | Recorded |
|---|---|---|
| `hidden` | The width of the size model's network. | 512 |
| `layers` | Its depth. | 2 |
| `epochs` | How many passes over the (label, atom count) pairs. | 200 |
| `lr` | The learning rate. | 1e-3 |
| `batch_size` | Pairs per batch. | 256 |

`metrics` has two numbers to read. `gain_nats` is how much better than the dataset's overall size distribution the model
predicts held-out sizes: near 0 means that the property says nothing about size, and the size model should not be used for
it. `shrink` says what kind of improvement it is: near 0.9 means that conditioning moves the centre of the size
distribution much more than it narrows it.

## 8. Fit a property head (optional)

Feynman-Kac steering scores each candidate with a head: a small network that reads a graph and predicts its property. It
needs no adapter. It is fitted by regression on the same labelled graphs, with some held out so that you can judge it.
`prop_mean` and `prop_std` let it return the property in its own units.

```python
--8<-- "docs/guides/train_an_adapter.py:head"
```

| Setting | What it does | Recorded |
|---|---|---|
| `hid` | The width of the head's network. | 128 |
| `layers` | Its depth. | 3 |
| `epochs` | How many passes over the labelled graphs. | 60 |
| `lr` | The learning rate. | 1e-3 |
| `batch_size` | Graphs per batch. | 32 |

Unlike the other `save` calls, `PropertyHead.save` does not add `.ckpt`: it writes exactly the name you give it. Then judge
the head on the graphs it never saw.

```python
--8<-- "docs/guides/train_an_adapter.py:head_check"
```

The script prints two numbers: the head's mean absolute error on the held-out graphs, in the property's own units, and the
standard deviation of the labels, which is the error of always predicting the mean. A head must be clearly better than
that. If its error is not smaller, it carries no signal, and Feynman-Kac would steer by noise.

## Going further

- **The recorded recipe.** `examples/adapter_training__zinc.py` trains adapters for logP, TPSA, QED and SAScore on
  ZINC250k and evaluates each; `examples/fit_property_head__zinc.py` fits the head and, with `--with-size-model`, the size
  model. The recipes use PyComex (`pip install pycomex`), which caps numpy below 2, so give it an environment of its own.
  See [Recipes](../reference/recipes.md).
- **Keep the recorded recipe.** `ADAPTER_ARCHITECTURE` and `ADAPTER_TRAINING` are one measured recipe. A shortened run is
  under-trained, and changing one setting in isolation does not reproduce it. Two seeds of the same recipe differ by more
  than most differences between architecture settings, so a sweep of single seeds does not rank configurations.
- **Other properties.** Only the label changes: measure TPSA, QED or SAScore instead of logP in step 3, and use the weight
  recorded for it (`ADAPTER_WEIGHT`; QED has 2.0).
- **Tighten it with RL.** `examples/adapter_rl_finetune__zinc.py` fine-tunes a trained adapter by policy gradient. It needs
  the head and the size model of steps 7 and 8.

## The final script

All the steps in one piece, for you to copy. It reads `molecules.smi` and `base.ckpt` from the folder it runs in and writes
`logp_adapter.ckpt`, `logp_head_size.ckpt` and `logp_head.ckpt` there, which with `base.ckpt` are the four files that
[Steer toward a target](steer-toward-a-target.md) loads. To leave out an optional step, delete its block.

```python
--8<-- "docs/guides/train_an_adapter.py"
```
