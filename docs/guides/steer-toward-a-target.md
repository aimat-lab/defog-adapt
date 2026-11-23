# Steer toward a target

This guide takes a trained base and adapter and samples molecules whose logP is close to a target. You start with the
adapter alone, then add the optional layers one at a time, a size model and Feynman-Kac steering, and at the end you
measure every run the same way. For what the pieces are, see [Concepts](../concepts/index.md).

## What you need

**Required**

- **A base model and an adapter** trained for the property: see [Train an adapter](train-an-adapter.md).

**Optional**

- **A size model**, which draws the number of atoms from `P(n | target)` instead of from the dataset's overall
  distribution. It is a bias correction on the size draw.
- **For Feynman-Kac steering, a property head**, the ruler that scores each candidate. An adapter-only run needs
  neither the head nor Feynman-Kac.
- **A judge**, which measures the decoded molecules. RDKit is what the recorded numbers used, a better model also
  works, and you can leave the check out if you trust the generation.

The `fit_property_head__zinc.py` recipe fits the head and, with `--with-size-model`, the size model. Fit both under
the same label convention as the adapter.

![Diagram of what a steered run needs. In the solid main flow, a proposal, which is a frozen base model with an adapter on it, generates the molecules; the base model and the adapter are both required files. Dashed frames mark what is optional: a size model that draws the number of atoms; a Feynman-Kac add-on in which a property head, the ruler, scores the predicted clean graphs and the candidates are resampled; and a judge that measures the decoded molecules, with RDKit, with a better model or not at all. The classes shown are DeFoGModel, CrossAttentionAdapter, PropertyHead and LearnedSizeDistribution.](../assets/diagrams/steering-roles.svg)

*What a steered run needs. Solid frames are required and dashed frames optional. The ruler steers the run and the judge measures it: the property head is the ruler, never the judge.*

The steps below build the figure from its solid middle outwards: first the proposal, then each dashed part. All the code
is one script. It is shown here in pieces, and in full at the end of the page.

## 1. Describe the molecules

A generated graph is a list of atom and bond classes. The domain turns it into a molecule and back, and it needs RDKit
(the `chem` extra). The recorded base was trained on ZINC250k, so its vocabulary is that reference split's. The order is
not a choice: the model's output channels are indexed by it, so an adapter trained against another order would steer
toward other elements.

```python
--8<-- "docs/guides/steer_toward_a_target.py:domain"
```

`domain` decodes the graphs you generate in step 8 and, if you add Feynman-Kac, scores them in step 6.

## 2. Load the base and the adapter

These are the two required files: one frozen base and one small adapter file per property. `check_compatible` raises if
the adapter does not fit the base, and warns if it was trained on a different one.

```python
--8<-- "docs/guides/steer_toward_a_target.py:load"
```

Nothing in the base changes: the adapter works beside it. An adapter is tied to the base it was trained on, to its
vocabulary for certain and in practice to its weights, so always load the pair that belongs together.

## 3. Tie the adapter to your target

A `ConditionBranch` pairs the adapter with the target you want, here a logP of 4, and a guidance weight, which sets how
hard the adapter pulls. `ADAPTER_WEIGHT` holds the recorded weight of each property. An `AdapterComposition` combines
branches, one per property (see [Going further](#going-further)).

```python
--8<-- "docs/guides/steer_toward_a_target.py:target"
```

The target is a raw property value: give it as logP, not as a z-score, because the adapter standardises it itself.

## 4. Sample with the adapter alone

`AdaptedSampler` runs the base with the adapter acting at every step. `ADAPTER_SAMPLING` holds the recorded sampler
settings.

```python
--8<-- "docs/guides/steer_toward_a_target.py:adapter_run"
```

`graphs` is a list of 10 generated graphs, with their number of atoms drawn from the dataset's overall distribution. They
are graphs, not molecules yet: step 8 decodes them. This run is all that is required. Every step below is optional and
adds a layer.

## 5. Add a size model (optional)

Skip it and the number of atoms is drawn from the dataset's overall distribution, whatever the target. A size model draws
it from `P(n | target)` instead. The `condition` is the raw target, one row per graph; the size model standardises it
itself.

```python
--8<-- "docs/guides/steer_toward_a_target.py:size_run"
```

`graphs_sized` is another list of 10 graphs. The sampler is the same; only the number of atoms is drawn differently.

## 6. Add Feynman-Kac steering (optional)

Skip it and the adapter alone proposes the molecules. Feynman-Kac runs a population of candidates through the denoising
together and, at checkpoints, drops the ones that look far from the target and copies the ones that look close, as the
figure in [Steering at sampling time](../concepts/steering.md) shows. The ruler decides what looks close: it scores each
candidate's predicted clean graph, and the squared error to the target is the energy.

```python
--8<-- "docs/guides/steer_toward_a_target.py:ruler"
```

`LearnedPropertyEnergy` decodes the predicted graph, asks the head for its property and returns the squared error to the
target. `StandardisedEnergy` divides it by the property's variance, so that `beta`, the strength of the steering, means
the same for every property. The head is the second optional file; fit it as described under
[What you need](#what-you-need).

`FeynmanKacSampler` takes the base, the energy and the same `composition`: the adapter still proposes, and the energy
selects. `FK_STEERING` holds the recorded settings, `beta` among them. The 10 you ask for are the 10 candidates, and the
10 you get back.

```python
--8<-- "docs/guides/steer_toward_a_target.py:fk_run"
```

`graphs_fk` is a third list of 10 graphs, and the sampler keeps a `trace` of the run for the next step.

## 7. Check that it engaged (optional)

A `beta` that is too small makes the steering look useless when it never engaged. So after every run, read
`fk_sampler.trace`: one record per checkpoint.

```python
--8<-- "docs/guides/steer_toward_a_target.py:trace"
```

Each line is one record, and these are its fields:

| Field | What it means | A starved run looks like |
|---|---|---|
| `ess_over_k` | The effective sample size, as a fraction of the particles. | Near 1.0 at every checkpoint. |
| `logw_range` | The spread of the accumulated log-weights: the steering pressure. | Near 0. |
| `resampled` | Whether it resampled at that checkpoint. | Rarely `True`. |

A run that never resamples is the unguided sampler with extra forward passes. `beta` is the lever: start from the
recorded `FK_STEERING` value and sweep around it, for example with `**{**FK_STEERING, "beta": 30.0}`. Raising the number
of particles only removes finite-sample bias; to lower the error further, raise `beta`, at a cost in diversity.

## 8. Judge the molecules (optional)

Decode each graph and measure the **true** property of the decoded molecule, beside validity and uniqueness. RDKit is the
judge here, as it was for the recorded numbers. The property head is the steering ruler, not a measurement: do not score
a run with it. `judge` returns the fraction of graphs that decode to a valid molecule, the fraction of the valid ones
that are distinct, and the mean absolute difference between their logP and the target.

```python
--8<-- "docs/guides/steer_toward_a_target.py:judge_fn"
```

Then judge all three runs the same way:

```python
--8<-- "docs/guides/steer_toward_a_target.py:judge_runs"
```

Each line shows these three numbers for one run, with `nan` where no molecule was valid. Feynman-Kac trades diversity
for accuracy, so a better error with a lower uniqueness is the expected trade, not a free gain.

## Going further

- **More than one property.** Give `steer_zinc.py` several `--property` groups, each with its own adapter, head and size
  model. The adapters combine by `--mode`, either `product` or `mean`. Which one is better depends on the pair of
  properties, so try both rather than hard-coding one. Details: [Steering at sampling time](../concepts/steering.md).
- **The command line.** `examples/steer_zinc.py` is this tutorial as a command, with an evaluation mode (`--evaluate`)
  that follows the shape of the protocol the recorded numbers used. For adapter-only sampling pass `--method adapter`.

```bash
python examples/steer_zinc.py --base ckpts/zinc_rl2_seed42/best_model --method fk \
    --property logp --adapter runs/logp_adapter.ckpt --head ckpts/heads/logp_head.ckpt \
    --size ckpts/heads/logp_head_size.ckpt --target 4.0 --n 10
```

## The final script

All the steps in one piece, for you to copy. It loads four files from the folder it runs in. To leave out a layer, delete
its block and the arguments that refer to it.

```python
--8<-- "docs/guides/steer_toward_a_target.py"
```
