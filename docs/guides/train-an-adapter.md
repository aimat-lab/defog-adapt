# Train an adapter

This guide trains one adapter, for one property, on top of a trained base model. The recipes train adapters for logP,
TPSA, QED and SAScore, on ZINC250k. For what an adapter is, see [Concepts](../concepts/index.md#adapters).

## Before you start

You need three things, and none of them ships with the package:

- **A trained, unconditional base model.** Train one with the `training__e1.py` recipe, then clean it up with two
  rounds of `rl_sanity__zinc.py`. The recorded adapters were trained on a base that had had exactly these two rounds.
  Use kekulized molecules: the aromatic bond class loses information.
- **A GPU and the dataset.** The recipes read ZINC250k through `defog_adapt.data`.
- **PyComex**, which the recipes use (`pip install pycomex`). It caps numpy below 2, so give it an environment of its
  own.

!!! warning "Decide how the label is computed, before you train anything"
    A graph stores atoms and bonds but no formal charge, so the property of a source SMILES and of the molecule the
    graph represents can differ. Every recorded result labels by the **decoded graph** (`property_from="decoded"`,
    which is the recipes' default). The adapter, the property head and the size model must all use the same
    convention. If they do not, they disagree about what the target means, and the disagreement is largest at the low
    end of logP.

## Train it

The library side is a few lines:

--8<-- "docs/concepts/adapters.md:train_adapter"

`for_base` builds the recorded architecture and stamps the adapter with a fingerprint of its base, so using it with
the wrong base fails when the sampler is built. `AdapterModule` freezes the base, keeps it in evaluation mode, trains
exactly the adapter's parameters, and leaves the base's weights out of the checkpoint.

In practice you run the recipe, which wraps this in an experiment and evaluates the adapter at the end. The recipes
use PyComex, which takes Python literals, so a string needs two layers of quotes:

```bash
python examples/adapter_training__zinc.py --BASE_CKPT "'ckpts/zinc_rl2_seed42/best_model'" --PROPERTY "'logp'"
```

## Keep the recorded recipe

`ADAPTER_ARCHITECTURE` and `ADAPTER_TRAINING` in `defog_adapt.presets` are one measured recipe. A shortened run is
under-trained, and changing one setting in isolation does not reproduce it. Two seeds of the same recipe differ by more
than most differences between architecture settings, so a sweep of single seeds does not rank configurations.

## What to do next

An adapter steers on its own. To add Feynman-Kac steering you also need a property head and a size model, fitted under
the same label convention:

```bash
python examples/fit_property_head__zinc.py --base ckpts/zinc_rl2_seed42/best_model --property logp \
    --out ckpts/heads/logp_head --with-size-model
```

Then [steer toward a target](steer-toward-a-target.md). To tighten a trained adapter further, the
`adapter_rl_finetune__zinc.py` recipe fine-tunes it by policy gradient; see the [recipes](../reference/recipes.md).
Details: [Adapters](../concepts/adapters.md).
