# Getting started

This page installs the package and runs the whole pipeline once, on a CPU, with nothing to download: a toy DeFoG
model, a logP adapter on top of it, and sampling steered toward two logP targets. It takes about half a minute.

## Install

```bash
pip install "defog-adapt[chem]"
```

The `chem` extra adds RDKit, which the molecular domain needs. If you want a particular PyTorch build, a GPU build
or a CPU-only one, install PyTorch for your platform first, from [pytorch.org](https://pytorch.org/get-started/locally/),
and then `defog-adapt`: pip keeps the PyTorch you installed. The package is `defog_adapt`; it does not clash with
the unrelated `defog` package on PyPI.

## Run it

--8<-- "README.md:quickstart"

## What it does

1. It trains a toy **base** model on a few dozen small molecules. This is the frozen, unconditional model that
   everything else sits on.
2. It trains an **adapter** on top of it, with the base frozen, so the adapter learns to condition the base on logP.
3. It **samples** with the adapter, once for each of two logP targets, decodes the graphs back to molecules and
   prints how many were valid and their mean logP.

It prints one line per target, after some start-up messages from PyTorch Lightning. The details vary a little
between machines and versions:

--8<-- "README.md:quickstart_output"

In this output the mean logP lands close to each target: about 0 for the first and about 2 for the second.

!!! note "A wiring check, not a result"
    The model is tiny and sees a few dozen molecules, so expect some odd molecules among the samples. What the run
    shows is the idea: one frozen base, and a small adapter that moves the samples toward the property you ask for.
    This exact code is run by the project's tests, so it keeps working as the library changes.

## Where next

- [Concepts](concepts/index.md): what adapters, Feynman-Kac steering and RL fine-tuning are, in one page.
- [Train an unconditional base](guides/train-an-unconditional-base.md), [Train an adapter](guides/train-an-adapter.md) and
  [Steer toward a target](guides/steer-toward-a-target.md): the same steps, one guide each, on molecules of your own.
- [Recipes](reference/recipes.md): the scripts that train the models this page only imitates.
