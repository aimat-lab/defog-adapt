# Recipes

The recipes are code you read and adapt, not a command-line product. The library (`defog_adapt`) provides the building
blocks, and these scripts show how they were wired for every recorded result. Each of them runs.

They live in the [`examples/`](https://github.com/aimat-lab/defog-adapt/tree/main/examples) folder of the repository,
not in the installed package, so clone the repository to run them.

| Recipe | What it does | Needs |
|---|---|---|
| [`quickstart_cpu.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/quickstart_cpu.py) | Every building block, end to end, on a toy problem. Tens of seconds on a desktop CPU. A wiring check, not a result. | `defog-adapt[chem]` |
| [`training__e1.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/training__e1.py) | Trains the unconditional base on ZINC250k or MOSES. With `--CONDITION "'logp'"` (ZINC) the same recipe trains the model end to end on a property: the model an adapter is measured against. | PyComex, the dataset, a GPU |
| [`rl_sanity__zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/rl_sanity__zinc.py) | **RL on the base.** Fine-tunes a ZINC base by policy gradient against disconnected fragments and odd ring sizes. | PyComex, a base checkpoint |
| [`adapter_training__zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/adapter_training__zinc.py) | Trains one frozen-base adapter for logP, TPSA, QED or SAScore. | PyComex, a base checkpoint, a GPU |
| [`fit_property_head__zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/fit_property_head__zinc.py) | Fits the property head (the Feynman-Kac ruler and the RL reward) and the learned size model. A plain script. | `defog-adapt[chem]`, a base checkpoint |
| [`adapter_rl_finetune__zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/adapter_rl_finetune__zinc.py) | **RL on an adapter.** Tightens a trained adapter by policy gradient. | PyComex, a base and an adapter, and a head and size model |
| [`steer_zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/steer_zinc.py) | The assembly: adapters, learned sizes, Feynman-Kac steering and decoding, with an evaluation mode. A plain script. | `defog-adapt[chem]`, the parts above |
| [`compare_arms__zinc.py`](https://github.com/aimat-lab/defog-adapt/blob/main/examples/compare_arms__zinc.py) | The comparison behind the results figure: the ways of conditioning, on one logP benchmark and several seeds. A plain script. | `defog-adapt[chem]`, the parts above |

A file called `_common.py` holds the helpers they share: the property functions, the ZINC vocabulary and the labelled
graphs.

## The pipeline, in order

1. **Train a base** with `training__e1.py --DATASET "'zinc'"`. The recorded run is 300 epochs, which does not fit in one
   cluster allocation, so the recipe chains across wall-clock windows; its docstring explains how. The guide
   [Train an unconditional base](../guides/train-an-unconditional-base.md) does the same at a small scale.
2. **Clean it up** with `rl_sanity__zinc.py`, twice. The second round starts from the first round's best checkpoint. The
   recorded adapters were trained on a base that had had exactly these two rounds, so steps 1 and 2 reproduce their
   lineage. See [the adapter base is a re-packaged checkpoint](../about/results-and-caveats.md#the-adapter-base-is-a-re-packaged-checkpoint)
   for a trap in reading that checkpoint's stored hyperparameters.
3. **Train an adapter** per property with `adapter_training__zinc.py`, then a **head and a size model** for it with
   `fit_property_head__zinc.py --with-size-model`, using the *same* `--property-from` convention. The guide
   [Train an adapter](../guides/train-an-adapter.md) does the same at a small scale.
4. **Steer** with `steer_zinc.py`, or tighten the adapter first with `adapter_rl_finetune__zinc.py`.
5. **Compare** the ways of conditioning on one benchmark with `compare_arms__zinc.py`: validation first, to choose the
   guidance weights, then the held-out test split once, with them frozen.

## Running them

The PyComex recipes take Python literals on the command line, so a string needs two layers of quotes:

```bash
pip install "defog-adapt[chem]" pycomex     # the recipes need RDKit and PyComex
python examples/training__e1.py --DATASET "'zinc'" --__TESTING__ True     # smoke test, CPU, about a minute
python examples/training__e1.py --DATASET "'zinc'"                        # the real run
```

`--__TESTING__ True` is a smoke configuration that every PyComex recipe has: a tiny model, a few hundred molecules and
a handful of sampling steps. It proves that the pipeline executes and its checks pass. The molecules it produces are
mostly invalid, because nothing has trained.

Results land in `examples/results/<recipe>/<timestamp>/` (ignored by git). The reference data is read from
`$DEFOG_ADAPT_HOME` or a per-user cache and is **never downloaded silently**: pass `--DOWNLOAD True` to `training__e1.py`
or `adapter_training__zinc.py`, or call `defog_adapt.data.zinc_reference.download_reference()` once, to opt in. Files are
checked against pinned SHA256s.

### A note on dependencies

PyComex caps `numpy<2`, which is why it is not a dependency of this package: it would hold every user on numpy 1.x. An
environment with PyComex resolves to a numpy 1.x stack, where `defog-adapt[chem]` alone is free to use numpy 2. Both
work. If you want numpy 2 elsewhere, run the recipes in an environment of their own, or use the plain scripts
(`quickstart_cpu.py`, `steer_zinc.py`, `fit_property_head__zinc.py`), which do not use PyComex. The recipes' smoke tests
pass on PyComex 0.13.1 and on the latest release.

## What to read before trusting a number

Two things recur across the recipes, because each of them cost a recorded run real time:

* **Label convention.** A graph stores atoms and bonds but not formal charges, so a molecule's property and the property
  of the graph that represents it differ, by up to 1.6 log units at the low end of logP. Every recorded result is labelled
  by the *decoded* graph (`--property-from decoded`). The adapter, its head, its size model and the RL targets must all
  agree. `steer_zinc.py` refuses a head whose statistics disagree with its adapter's.
* **The operating point must be the deployment point.** Guidance weight, the node-count draw and the sampling
  stochasticity each have a default that is easy to leave different from what you will deploy at. When a training run and
  its benchmark disagree, check first that they are the same policy.

The numbers the recipes were measured at, their uncertainties, and what did *not* work are on
[Results and caveats](../about/results-and-caveats.md).
