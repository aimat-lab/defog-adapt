# Recipes

The recipes are code you read and adapt, not a command-line product. The library provides the building blocks, and
these scripts show how they were wired for every recorded result. Each of them runs.

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

## The pipeline, in order

```text
training__e1.py ─▶ rl_sanity__zinc.py (×2) ─▶ adapter_training__zinc.py ─▶ adapter_rl_finetune__zinc.py
   base model        a cleaner base              one adapter per property      (optional) tighter adapter
                              │                          │
                              └──────────▶ fit_property_head__zinc.py ──▶ head + size model
                                                         │
                                          steer_zinc.py ◀┘   sample, steer, decode, score
```

1. **Train a base** with `training__e1.py`.
2. **Clean it up** with `rl_sanity__zinc.py`, twice. The second round starts from the first round's best checkpoint.
3. **Train an adapter** per property with `adapter_training__zinc.py`, then a **head and a size model** for it with
   `fit_property_head__zinc.py --with-size-model`, under the same label convention.
4. **Steer** with `steer_zinc.py`, or tighten the adapter first with `adapter_rl_finetune__zinc.py`.
5. **Compare** the ways of conditioning with `compare_arms__zinc.py`.

## Running them

The PyComex recipes take Python literals on the command line, so a string needs two layers of quotes:

```bash
pip install "defog-adapt[chem]" pycomex     # the recipes need RDKit and PyComex
python examples/training__e1.py --DATASET "'zinc'" --__TESTING__ True     # smoke test, CPU, about a minute
python examples/training__e1.py --DATASET "'zinc'"                        # the real run
```

`--__TESTING__ True` is a smoke configuration that every PyComex recipe has: a tiny model, a few hundred molecules and
a handful of sampling steps. It proves that the pipeline executes. The molecules it produces are mostly invalid,
because nothing has trained. The reference data is read from `$DEFOG_ADAPT_HOME` or a per-user cache and is never
downloaded silently: pass `--DOWNLOAD True` to opt in.

PyComex caps numpy below 2, which is why it is not a dependency of this package. Run the recipes in an environment of
their own if you want numpy 2 elsewhere, or use the plain scripts, which do not use PyComex.

Two things recur across the recipes, because each cost a recorded run real time. The **label convention** (`decoded`
in every recorded result) must agree between the adapter, its head, its size model and the RL targets. And the
**operating point must be the deployment point**: when a training run and its benchmark disagree, check first that
they use the same guidance weight, node-count draw and sampling stochasticity.
