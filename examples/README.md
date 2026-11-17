# Examples and recipes

Everything here is code you read and adapt, not a command-line product. The library
(`defog_adapt`) provides the building blocks; these files show how they were wired for every
recorded result, and each of them runs.

| file | what it is | needs |
|---|---|---|
| [`quickstart_cpu.py`](quickstart_cpu.py) | Every building block, end to end, on a toy problem. Tens of seconds on a desktop CPU. **A wiring check, not a result.** | `defog-adapt[chem]` |
| [`training__e1.py`](training__e1.py) | Train the unconditional base on **ZINC250k** or **MOSES** (PyComex). | `[recipes]`, the dataset, a GPU |
| [`rl_sanity__zinc.py`](rl_sanity__zinc.py) | Policy-gradient fine-tune of a ZINC base against disconnected fragments and odd ring sizes (PyComex). | `[recipes]`, a base checkpoint |
| [`adapter_training__zinc.py`](adapter_training__zinc.py) | Train one frozen-base adapter for logP, TPSA, QED or SAScore (PyComex). | `[recipes]`, a base checkpoint, a GPU |
| [`fit_property_head__zinc.py`](fit_property_head__zinc.py) | Fit the property head (the Feynman-Kac ruler and RL reward) and the learned size model. A plain script. | `[chem,data]`, a base checkpoint |
| [`adapter_rl_finetune__zinc.py`](adapter_rl_finetune__zinc.py) | Tighten a trained adapter by policy gradient (PyComex). | `[recipes]`, base + adapter (+ head, size model) |
| [`steer_zinc.py`](steer_zinc.py) | **The assembly**: adapters + learned sizes + Feynman-Kac + decoding, and an evaluation mode. A plain script. | `[chem,data]`, the parts above |
| [`_common.py`](_common.py) | The shared helpers: property functions, the ZINC vocabulary, labelled graphs. | |

## The pipeline, in order

```
training__e1.py ─▶ rl_sanity__zinc.py (×2) ─▶ adapter_training__zinc.py ─▶ adapter_rl_finetune__zinc.py
   base model        a cleaner base              one adapter per property      (optional) tighter adapter
                              │                          │
                              └──────────▶ fit_property_head__zinc.py ──▶ head + size model
                                                         │
                                          steer_zinc.py ◀┘   sample, steer, decode, score
```

1. **Train a base**: `python examples/training__e1.py --DATASET "'zinc'"` (300 epochs, which does not
   fit in one cluster allocation, so it chains across wall-clock windows; see that file's docstring).
2. **Optionally clean it up** with `rl_sanity__zinc.py`. The recorded adapters were trained on a base
   that had had two rounds of this. See [`docs/caveats.md`](../docs/caveats.md): that base was also
   trained under an *earlier* configuration than `training__e1.py` reproduces, and no adapter run on an
   `training__e1.py` base is recorded.
3. **Train an adapter** per property (`adapter_training__zinc.py`), then a **head and size model** for it
   (`fit_property_head__zinc.py --with-size-model`) using the *same* `--property-from` convention.
4. **Steer** with `steer_zinc.py`, or fine-tune the adapter further with `adapter_rl_finetune__zinc.py`.

## Running them

The PyComex recipes take Python literals on the command line, so strings need two layers of quotes:

```bash
pip install "defog-adapt[recipes]"          # PyComex, RDKit, pandas
python examples/training__e1.py --DATASET "'zinc'" --__TESTING__ True     # smoke test, CPU, ~1 min
python examples/training__e1.py --DATASET "'zinc'"                        # the real run
```

`--__TESTING__ True` is a smoke configuration that every PyComex recipe has: tiny model, a few hundred
molecules, a handful of sampling steps. It proves the pipeline executes and its checks pass; the
molecules it produces are mostly invalid, because nothing has trained.

Results land in `examples/results/<recipe>/<timestamp>/` (git-ignored). The reference data is read from
`$DEFOG_ADAPT_HOME` or the per-user cache, and is **never downloaded silently**: pass `--DOWNLOAD True`
to `training__e1.py` or `adapter_training__zinc.py`, or call
`defog_adapt.data.zinc_reference.download_reference()` once, to opt in. Files are checked against pinned
SHA256s.

### A note on dependencies

PyComex caps `numpy<2`, so `defog-adapt[recipes]` resolves to a numpy 1.x stack, where plain
`defog-adapt[chem,data]` is free to use numpy 2. Both work. If you want numpy 2 elsewhere, run the
recipes in their own environment, or use the plain scripts (`quickstart_cpu.py`, `steer_zinc.py`,
`fit_property_head__zinc.py`), which do not use PyComex.

## What to read before trusting a number

Two things recur across the recipes, because each of them cost a recorded run real time:

* **Label convention.** A graph stores atoms and bonds but not formal charges, so a molecule's
  property and the property of the graph that represents it differ, by up to 1.6 log units at the low
  end of logP. Every recorded result labelled by the *decoded* graph (`--property-from decoded`). The
  adapter, its head, its size model and the RL targets must all agree. `steer_zinc.py` refuses a head
  whose statistics disagree with its adapter's.
* **The operating point must be the deployment point.** Guidance weight, the node-count draw and the
  sampling stochasticity each have a default that is easy to leave different from what you will deploy
  at. When a training run and its benchmark disagree, check they are the same policy first.

The numbers the recipes were measured at, their uncertainties, and what did *not* work are in
[`docs/results.md`](../docs/results.md) and [`docs/caveats.md`](../docs/caveats.md).
