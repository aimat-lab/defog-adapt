# Steer toward a target

This guide samples molecules whose property is close to a target, from a trained base and adapter, with Feynman-Kac
steering on top. For what the pieces are, see [Concepts](../concepts/index.md).

## What you need

- **A base model and an adapter** trained for the property: see [Train an adapter](train-an-adapter.md).
- **A property head**, the ruler that Feynman-Kac scores each candidate with.
- **A size model**, which draws the number of atoms from `P(n | target)` instead of from the dataset's overall
  distribution. It is a bias correction on the size draw.

The `fit_property_head__zinc.py` recipe fits the head and, with `--with-size-model`, the size model. Fit both under
the same label convention as the adapter.

## The assembly

--8<-- "README.md:fk_steering"

`examples/steer_zinc.py` is this, assembled and runnable, with an evaluation mode:

```bash
python examples/steer_zinc.py --base ckpts/zinc_rl2_seed42/best_model --method fk \
    --property logp --adapter runs/logp_adapter.ckpt --head ckpts/heads/logp_head.ckpt \
    --size ckpts/heads/logp_head_size.ckpt --target 4.0 --n 10
```

For adapter-only sampling pass `--method adapter`, or leave out the energy in the code above and use `AdaptedSampler`.

## Choose `beta`, and check that it engaged

`beta` is the lever. Start from the recorded `FK_STEERING` preset and sweep `beta` around it, because a `beta` that is
too small makes the steering look useless when it never engaged. After every run, read `sampler.trace`, which has one
record per checkpoint:

| Field | What it means | A starved run looks like |
|---|---|---|
| `ess_over_k` | The effective sample size, as a fraction of the particles. | Near 1.0 at every checkpoint. |
| `logw_range` | The spread of the accumulated log-weights: the steering pressure. | Near 0. |
| `resampled` | Whether it resampled at that checkpoint. | Rarely `True`. |

A run that never resamples is the unguided sampler with extra forward passes. Raising the number of particles only
removes finite-sample bias; to lower the error further, raise `beta`, at a cost in diversity. Wrap the energy in
`StandardisedEnergy`, as above, so that `beta` means the same for every property.

## Judge the result on the decoded molecule

Decode the graphs, compute the **true** property of each decoded molecule with RDKit, and report it beside validity and
uniqueness. The property head is the steering ruler, not a measurement: do not score a run with it. Feynman-Kac trades
diversity for accuracy, so a better error with a lower uniqueness is the expected trade, not a free gain.
`steer_zinc.py --evaluate` follows the shape of the protocol the recorded numbers used.

## More than one property

Give `steer_zinc.py` several `--property` groups, each with its own adapter, head and size model. The adapters combine
by `--mode`, either `product` or `mean`. Which one is better depends on the pair of properties, so try both rather
than hard-coding one.

Details: [Steering at sampling time](../concepts/steering.md).
