# Concepts

A DeFoG model generates a graph by denoising: it starts from noise and, over many small steps, moves toward a clean
graph. The **base model** in this package is *unconditional*. It has learned what molecules look like, not which ones
have a given logP. defog-adapt leaves it as it is and gives you three ways to steer it. This page says what each one
is; the [guides](guides/train-an-adapter.md) show how to use them.

| Tool | What it changes | When it acts |
|---|---|---|
| [Adapter](#adapters) | Nothing in the base. A small module is trained beside it. | Training, once per property; then at every sampling step. |
| [Feynman-Kac steering](#feynman-kac-steering) | Nothing: the generator is neither trained nor changed. | Sampling time only. |
| [RL fine-tuning](#rl-fine-tuning) | The weights of the base model or of an adapter. | Training. Sampling afterwards needs no reward. |

## Adapters

An adapter is a small network that reads the target (say, logP = 4) and the flow time, and turns them into a
modulation of every layer of the base transformer: a scale, a shift and a gate, for the atoms, the bonds and the
graph as a whole. The gates start at zero, so **an untrained adapter reproduces the base exactly**. It is trained with
the base's own denoising loss while the base stays frozen, so what you get is a conditional denoiser sitting on top
of an unchanged unconditional model.

Two additions to this scheme carry the recorded results. *Cross-attention* lets each atom look at the condition,
instead of every atom receiving the same correction. *Fourier features* of the condition make it easier to separate
nearby targets such as 3.5 and 4.2, which a single raw number makes slow to learn.

Three properties follow, and they are the reason adapters exist:

- **Swappable.** One frozen base, and one small file per property.
- **Composable.** Two adapters can be combined into a joint condition without training a joint model.
- **The unconditional path survives.** The base can be improved separately, for example by structural RL, and the
  same machinery still applies.

The price is that an adapter is tied to the base it was trained on: to its vocabulary for certain, and in practice to
its weights. At sampling time a **guidance weight** sets how hard the adapter pulls. The best value differs by
property and is recorded in `ADAPTER_WEIGHT` in `defog_adapt.presets`; guiding beyond about 2 degrades validity.

Details: [adapters.md](https://github.com/aimat-lab/defog-adapt/blob/main/docs/adapters.md).

## Feynman-Kac steering

Feynman-Kac steering works at sampling time and trains nothing. It runs a population of candidates, the
*particles*, through the denoising together. At checkpoints it **resamples** them: candidates that look far from the
target are dropped and candidates that look close are copied. A candidate's weight is `exp(-beta * energy)`, and the
energy scores its *predicted clean graph*: decode it, measure the property, compare it with the target.

Nothing is differentiated, so the energy can be any black box, such as an RDKit calculation or a learned property
predictor. The particles can be proposed by the base alone or by an adapter.

Two things matter in practice:

- **`beta` is the lever, the number of particles is not.** A larger `beta` pulls harder toward the target and costs
  diversity. `StandardisedEnergy` makes `beta` dimensionless, so one value carries from one property to another.
- **Read the trace.** A run that never resamples is the unguided sampler with extra forward passes, and no error
  metric will tell you. `FeynmanKacSampler.trace` says whether the steering engaged.

Details: [steering.md](https://github.com/aimat-lab/defog-adapt/blob/main/docs/steering.md).

## RL fine-tuning

RL fine-tunes a model against a **reward on the generated molecule**, which can be non-differentiable: an RDKit
property, a connectivity check, a learned property predictor. It uses the *eager* policy gradient of
[Graph Diffusion Policy Optimization](https://arxiv.org/abs/2402.16302), which fits DeFoG's way of predicting the
clean graph: the update is an advantage-weighted cross-entropy on noisy states along each generated trajectory. The
fine-tuned weights are a plain checkpoint, and sampling from them needs no reward.

There are two uses, with one estimator:

- **Structural RL on the base**, against a reward for valid, connected molecules with sensible rings. The result is a
  cleaner base with fewer disconnected fragments and odd rings.
- **RL on an adapter**, against a reward for hitting the target. The result is tighter conditioning.

A run can collapse after it peaks, and a reward can be raised by narrowing the distribution. So keep the best
checkpoint rather than the last one, and check the quality of the whole distribution, not just the reward.

Details: [rl.md](https://github.com/aimat-lab/defog-adapt/blob/main/docs/rl.md).

## How they fit together

The recorded pipeline trains a base model, cleans it up with two rounds of structural RL, trains one adapter per
property on that base, and then samples with the adapter and Feynman-Kac steering. Tightening an adapter by RL is an
optional extra step. The [recipes](reference/recipes.md) are these steps as scripts.

Adapters and Feynman-Kac steering stack: the adapter proposes, and the energy selects. The gains from RL and from
Feynman-Kac steering do **not** add up.

## Where the evidence is

This site is the short version. Every measured number, with its protocol, is in
[results.md](https://github.com/aimat-lab/defog-adapt/blob/main/docs/results.md), and the ways each tool can be wrong
without raising an error are in [caveats.md](https://github.com/aimat-lab/defog-adapt/blob/main/docs/caveats.md).
Read both before relying on a gain. See also the [limits](about/limits.md).
