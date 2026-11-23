# Reinforcement learning

Two uses, one estimator: **structural RL on an unconditional base** (fewer disconnected fragments and odd
rings) and **RL on an adapter** (tighter conditioning). Both fine-tune against a reward on the *generated*
molecule, which can be non-differentiable: an RDKit property, a connectivity check, a learned head. Results
and limits are on [Results and caveats](../about/results-and-caveats.md); read it before relying on a gain.

## The estimator: GDPO's eager policy gradient

Denoising is an MDP whose reward is paid only at the terminal clean graph. Naive REINFORCE pushes the
log-probability of every transition, which for DeFoG's clean-graph parameterisation expands into a
reward-agnostic sum over all clean graphs, with high variance. The **eager** estimator of Graph Diffusion
Policy Optimization ([arXiv:2402.16302](https://arxiv.org/abs/2402.16302)) replaces that term with the
log-probability of the trajectory's *own realised endpoint*:

```
g(theta) = 1/K * sum_k  A_k * sum_{t in T_k}  grad log p_theta(G1_k | G_t,k)
```

DeFoG's network already emits `p_theta(G1 | G_t)`, so this is exactly the masking- and symmetry-aware
cross-entropy the training loss computes, with the rollout's sampled endpoint as the one-hot target: **an
advantage-weighted cross-entropy at noisy states along each rollout.** The rollout reuses the ordinary
`Sampler`; the fine-tuned weights are a plain DeFoG checkpoint that loads with `DeFoGModel.load` and samples
with no reward at generation time.

`GDPOTrainer` and `AdapterGDPOTrainer` share the machinery (K rollouts per iteration, `subsample_steps` noisy
states per trajectory in the gradient, a KL pull to a frozen reference, EMA weights, gradient clipping). Their
**defaults are the recorded recipes** (`presets.SANITY_RL_TRAINER`, `presets.ADAPTER_RL_TRAINER`).

| knob | what it does |
|---|---|
| `eta` | CTMC stochasticity of the rollouts = the exploration temperature. Not the evaluation eta. |
| `advantage_mode` | `"mean"` (Dr. GRPO: mean baseline, no std normalisation, the default), `"grpo"` (std-normalised, which amplifies medium-variance groups and biases learning toward mid-difficulty targets), `"none"`. |
| `reduction` | `"sum"` is the true joint log-likelihood; `"mean"` divides by the real node and edge counts, so large molecules do not dominate the gradient on a variable-size dataset. |
| `kl_coef` | the pull to the frozen starting policy: the main defence against collapsing the distribution. |
| `crn` | common random numbers: trajectories in one advantage group share their initial noise and size, so the group baseline has lower variance (needs groups and `eta > 0`). |
| `condition_sampler` | `() -> (cond, groups)` for conditional runs; `make_condition_sampler(lo, hi, K, G, seed)` draws `G` targets uniformly over a range with `K // G` rollouts each. |

## Structural RL on an unconditional base

```python
from defog_adapt.core import DeFoGModel, GDPOTrainer, SanityReward
from defog_adapt.domains import MoleculeDomain
from defog_adapt.presets import SANITY_RL_TRAINER

model = DeFoGModel.load("base.ckpt")
reward = SanityReward(MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles))
trainer = GDPOTrainer(model, reward, **SANITY_RL_TRAINER, seed=42)
trainer.fit(25, on_iter=keep_the_best)          # evaluate and checkpoint INSIDE the loop
trainer.save("rl_round1/best_model", use_ema=True)
```

`SanityReward` is graded, `r = valid + connected + rings_in_[3, 8]`, each 0 or 1. A single "all three" flag is
the exact target, but 95% of samples already satisfy it, which leaves a near-constant reward and almost no
group-relative advantage; the graded form separates the failing 5% and tells a one-fault sample from a two-fault
one. An invalid molecule scores 0 outright, so it is strictly worse than any valid one and there is no incentive
to trade validity for either term. `examples/rl_sanity__zinc.py` is the full recipe.

* **Never rely on the final iteration.** A run peaked near iteration 20 and collapsed by iteration 59, and
  because only the last model was saved the good policy was unrecoverable. EMA at 0.9 has a memory of about ten
  iterations, so the final EMA reflected the collapsed region too. Keep periodic checkpoints and a best-so-far,
  selected on a **rolling mean** (a single batch of 128 carries about 2% noise and would reward a lucky
  iteration).
* **Score FCD, not just the reward.** Sanity can be raised by collapsing onto a narrow, safe region of chemical
  space, so a sanity gain with a degraded FCD is a failed run. The recipe dumps the SMILES before and after for
  external scoring against the validation split.
* **Two rounds is the useful depth.** Round two starts from round one's best checkpoint, and the KL reference is
  rebuilt from the model the trainer is handed, so the pull is toward round one, not back to the original base:
  a ratchet, not a reset. A third round moved nothing outside the noise.
* **A representation can look like a reward-design problem.** The same reward raised validity on an aromatic
  MOSES base while FCD doubled on four of four seeds; on a kekulized base it does not happen.

## RL on an adapter

```python
from defog_adapt.core import AdapterGDPOTrainer, HeadPropertyMatchReward, make_condition_sampler
from defog_adapt.presets import ADAPTER_RL_TRAINER, ADAPTER_WEIGHT

reward = HeadPropertyMatchReward(head, atom_enc, bond_enc, atom_dec, bond_dec, device, scale=prop_std)
trainer = AdapterGDPOTrainer(
    base, adapter, reward,
    rollout_weight=ADAPTER_WEIGHT["logp"],              # REQUIRED: the weight it will be deployed at
    condition_sampler=make_condition_sampler(p_lo, p_hi, 128, 8, seed=42),
    size_dist=size_model,                               # draw rollout sizes from P(n | target)
    **ADAPTER_RL_TRAINER, seed=42)
metrics = trainer.step()                                # reward_mean, reward_min, kl, adv_std, grad_norm
```

The base stays frozen and only the adapter moves, so composability is preserved. The reward is
**conditional**: `cond_reward(X1, E1, node_mask, cond) -> (K,)` scores each rollout against its own target, and
the advantage is grouped by target. The KL reference is a frozen copy of the pre-RL adapter.

### The operating point must be the deployment point

Three knobs each had a default that quietly differed from deployment, and earlier runs were wrong on all three
at once, so RL improved a policy nobody runs. When an RL run and its benchmark disagree, check they are the same
policy before looking at anything else.

1. **Guidance weight.** `rollout_weight` is the weight the rollouts are generated at **and** the gradient is
   scored at (they are the same number by construction, so the scored policy equals the behaviour policy at any
   weight). It has no default, on purpose: it must be the weight the adapter is sampled at (1.5 for logP, TPSA and
   SAScore, 2.0 for QED). Training at 1 and deploying at 1.5 optimises a policy nobody uses.
2. **The node-count draw.** Deploy with a learned `P(n | target)` and the rollouts must draw from it too:
   otherwise the policy is trained under a distribution you do not use (rollouts averaged +4.5 heavy atoms over
   the evaluation: macrocycles and fragments). The evaluation and the probe need the same draw, or they grade a
   different policy than the one being trained.
3. **Stochasticity.** Rollouts explore at a small eta (1.0, a swept winner under common random numbers); the
   benchmark samples at eta 25. They are different quantities; do not "match" them without a sweep.

### The rewards

| reward | shape | used for |
|---|---|---|
| `HeadPropertyMatchReward` | connectivity-first: invalid -10 < disconnected -4 < connected in [-3, 0]; the property term comes from a learned head, after a decode and re-encode | the confirmed runs |
| `PropertyMatchReward` | the same tiers, with the closed-form (RDKit) property | properties with a closed form |
| `WeightedSanityPropertyReward` | `3 * targeting + 1 * sanity`, both in [0, 1], targeting reaching 0 at three standard deviations off; sanity is valid, connected and rings 3 to 8 | emphasising targeting |

The **tiering is the load-bearing part** of the first two and must not be "simplified": the disconnected reward
sits strictly below the property clamp, so any connected molecule outranks any disconnected one, however far
off-target, while the property term still gives a full gradient among connected molecules. Flatten the ordering
and the policy learns to hit the target with fragments, which score well on most property functions and are not
molecules. The weighted reward does the opposite on purpose (targeting can outrank sanity); on QED it gave -26%
in the internal evaluation without costing sanity, but -9.8% on a duplicate seed, so trust the direction and not
the size.

### Probing, selecting and not fooling yourself

* **Probe a wide, fixed set of targets.** The probe that selects the snapshot to ship scored the two extreme
  percentiles the report used, five times; best-of-five selection overfit those two points, and the probe claimed
  -16.8% on its low target while the 100-target protocol measured +0.035 there. `adapter_rl_finetune__zinc.py`
  scores nine fixed targets across the 5th to 95th percentiles (fixed, so probe error stays comparable across
  iterations).
* **Always compute the true property on the molecules you select on**, and log it beside the selection metric.
  With a head reward the head's ranking of candidate snapshots inverted against RDKit once, and truth-based
  selection beat head-based selection by 0.0084 (p = 0.043). The recipe saves a second, truth-chosen snapshot.
* **Gate on validity.** A snapshot must keep validity within a margin of the pre-RL adapter to be eligible.
* **Check that it moved.** The recipe prints the largest weight change of the deployed adapter against its
  input; a near-zero change means RL never learned and any pre-to-post difference is evaluation noise. An early
  run shipped an unchanged adapter as a result; the learning rate had been 20 times below the adapter's training
  rate, and the EMA at 0.999 lagged it.
* **Guard what you rely on.** RL optimises what it measures: in a fingerprint-adapter experiment (not part of
  this package) it erased the adapter's inference of molecule size because nothing protected it.
* **Keep one round.** A second round on QED regressed significantly on the 100-target protocol while its own
  probe claimed the opposite.

### What does not add

RL on an adapter and Feynman-Kac steering correct the same error. On top of FK the RL increment was not
significant (a research-fork result; see [the earlier numbers](../about/results-and-caveats.md#the-earlier-numbers)). Pick the cheaper one for your use, or use both and
do not expect the sum.

## Other estimators

The research fork also tried ratio-matching (RAM), adjoint matching (DAM), a likelihood-free DiffusionNFT
estimator and credit assignment. None has a positive result and none is in this package; the reasons are in
[Migrating from the research fork](../reference/migration.md#what-was-removed).
