# Migrating from the research fork

`defog-adapt` is extracted from a research fork of [DeFoG](https://github.com/qym7/DeFoG) whose Python package
was called `defog`. This page is for anyone with code, scripts or checkpoints from that fork. Nothing here
affects you if you are starting fresh.

## Names

| | research fork | this package |
|---|---|---|
| PyPI distribution | (not published) | `defog-adapt` |
| import name | `defog` | `defog_adapt` |
| experiment helpers | `experiments.utils` | `defog_adapt.domains.molecule` |
| the adapter class | `AdaLNAdapter` | `CrossAttentionAdapter` (`AdaLNAdapter` stays importable: the same class) |

The distribution on PyPI called `defog` is an **unrelated** project (a text-to-SQL client); the import name
changed too so that the two can be installed side by side. Their top-level modules do not overlap.

```python
# before
from defog.core import DeFoGModel, AdaLNAdapter, FeynmanKacSampler
from defog.domains.molecule import smiles_to_pyg_data
from experiments.utils import build_encoders
# after
from defog_adapt.core import DeFoGModel, CrossAttentionAdapter, FeynmanKacSampler
from defog_adapt.domains.molecule import smiles_to_pyg_data, build_encoders
```

Behaviour of everything that **was kept** is unchanged, and that is checked, not assumed: a golden-output test
runs real checkpoints through the real sampling, adapter, Feynman-Kac and RL code and matches the research
package's outputs bit for bit.

## What was removed

Only features with a recorded positive result were kept. Everything below is **not in the package**; if you
used it, the reason and the way out are in the right-hand column.

| removed | why / what to do |
|---|---|
| `AdapterRAMTrainer` | untried at scale. Use `AdapterGDPOTrainer`. |
| `AdapterDAMTrainer` | discrete adjoint matching: a negative result, closed. Use `AdapterGDPOTrainer`. |
| `AdapterNFTTrainer`, `NFTTrainer`, `nft_advantage`, `nft_logit_loss`, `RunningStd` | DiffusionNFT: on a four-seed 100-target evaluation the gain was -0.016, p = 0.15, so not significant. Use `GDPOTrainer` / `AdapterGDPOTrainer`. |
| `renoise_states`, `draw_times` | the re-noising helpers shared by the removed RAM, DAM and NFT estimators. |
| `CreditGuidance`, `CreditHead`, `credit_gkl`, `gather_log_m`, `guided_logmarginals`, `edge_mask_of`, `constant_baseline`, `per_class_baseline` | credit assignment and its helpers: a null result. |
| `FragmentTypicalityPenalty`, `FragmentVocabulary`, `MMDPenalty`, `brics_fragments`, `morgan_fingerprints` | distribution-fidelity penalties for the RL reward, and their helpers. The fragment-typicality term *rewarded* the reward hack it was meant to stop (run-level AUC 0.048 against the known hacked policy), and the underlying problem was the aromatic representation, which the kekulized one fixes. Compare distributions externally (FCD, scaffold similarity); for fingerprints use RDKit's `rdFingerprintGenerator` (and `morgan_condition` for an adapter's condition vector). |
| `ExactGuidance`, `ClassifierRatio`, `DensityRatio`, `EnergyRatio`, `RewardRatio`, `CompositeGuidance`, `GuidanceModule`, `AmortizedPropertyGuidanceModule`, `LatentGuidanceModule`, `build_guidance_network`, `bregman_loss`, `tanimoto_similarity`, `GuidedSampler`, `JointGuidanceSampler` | the guidance family and its helpers, superseded by adapters (and null for the fingerprint latent-guidance variant). Use RDKit's `DataStructs` for Tanimoto similarity. |
| `GuideBranch` | autoguidance, no recorded positive result. `AdapterComposition` no longer takes `guide=`. |
| `RefinementSampler` | no recorded positive result. |
| `GroundedAdapterModule`, `AdapterRegistry` | joint adapter-and-head training, and a registry: no recorded positive result. Train the head on its own (`fit_property_head`) and load adapters directly. |
| `MultiPropertyEnergy` | use `StandardisedEnergy([(e1, s1, w1), (e2, s2, w2)])`. |
| `Reward`, `reward_from_energy` | a generic wrapper; a reward is any `(X1, E1, node_mask) -> (K,)` callable. |
| `SpectrumEncoder` | the spectrum *condition encoder*; the spectrum *energy* (`SpectrumEnergy`, `Band`) is kept. |
| `UniformSizeDistribution` | untried. |
| `defog.data.guacamol_reference` (`GuacamolReferenceSplit`) | the GuacaMol result was trained on a split that contained the test set. |

Removed **options** on classes that were kept:

| class | removed argument | note |
|---|---|---|
| `CrossAttentionAdapter` (`AdaLNAdapter` in the fork) | `interior_ff`, `interior_attn` | interior conditioning, superseded by node cross-attention (see below) |
| `AdapterComposition` | `blend_space`, `guide` | only probability-space blending is kept: rate-space blending was numerically broken above weight 1 |
| `AdapterModule` | `cond_drop_prob`, `l10_lr_scale` | condition dropout (recorded value 0) and the interior-head learning-rate scale |
| `FeynmanKacSampler` | `rejuvenate`, `jump_length`, `proposal_transform`, `constraint`, `n_free` | rejuvenation was dominated; `beta` is now a plain number (beta schedules were dominated). Inpainting is `InpaintingSampler`. |
| `Sampler` | `posterior_transform` | the per-step posterior hook, with the samplers built on it |
| `GDPOTrainer` | `positive_only`, `kl_target` | |
| `AdapterGDPOTrainer` | `kl_target` | **`rollout_weight` is now required** (it used to default to 1) |
| `LearnedSizeDistribution` | `cond_encoder` | a reserved slot, never used |
| `RolloutSampler` | `record_trace` | |

If your code passed a removed option at a neutral value it must now drop the argument; passing it at a
non-neutral value was already a different model.

### Defaults that changed

The trainers' defaults are now **the recorded recipes** (`GDPOTrainer`: lr 2e-5, EMA 0.9, KL 0.3, `"sum"`
reduction, 128 rollouts of 100 steps at eta 0; `AdapterGDPOTrainer`: lr 1e-4, EMA 0.9, KL 0.05, 250 steps at
eta 1.0, `"mean"` reduction), and `defog_adapt.presets` states them as data. `DeFoGModel`'s own constructor
defaults are **unchanged** (they differ from every recorded run), so a model you build without arguments behaves
as before; use `presets.E1_MODEL`.

## Checkpoints

* **Base models** saved by the research fork load with `DeFoGModel.load`. The golden test does exactly this
  with real checkpoints.
* **Adapters** load with `CrossAttentionAdapter.load` (or `AdaLNAdapter.load`, its name in the fork) unless they enable an interior feature or use the spectrum
  condition encoder; those raise an error that names the feature, rather than loading a different model. Load
  them with the research package.
* **Heads and size models** (`PropertyHead`, `LearnedSizeDistribution`) load as before.
* **Config keys** that are accepted today stay accepted: a stored `interior_ff: false` is read and ignored.
* **Do not trust the hyperparameters stored in a re-packaged checkpoint.** See
  [Results and caveats](../about/results-and-caveats.md#the-adapter-base-is-a-re-packaged-checkpoint).
* Checkpoints are pickles. Only load ones you trust ([Adapters](../concepts/adapters.md#file-format-compatibility-and-trust)).

## Reference data

The loaders (`zinc_reference`, `moses_reference`, `chembl_reference`) read from `$DEFOG_ADAPT_HOME` or a
per-user cache directory instead of a path inside the repository, and **never download silently**:
`load_reference_split(download=True)` (or `download_reference()`) opts in. Files are still checked against
pinned SHA256s.

## What is new

`StandardisedEnergy` (a variance-standardised sum of energies, so the Feynman-Kac `beta` is dimensionless),
`SanityReward` (the unconditional structural reward, previously in a script), `fit_size_model` exported,
`morgan_condition`, `defog_adapt.presets`, `FeynmanKacSampler.trace`, and `defog_adapt.__version__` read from the
installed metadata. `MoleculeDomain.encode` now encodes an aromatic core in Kekule form when the vocabulary is
kekulized (it used to raise on the aromatic bond class). The library no longer calls `matplotlib.use` at import
or leaks one pyplot figure per epoch.
