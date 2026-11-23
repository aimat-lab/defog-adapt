# Main classes

The classes and functions you will reach for first, grouped by task. Each heading names the module to import from.
This is a map, not an API reference: for the signature and the details of anything below, read its docstring
(`help(...)`), which is the source of truth.

## Models and sampling (`defog_adapt.core`)

| Name | What it is |
|---|---|
| `DeFoGModel` | The base model, a graph transformer trained by discrete flow matching. `DeFoGModel.load(path)` reads a checkpoint and `DeFoGModel.from_dataloader(loader, ...)` builds one from data. |
| `Sampler` | The plain sampler: it denoises a model's noise into graphs, with no steering. |
| `AdaptedSampler` | A `Sampler` that applies an adapter composition at every step. |
| `FeynmanKacSampler` | Feynman-Kac particle steering: resamples a population of candidates toward the target. The proposal is the base or an adapter composition. |
| `InpaintingSampler` | Constrained generation: freezes a substructure and grows new atoms around it. |
| `SubgraphConstraint` | The substructure `InpaintingSampler` freezes: a rigid scaffold whose atom classes and internal bonds stay fixed. |

## Adapters (`defog_adapt.core`)

| Name | What it is |
|---|---|
| `CrossAttentionAdapter` | The adapter: gated FiLM modulation of a frozen base, with node-to-condition cross-attention. `AdaLNAdapter` is another name for the same class. |
| `AdapterModule` | The Lightning module that trains an adapter on a frozen base, with the base's own denoising loss. |
| `ConditionBranch` | One condition in a composition: an adapter, a raw target and a guidance weight. |
| `AdapterComposition` | Several branches combined, as a product or a mean. It is what `AdaptedSampler` takes. |

## Steering (`defog_adapt.core`)

| Name | What it is |
|---|---|
| `StandardisedEnergy` | A weighted sum of property energies, each divided by its property's variance, so that `beta` is dimensionless. |
| `MoleculePropertyEnergy` | The squared error between a property computed by RDKit and the target. |
| `LearnedPropertyEnergy` | The same, with a trained `PropertyHead` in place of RDKit. |
| `SpectrumEnergy` | The weighted squared error between a molecule's predicted spectrum and the one asked for. |
| `PropertyHead` | A graph-to-scalar property predictor: the Feynman-Kac ruler, and a learned reward for RL. |
| `fit_property_head` | Fits a `PropertyHead` on its own, with no adapter in the loop. |
| `LearnedSizeDistribution` | The distribution of the node count given the target, learned by a small network. |
| `fit_size_model` | Fits a `LearnedSizeDistribution` on pairs of condition and node count. |

## Reinforcement learning (`defog_adapt.core`)

| Name | What it is |
|---|---|
| `GDPOTrainer` | Policy-gradient fine-tuning of a base model against a reward, with the eager estimator of GDPO. |
| `AdapterGDPOTrainer` | The same for a frozen-base adapter: only the adapter moves, so the unconditional path and composability are kept. |
| `SanityReward` | The structural reward: valid, connected and ring sizes in range, each worth 0 or 1. |
| `PropertyMatchReward` | A conditional reward from a closed-form property function. |
| `HeadPropertyMatchReward` | A conditional reward from a learned property head. It is not ground truth. |
| `WeightedSanityPropertyReward` | A weighted sum of a targeting term and the structural term. |
| `make_condition_sampler` | Draws the targets for an adapter RL run: uniformly over a range, with several rollouts per target. |

## Molecules (`defog_adapt.domains.molecule`)

| Name | What it is |
|---|---|
| `MoleculeDomain` | The molecule domain: RDKit reconstruction, depiction and SMILES metrics. |
| `build_encoders` | Builds the atom and bond encoders and decoders from lists of types. |
| `smiles_to_pyg_data` | Converts a SMILES string to a graph with one-hot atoms and bonds. |
| `pyg_data_to_mol` | Converts such a graph back to an RDKit molecule. |
| `mol_to_smiles` | Canonical SMILES of a molecule, or `None` if it fails to sanitize. |
| `molecular_metrics` | The full evaluation suite over a batch of generated samples. |
| `validity_report` | All three validity conventions, with both denominators for uniqueness and novelty. |

## Data (`defog_adapt.data`)

| Name | What it is |
|---|---|
| `zinc_reference` | The ZINC250k reference split, with its vocabulary (`zinc_reference.ATOM_TYPES`). |
| `moses_reference` | The MOSES reference split. |
| `chembl_reference` | The ChEMBL reference split. |

Import through the dataset module. The datasets disagree on atom and bond vocabularies, so a bare
`from defog_adapt.data import ATOM_TYPES` would silently pick one, and it does not exist for that reason.

## Presets (`defog_adapt.presets`)

The recorded recipes as plain data. Every value has its evidence in a comment next to it in the source.

| Name | What it is |
|---|---|
| `E1_MODEL` | The model and optimiser settings behind the recorded unconditional results. |
| `E1_SAMPLING` | How the unconditional results were sampled. |
| `ADAPTER_ARCHITECTURE` | The adapter architecture behind every recorded adapter result. |
| `ADAPTER_TRAINING` | The adapter training settings: one measured recipe. |
| `ADAPTER_SAMPLING` | How the conditional benchmark was sampled. |
| `ADAPTER_WEIGHT` | The guidance weight each property's adapter measured best at. |
| `FK_STEERING` | Feynman-Kac steering as recorded. |
| `SANITY_RL_TRAINER` | Structural RL on the base, for `GDPOTrainer`. |
| `ADAPTER_RL_TRAINER` | Adapter fine-tuning by policy gradient, for `AdapterGDPOTrainer`. |
