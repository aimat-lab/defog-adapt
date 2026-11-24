# Train an unconditional base

This guide trains an unconditional DeFoG model on molecules of your own: the base that every adapter is trained on. It is
the first step of the recorded pipeline (see [how the tools fit together](../concepts/index.md#how-they-fit-together)).
The guide is small enough to run on one GPU: your own file of SMILES, a small model and a few epochs. That gives you a
working base to build on, but not the recorded one, and [Going further](#going-further) says what changes at scale.

## What you need

- **A file of molecules**, `molecules.smi`, with one SMILES per line and nothing else. The more molecules, the better the
  base; a handful runs, but only shows that the code works.
- **RDKit**: install `defog-adapt[chem]`. A **GPU** is used if there is one.

The base is **unconditional**: it learns what molecules look like, not which ones have a given property. An adapter adds
the property afterwards, in [Train an adapter](train-an-adapter.md).

All the code is one script. It is shown here in pieces, and in full at the end of the page.

## 1. Read the molecules

Read the file and shuffle it, so that the molecules you set aside for validation in the next step are a random tenth and
not the end of the file.

```python
--8<-- "docs/guides/train_an_unconditional_base.py:data"
```

`smiles` is a list of strings. Nothing is checked yet: a molecule that the model cannot represent is dropped in the next
step.

## 2. Turn them into graphs, and back

A molecule becomes a graph of atom and bond *classes*: the model sees class numbers, not chemistry. The vocabulary says
which class is which element and which bond type. It is fixed and ordered, and never read off the data: a checkpoint
decodes correctly only under the vocabulary it was trained with, and under another one it decodes to plausible molecules
made of the wrong elements, without raising an error. This guide uses the ZINC vocabulary, which is the recorded one: nine
elements and three bond types, kekulized. The aromatic bond class is left out on purpose. It promises something about a
whole ring system, and a model that decides one bond at a time cannot keep that promise, so dropping it makes a whole kind
of validity failure impossible.

```python
--8<-- "docs/guides/train_an_unconditional_base.py:graphs"
```

`build_graphs` returns the graphs, the SMILES that produced them (one for one) and how many molecules it skipped: an
element outside the nine, or a molecule that cannot be kekulized. A large `n_skipped` means that the vocabulary and your
data disagree. The first tenth of the graphs is set aside for validation. `domain` turns graphs back into molecules, and it
knows the training molecules, so that it can report novelty in step 5.

## 3. Build the model

The model infers its numbers of atom and bond classes, its distribution of graph sizes and its noise marginals from the
data, so it is built from a data loader. The settings below are the recorded ones (`E1_MODEL` in `defog_adapt.presets`),
with the size turned down so that it fits a small run: 4 layers of width 64 instead of 9 of width 256, and the molecular
features switched off.

```python
--8<-- "docs/guides/train_an_unconditional_base.py:model"
```

| Setting | What it does | Recorded |
|---|---|---|
| `n_layers` | The number of transformer layers. | 9 |
| `hidden_dim` | The width of the node representations. The edge and global ones are a quarter of it. | 256 |
| `hidden_mlp_dim` | The width of the MLPs that read the input and write the output. | 512 |
| `n_heads` | The attention heads in each layer. | 8 |
| `dropout` | The dropout probability. | 0.1 |
| `noise_type` | The noise that the denoising starts from. `marginal` draws each atom and bond class with its frequency in the data. | `marginal` |
| `extra_features_type` | The extra input features. `rrwp` adds relative random-walk features, which tell the model about the structure of the graph. | `rrwp` |
| `rrwp_steps` | How many random-walk steps those features use. | 20 |
| `molecular_features` | Valence and weight features for each atom. They need a valence and a weight for each element, so they are off here; the recorded base has them on. | `True` |
| `lambda_edge` | The weight of the bond loss relative to the atom loss. | 5 |
| `train_time_distortion` | How the training times are spread along the flow. `polydec` puts more of them near the clean end. | `polydec` |
| `lr` | The learning rate. | 2e-4 |
| `lr_scheduler` | How the learning rate falls over the training: along a cosine curve... | `cosine` |
| `lr_min` | ...down to this value. | 1e-6 |
| `weight_decay` | The weight decay of the optimiser. | 1e-5 |

Everything that is not listed stays at its default. The sampling settings are not part of the model: you choose them when
you sample, in step 5.

## 4. Train it

Training is an ordinary Lightning loop. The validation loss, measured every epoch on the molecules you set aside, decides
which epoch to keep: `BestValLossCheckpoint` saves the model whenever that loss reaches a new low, here as `base.ckpt`.
The script trains for 20 epochs, which is enough to watch the loss fall on a small file. The recorded base trained for 300
epochs on ZINC250k, at batch size 256.

```python
--8<-- "docs/guides/train_an_unconditional_base.py:train"
```

The script prints `best.best`, the lowest validation loss, and `best.best_epoch`, the epoch that reached it. `base.ckpt`
is the model of that epoch. If it is the last epoch, the loss was still falling: train for longer.

## 5. Look at what it makes (optional)

Sample from the base as the recorded runs were sampled, and measure the molecules. The four sampling settings are the
recorded ones (`E1_SAMPLING` in `defog_adapt.presets`).

```python
--8<-- "docs/guides/train_an_unconditional_base.py:look"
```

| Setting | What it does | Recorded |
|---|---|---|
| `sample_steps` | How many denoising steps take the sampler from noise to a graph. | 500 |
| `eta` | Stochasticity: how many extra random transitions the sampler adds while it denoises. They can undo an earlier choice. 0 switches them off. | 25 |
| `omega` | Target guidance: an extra pull toward the graph the model currently predicts. 0 switches it off. | 0 |
| `time_distortion` | How the steps are spaced in time. `polydec` takes smaller steps toward the end, when the graph is nearly clean. | `polydec` |

`domain.metrics` returns a dictionary. Read three entries first: `validity`, the fraction of the samples that decode to a
valid molecule; `uniqueness`, the fraction of the valid ones that are distinct; and `novelty`, the fraction of the distinct
ones that are not among the training molecules. `sanity` is stricter than `validity`: the molecule must also be connected
and have only rings of 3 to 8 atoms. A small base trained for a few epochs will not score well. What matters is that the
numbers rise as you give it more molecules and more epochs.

## 6. Clean it up with RL (optional)

The recorded pipeline cleans the base up before it trains adapters on it, with two rounds of structural reinforcement
learning. The reward is three yes-or-no checks on a generated molecule: it is valid, it is connected, and every ring has
between 3 and 8 atoms. An invalid molecule scores 0. In each iteration the model samples molecules, they are rewarded, and
one gradient step nudges the model toward the better ones, while a KL penalty holds it near the model the round started
from, so that it is less likely to collapse onto a narrow set of safe molecules. The result is a cleaner base, with fewer
disconnected fragments and odd rings.

A run can peak and then collapse, and the last iteration is not the best one. So the script takes the mean reward of the
last three iterations and saves the model whenever that mean reaches a new high. A round is 25 iterations, and round two
starts from the best model of round one.

```python
--8<-- "docs/guides/train_an_unconditional_base.py:clean"
```

| Setting | What it does | Recorded |
|---|---|---|
| `rollout_size` | How many molecules are sampled and rewarded in each iteration. | 128 |
| `sample_steps` | The denoising steps of those samples: fewer than in step 5, so that an iteration is cheap. | 100 |
| `eta` | The stochasticity of those samples: the exploration temperature. | 0 |
| `omega` | Target guidance. Off. | 0 |
| `time_distortion` | How the steps are spaced, as in step 5. | `polydec` |
| `subsample_steps` | How many noisy states of each sample enter the gradient. | 12 |
| `minibatch_size` | How many samples go through the network at once when the gradient is computed. It saves memory and does not change the result. | 16 |
| `lambda_edge` | The weight of the bond term relative to the atom term in the log-likelihood. | 1 |
| `reduction` | `sum` takes the joint log-likelihood of atoms and bonds. | `sum` |
| `advantage_mode` | How a reward becomes an advantage. `mean` subtracts the mean reward and does not rescale. | `mean` |
| `kl_coef` | How strongly the KL penalty pulls the model back toward the one the round started from. | 0.3 |
| `lr` | The learning rate. | 2e-5 |
| `weight_decay` | The weight decay of the optimiser. | 1e-5 |
| `grad_clip` | The largest gradient norm. | 1 |
| `ema_decay` | The saved weights are a moving average of the iterates with this decay. | 0.9 |

The script saves `base_rl1.ckpt` after round one and `base_clean.ckpt` after round two, and does not touch `base.ckpt`, so
you can always go back to the base before the cleanup. The next guides load `base.ckpt`: to use the cleaned base, change
that file name in their first step.

## Going further

- **The recorded base.** It is the same model at full size, on ZINC250k: 9 layers of width 256, the molecular features on
  (they need a valence and a weight for each of the nine elements), an exponential moving average of the weights, and 300
  epochs at batch size 256, which needs more than the 24 GB of a consumer GPU unless you accumulate gradients over
  micro-batches. A 300-epoch run does not fit in one cluster allocation either, so the recipe resumes from its last
  checkpoint across wall-clock windows.
- **The recipe.** `examples/training__e1.py` does all of that, carves its validation split from the reference training
  split, and evaluates at the end; `examples/rl_sanity__zinc.py` is the cleanup. See [Recipes](../reference/recipes.md).
- **Other elements.** The vocabulary is a decision you make once. To use other elements, declare your own atom types, in a
  fixed order, in step 2, and use the same ones in every later step: an adapter is tied to the vocabulary of its base.

## The final script

All the steps in one piece, for you to copy. It reads `molecules.smi` from the folder it runs in and writes `base.ckpt`
there. To leave out an optional step, delete its block.

```python
--8<-- "docs/guides/train_an_unconditional_base.py"
```
