# defog-adapt

Frozen-base **adapters**, **Feynman-Kac steering** and **reinforcement-learning fine-tuning** for
[DeFoG](https://arxiv.org/abs/2410.04263) discrete flow matching on graphs, with **molecules** as the
validated domain.

> **Status: alpha (0.1).** Research code with a narrow, honest scope. It ships **no model weights** and **no
> evaluation harness**, and its numbers were measured in the research fork it was extracted from. Read
> [`docs/caveats.md`](docs/caveats.md) before relying on any of them.

DeFoG generates graphs with a discrete flow-matching model whose sampling rates are built at generation time.
This package adds the pieces that turn an unconditional molecular DeFoG model into one you can **steer**, and
keeps **only the features that have a recorded positive result**. Things that were tried and came out null,
negative or untried are not here (see [`docs/migration.md`](docs/migration.md#what-was-removed) for the list and
the reasons).

## What is in it

| | what it does | evidence |
|---|---|---|
| **Frozen-base adapters** (`AdaLNAdapter`, `AdapterModule`, `AdapterComposition`) | A small conditional adapter over a frozen unconditional base: swappable (one file per property) and composable (two properties multiply). Node cross-attention and Fourier bands on the condition. | logP, QED, TPSA and SAScore on ZINC250k; joint logP + QED; Morgan fingerprints |
| **Learned size model** (`LearnedSizeDistribution`) | Draw the node count from `P(n \| target)` instead of the dataset marginal. | -0.0103 logP and -0.0058 QED error |
| **Feynman-Kac steering** (`FeynmanKacSampler`, `StandardisedEnergy`, `PropertyHead`) | Inference-time particle steering toward a target, scored by a learned property head, with a dimensionless `beta`. | the largest single lever measured: logP 0.3132 to 0.2299 |
| **GDPO fine-tuning** (`GDPOTrainer`, `AdapterGDPOTrainer`) | Policy-gradient RL on the base (fewer fragments, odd rings) or on an adapter (tighter conditioning). | -13.6% logP and -19.5% QED with validity up; structural RL on ZINC |
| **Kekulized molecular representation** | Dropping the aromatic bond class makes a whole kind of validity failure impossible. | MOSES validity 0.884 to 0.991 |
| **Inpainting** (`InpaintingSampler`) | Freeze a substructure and grow the rest (RePaint-style). | *thin*: one demonstration |
| **Spectrum energy** (`SpectrumEnergy`) | Steer toward IR bands, with a predictor you supply. | *thin*: see the caveats |

## Install

```bash
pip install "defog-adapt[chem]"        # + RDKit: the molecular domain, which every recorded result uses
pip install "defog-adapt[chem,data]"   # + pandas: the reference-split loaders (ZINC250k, MOSES)
pip install "defog-adapt[recipes]"     # + PyComex: the runnable experiments in examples/
```

Python 3.10 or newer, PyTorch 2.1 or newer, PyTorch Geometric 2.3 or newer and PyTorch Lightning 2.4 or newer.
Install PyTorch for your platform first if you want a GPU build. The package is `defog_adapt`; it does not clash
with the unrelated `defog` package on PyPI.

## Try it

The shortest tour of every building block is a toy that runs on a CPU in tens of seconds, with no downloads
and no checkpoints ([`examples/quickstart_cpu.py`](examples/quickstart_cpu.py)):

```bash
git clone https://github.com/the16thpythonist/defog-adapt && cd defog-adapt
pip install -e ".[chem]" && python examples/quickstart_cpu.py
```

It trains a tiny base, an adapter, a property head and a size model, samples with and without Feynman-Kac
steering, and decodes and scores the molecules. **It is a wiring check, not a result.** The assembly with real
checkpoints is [`examples/steer_zinc.py`](examples/steer_zinc.py); the core of it is:

```python
import torch
from defog_adapt.core import (AdaLNAdapter, AdapterComposition, ConditionBranch, DeFoGModel,
                              FeynmanKacSampler, LearnedPropertyEnergy, LearnedSizeDistribution,
                              PropertyHead, StandardisedEnergy)
from defog_adapt.data import zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import build_encoders
from defog_adapt.presets import ADAPTER_SAMPLING, ADAPTER_WEIGHT, FK_STEERING

atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(zinc_reference.ATOM_TYPES, zinc_reference.BOND_TYPES)
domain = MoleculeDomain(atom_dec, bond_dec)

base = DeFoGModel.load("base.ckpt")
adapter = AdaLNAdapter.load("logp_adapter.ckpt")                     # one frozen base, one file per property
head = PropertyHead.load("logp_head.ckpt")                           # the Feynman-Kac "ruler"
size = LearnedSizeDistribution.load("logp_head_size.ckpt")           # P(n | target)

target, std = 4.0, float(head.prop_std)
composition = AdapterComposition(
    [ConditionBranch(adapter, torch.tensor([target]), ADAPTER_WEIGHT["logp"])], base=base)
energy = StandardisedEnergy([(LearnedPropertyEnergy(head, target, domain, atom_enc, bond_enc, scale=std), std)])
sampler = FeynmanKacSampler(base, energy, composition=composition, **FK_STEERING, **ADAPTER_SAMPLING)
graphs = sampler.sample(10, size_dist=size, condition=torch.full((10, 1), target))
```

## Results at a glance

ZINC250k, logP, the FreeGress protocol (**100 targets from real molecules, 10 generations each**; error between
the target and the decoded property; validation). Lower is better, and the run-to-run noise floor is about 0.011.
Each row adds to the one above it, except the last two.

| arm | logP error |
|---|---|
| the first adapter (FiLM only) | 0.5420 |
| cross-attention + Fourier adapter | 0.3250 |
| + learned node counts | 0.3132 |
| + Feynman-Kac (`beta` 20) | **0.2299** |
| *instead of FK:* + RL on the adapter | 0.2807 |
| *RL and FK together* | 0.2157 (not significantly better than FK alone) |

Validity stays at 0.97 or above in the recorded runs. Feynman-Kac costs diversity (uniqueness about 0.76, against
1.0 without it), and the RL gain does **not** add to it. Every number, its protocol and its uncertainty are in
[`docs/results.md`](docs/results.md); what they do *not* support is in [`docs/caveats.md`](docs/caveats.md).

## Documentation

| | |
|---|---|
| [`docs/adapters.md`](docs/adapters.md) | what an adapter is; training, composing, the label convention, the file format |
| [`docs/steering.md`](docs/steering.md) | Feynman-Kac, choosing `beta`, learned sizes, several properties, inpainting, spectra |
| [`docs/rl.md`](docs/rl.md) | structural RL and adapter RL; the operating-point and selection traps |
| [`docs/results.md`](docs/results.md) | the measured numbers, with protocols |
| [`docs/caveats.md`](docs/caveats.md) | provenance, noise, what is thin, what was removed |
| [`docs/migration.md`](docs/migration.md) | for users of the research fork (`defog`) |
| [`examples/README.md`](examples/README.md) | the recipes, in pipeline order |

## What this is not

* **Not a trained model.** No weights ship with 0.1; the recipes in `examples/` produce them.
* **Not the evaluation harness.** The conditional numbers follow a published protocol and were produced by a
  separate script; `examples/steer_zinc.py --evaluate` follows the same protocol shape but is not bit-reproducible
  against it. FCD and the other benchmark metrics need packages this library does not depend on.
* **Not a pipeline or a command-line tool.** It is a library plus recipes you read and adapt.
* **Not a reproduction of every figure.** The code is checked for equivalence with the research package (a
  golden-output test on real checkpoints) and by its unit tests; the numbers come from the research fork's logs.

## Security

Checkpoints are loaded with `torch.load(..., weights_only=False)` because they carry configuration dictionaries,
and **unpickling executes code**. Only load checkpoints you trust.

## Relationship to DeFoG, citation and license

The base model follows the upstream [DeFoG](https://github.com/qym7/DeFoG) implementation (Qin, Madeira,
Thanou, Frossard; ICML 2025). Everything built on top of it here was developed in a research fork. This package is
**not affiliated with or endorsed by the DeFoG authors**; see [`NOTICE`](NOTICE) for the full credits and the
papers behind each method. If you use it, please cite this package and the DeFoG paper ([`CITATION.cff`](CITATION.cff)).

MIT, see [`LICENSE`](LICENSE). The upstream copyright notice is retained.

## Development

```bash
pip install -e ".[all]"
pytest                                  # the unit suite, about half a minute on a CPU
DEFOG_ADAPT_HOME=... DEFOG_ADAPT_CKPTS=... pytest tests/test_recipes.py   # recipe smoke runs (opt-in)
```

`tests/parity` runs real checkpoints through the library and compares against golden outputs from the research
package; it is opt-in (it needs the checkpoints) and is the release gate. See [`CLAUDE.md`](CLAUDE.md) for the
layout and conventions.
