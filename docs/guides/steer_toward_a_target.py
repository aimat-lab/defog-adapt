# Steer toward a target, as one script. The guide "Steer toward a target" walks through
# it step by step. It loads four checkpoints from the folder it runs in: base.ckpt,
# logp_adapter.ckpt, logp_head.ckpt and logp_head_size.ckpt.

# --8<-- [start:domain]
from defog_adapt.data import zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import build_encoders

atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(
    atom_types=zinc_reference.ATOM_TYPES,
    bond_types=zinc_reference.BOND_TYPES,
)
domain = MoleculeDomain(atom_dec, bond_dec)
# --8<-- [end:domain]

# --8<-- [start:load]
from defog_adapt.core import CrossAttentionAdapter, DeFoGModel

base = DeFoGModel.load("base.ckpt")
adapter = CrossAttentionAdapter.load("logp_adapter.ckpt")
adapter.check_compatible(base)
# --8<-- [end:load]

# --8<-- [start:target]
import torch

from defog_adapt.core import AdapterComposition, ConditionBranch

target = 4.0
branch = ConditionBranch(adapter, condition=torch.tensor([target]), weight=1.5)
composition = AdapterComposition([branch], base=base)
# --8<-- [end:target]

# --8<-- [start:adapter_run]
from defog_adapt.core import AdaptedSampler

sampler = AdaptedSampler(
    base,
    composition,
    sample_steps=500,
    eta=25.0,
    omega=0.0,
    time_distortion="polydec",
)
graphs = sampler.sample(10)
# --8<-- [end:adapter_run]

# --8<-- [start:size_run]
from defog_adapt.core import LearnedSizeDistribution

size = LearnedSizeDistribution.load("logp_head_size.ckpt")
condition = torch.full((10, 1), target)
graphs_sized = sampler.sample(10, size_dist=size, condition=condition)
# --8<-- [end:size_run]

# --8<-- [start:ruler]
from defog_adapt.core import LearnedPropertyEnergy, PropertyHead, StandardisedEnergy

head = PropertyHead.load("logp_head.ckpt")
std = float(head.prop_std)
prop_energy = LearnedPropertyEnergy(
    head,
    target,
    domain,
    atom_enc,
    bond_enc,
    scale=std,
)
energy = StandardisedEnergy([(prop_energy, std)])
# --8<-- [end:ruler]

# --8<-- [start:fk_run]
from defog_adapt.core import FeynmanKacSampler

fk_sampler = FeynmanKacSampler(
    base,
    energy,
    composition=composition,
    beta=20.0,
    warmup_frac=0.6,
    ess_frac=0.5,
    resample_scheme="residual",
    sample_steps=500,
    eta=25.0,
    omega=0.0,
    time_distortion="polydec",
)
graphs_fk = fk_sampler.sample(10, size_dist=size, condition=condition)
# --8<-- [end:fk_run]

# --8<-- [start:trace]
for record in fk_sampler.trace:
    print(
        record["step"],
        record["ess_over_k"],
        record["logw_range"],
        record["resampled"],
    )
# --8<-- [end:trace]

# --8<-- [start:judge_fn]
from rdkit import Chem
from rdkit.Chem import Crippen


def judge(generated, target):
    """Measure a run on its decoded molecules, with RDKit (not with the head)."""
    mols = [domain.decode(graph) for graph in generated]
    valid = [mol for mol in mols if mol is not None]
    smiles = {Chem.MolToSmiles(mol) for mol in valid}
    errors = [abs(Crippen.MolLogP(mol) - target) for mol in valid]
    return {
        "validity": len(valid) / len(generated),
        "uniqueness": len(smiles) / len(valid) if valid else float("nan"),
        "mean_abs_error": sum(errors) / len(errors) if errors else float("nan"),
    }
# --8<-- [end:judge_fn]

# --8<-- [start:judge_runs]
runs = [
    ("adapter only", graphs),
    ("+ size model", graphs_sized),
    ("+ Feynman-Kac", graphs_fk),
]
for name, run in runs:
    print(f"{name:14s}", judge(run, target))
# --8<-- [end:judge_runs]
