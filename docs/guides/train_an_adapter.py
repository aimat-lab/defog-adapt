# Train an adapter, as one script. The guide "Train an adapter" walks through it step by
# step. It reads molecules.smi and base.ckpt from the folder it runs in and writes
# logp_adapter.ckpt, logp_head_size.ckpt and logp_head.ckpt there.

# --8<-- [start:graphs]
import random
from pathlib import Path

from defog_adapt.data import zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import build_encoders

smiles = Path("molecules.smi").read_text().split()
random.Random(0).shuffle(smiles)
graphs, kept, n_skipped = zinc_reference.build_graphs(
    smiles,
    atom_types=zinc_reference.ATOM_TYPES,
    bond_types=zinc_reference.BOND_TYPES,
    kekulize=True,
)

_, atom_dec, _, bond_dec = build_encoders(
    atom_types=zinc_reference.ATOM_TYPES,
    bond_types=zinc_reference.BOND_TYPES,
)
domain = MoleculeDomain(atom_dec, bond_dec)
# --8<-- [end:graphs]

# --8<-- [start:base]
from defog_adapt.core import DeFoGModel

base = DeFoGModel.load("base.ckpt")
# --8<-- [end:base]

# --8<-- [start:label]
import torch
from rdkit.Chem import Crippen

labelled = []
for graph in graphs:
    mol = domain.decode(graph)
    if mol is not None and graph.num_nodes <= base.max_nodes:
        graph.cond = torch.tensor([[Crippen.MolLogP(mol)]])
        labelled.append(graph)
labels = torch.cat([graph.cond for graph in labelled])
# --8<-- [end:label]

# --8<-- [start:build]
from defog_adapt.core import CrossAttentionAdapter

adapter = CrossAttentionAdapter.for_base(
    base,
    cond_dim=1,
    cond_mean=[float(labels.mean())],
    cond_std=[float(labels.std())],
    name="logp_adapter",
    cond_type="logp",
    hidden=256,
    time_conditioned=True,
    streams=("X", "E", "y"),
    cond_fourier=3,
    xattn_tokens=64,
    xattn_dim=128,
    xattn_heads=16,
)
# --8<-- [end:build]

# --8<-- [start:train]
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from defog_adapt.core import AdapterModule

pl.seed_everything(0)
loader = DataLoader(labelled, batch_size=24, shuffle=True)
module = AdapterModule(
    base,
    adapter,
    cond_attr="cond",
    lr=4e-4,
)
trainer = pl.Trainer(
    max_epochs=80,
    accelerator="auto",
    devices=1,
    gradient_clip_val=1.0,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(module, loader)
adapter.eval()
adapter.save("logp_adapter")
# --8<-- [end:train]

# --8<-- [start:check]
from defog_adapt.core import AdaptedSampler, AdapterComposition, ConditionBranch

for target in (labels.quantile(0.1).item(), labels.quantile(0.9).item()):
    branch = ConditionBranch(adapter, condition=torch.tensor([target]), weight=1.5)
    sampler = AdaptedSampler(
        base,
        AdapterComposition([branch], base=base),
        sample_steps=500,
        eta=25.0,
        omega=0.0,
        time_distortion="polydec",
    )
    mols = [domain.decode(graph) for graph in sampler.sample(100)]
    values = [Crippen.MolLogP(mol) for mol in mols if mol is not None]
    mean = sum(values) / max(len(values), 1)
    print(f"target {target:.2f}: {len(values)} of 100 valid, mean logP {mean:.2f}")
# --8<-- [end:check]

# --8<-- [start:size]
from defog_adapt.core import fit_size_model

size_model, metrics = fit_size_model(
    labels,
    [graph.num_nodes for graph in labelled],
    max_size=base.max_nodes,
    hidden=512,
    layers=2,
    epochs=200,
    lr=1e-3,
    batch_size=256,
    property_name="logp",
    property_from="decoded",
)
size_model.save("logp_head_size")
print(metrics["gain_nats"], metrics["shrink"])
# --8<-- [end:size]

# --8<-- [start:head]
from defog_adapt.core import PropertyHead, fit_property_head, to_dense

n_hold = len(labelled) // 10
hold_graphs, fit_graphs = labelled[:n_hold], labelled[n_hold:]
head = PropertyHead(
    len(zinc_reference.ATOM_TYPES),
    len(zinc_reference.BOND_TYPES) + 1,
    hid=128,
    layers=3,
    prop_mean=float(labels.mean()),
    prop_std=float(labels.std()),
)
fit_property_head(
    head,
    fit_graphs,
    epochs=60,
    lr=1e-3,
    batch_size=32,
    seed=0,
)
head.save("logp_head.ckpt")
# --8<-- [end:head]

# --8<-- [start:head_check]
head.eval()
batch = next(iter(DataLoader(hold_graphs, batch_size=len(hold_graphs))))
dense, mask = to_dense(
    batch.x,
    batch.edge_index,
    batch.edge_attr,
    batch.batch,
)
dense = dense.mask(mask)
with torch.no_grad():
    predicted = head.predict(dense.X, dense.E, mask)
error = (predicted.reshape(-1) - batch.cond.reshape(-1)).abs().mean()
print(error.item(), labels.std().item())
# --8<-- [end:head_check]
