# Train an unconditional base, as one script. The guide "Train an unconditional base"
# walks through it step by step. It reads molecules.smi (one SMILES per line) from the
# folder it runs in and writes base.ckpt there.

# --8<-- [start:data]
import random
from pathlib import Path

smiles = Path("molecules.smi").read_text().split()
random.Random(0).shuffle(smiles)
# --8<-- [end:data]

# --8<-- [start:graphs]
from rdkit import Chem

from defog_adapt.data import zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import build_encoders

graphs, kept, n_skipped = zinc_reference.build_graphs(
    smiles,
    atom_types=zinc_reference.ATOM_TYPES,
    bond_types=zinc_reference.BOND_TYPES,
    kekulize=True,
)
n_val = len(graphs) // 10
val_graphs, train_graphs = graphs[:n_val], graphs[n_val:]

_, atom_dec, _, bond_dec = build_encoders(
    atom_types=zinc_reference.ATOM_TYPES,
    bond_types=zinc_reference.BOND_TYPES,
)
known = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in kept[n_val:]}
domain = MoleculeDomain(atom_dec, bond_dec, reference_smiles=known)
# --8<-- [end:graphs]

# --8<-- [start:model]
from torch_geometric.loader import DataLoader

from defog_adapt.core import DeFoGModel

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64)

model = DeFoGModel.from_dataloader(
    train_loader,
    n_layers=4,
    hidden_dim=64,
    hidden_mlp_dim=128,
    n_heads=4,
    dropout=0.1,
    noise_type="marginal",
    extra_features_type="rrwp",
    rrwp_steps=20,
    molecular_features=False,
    lambda_edge=5.0,
    train_time_distortion="polydec",
    lr=2e-4,
    lr_scheduler="cosine",
    lr_min=1e-6,
    weight_decay=1e-5,
)
# --8<-- [end:model]

# --8<-- [start:train]
import pytorch_lightning as pl

from defog_adapt.core import BestValLossCheckpoint

pl.seed_everything(0)
best = BestValLossCheckpoint(checkpoint_dir=".", monitor="val/loss", filename="base")
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="auto",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    callbacks=[best],
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(best.best, best.best_epoch)
# --8<-- [end:train]

# --8<-- [start:look]
from defog_adapt.core import Sampler

model = DeFoGModel.load("base.ckpt")
sampler = Sampler(
    model,
    sample_steps=500,
    eta=25.0,
    omega=0.0,
    time_distortion="polydec",
)
generated = sampler.sample(100)
print(domain.metrics(generated, compute_kl=False))
# --8<-- [end:look]

# --8<-- [start:clean]
from defog_adapt.core import GDPOTrainer, SanityReward


def clean_up(model, name):
    """One round of RL. The best iteration (by a rolling mean) is saved as name."""
    reward = SanityReward(domain, 3, 8)
    trainer = GDPOTrainer(
        model,
        reward,
        rollout_size=128,
        sample_steps=100,
        eta=0.0,
        omega=0.0,
        time_distortion="polydec",
        subsample_steps=12,
        minibatch_size=16,
        lambda_edge=1.0,
        reduction="sum",
        advantage_mode="mean",
        kl_coef=0.3,
        lr=2e-5,
        weight_decay=1e-5,
        grad_clip=1.0,
        ema_decay=0.9,
    )
    scores, best = [], [-1.0]

    def keep_the_best(iteration, stats):
        scores.append(reward.last["sanity_frac"])
        rolling = sum(scores[-3:]) / len(scores[-3:])
        if len(scores) >= 3 and rolling > best[0]:
            best[0] = rolling
            trainer.save(name, use_ema=True)

    trainer.fit(25, on_iter=keep_the_best)
    return DeFoGModel.load(name + ".ckpt")


model = DeFoGModel.load("base.ckpt")
model = clean_up(model, "base_rl1")
model = clean_up(model, "base_clean")
# --8<-- [end:clean]
