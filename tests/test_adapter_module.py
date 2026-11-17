"""`AdapterModule`: train ONLY the adapter, over a frozen base, with the base's own denoising loss.

This is the training path of every adapter in the package and it had no tests. What matters:

* the base is genuinely frozen and nothing but the adapter is optimised, so the shared
  unconditional path (and with it composability) survives training;
* the adapter checkpoint does not drag the base's weights along (they are ~10x bigger and the base
  is supplied separately at load time);
* the loss is the base's own, so a trained adapter is a conditional denoiser ``p(x1 | x_t, c)``,
  and it actually goes down when optimised.
"""

import pytest
import torch
from torch_geometric.loader import DataLoader

from defog_adapt.core import AdaLNAdapter, AdapterModule, DeFoGModel
from defog_adapt.domains.molecule import build_encoders, smiles_to_pyg_data

SMILES = ["CCO", "CCNC", "CC", "CNOCC", "OCC", "NCCO"]


def build(cond_attr="cond", n_layers=2):
    atom_enc, _, bond_enc, _ = build_encoders(["C", "N", "O"], ["SINGLE", "DOUBLE"])
    graphs = []
    for i, smi in enumerate(SMILES):
        g = smiles_to_pyg_data(smi, atom_enc, bond_enc)
        setattr(g, cond_attr, torch.tensor([float(g.x.shape[0])]))      # condition = node count
        graphs.append(g)
    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    base = DeFoGModel.from_dataloader(
        loader, n_layers=n_layers, hidden_dim=32, hidden_mlp_dim=64, n_heads=2, dropout=0.0,
        noise_type="marginal", extra_features_type="rrwp", rrwp_steps=3, molecular_features=False)
    base.eval()
    adapter = AdaLNAdapter.for_base(base, cond_dim=1, hidden=16, cond_mean=[3.5], cond_std=[1.0])
    return base, adapter, loader


def test_only_the_adapter_is_optimised_and_the_base_is_frozen():
    base, adapter, _ = build()
    module = AdapterModule(base, adapter)
    assert all(not p.requires_grad for p in base.parameters()), "the base must be frozen"
    (opt,) = [module.configure_optimizers()]
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in adapter.parameters()}, "the optimiser must hold exactly the adapter"
    assert not opt_ids & {id(p) for p in base.parameters()}


def test_the_learning_rate_reaches_the_optimiser():
    base, adapter, _ = build()
    opt = AdapterModule(base, adapter, lr=3e-3).configure_optimizers()
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-3)


def test_a_training_step_gives_the_adapter_gradients_and_the_base_none():
    base, adapter, loader = build()
    module = AdapterModule(base, adapter)
    torch.manual_seed(0)
    loss = module.training_step(next(iter(loader)), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in base.parameters()), "the frozen base received a gradient"
    got = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert got and any(float(g.abs().sum()) > 0 for g in got), "the adapter received no gradient"


def test_the_base_stays_in_eval_mode_even_when_the_module_is_training():
    """Dropout in the base would make the denoising target noisy in a way deployment does not."""
    base, adapter, loader = build()
    module = AdapterModule(base, adapter)
    module.train()
    base.train()                                   # something put it back; the step must undo that
    module.training_step(next(iter(loader)), 0)
    assert not base.training


def test_the_condition_is_read_from_the_named_attribute():
    base, adapter, loader = build(cond_attr="target")
    module = AdapterModule(base, adapter, cond_attr="target")
    torch.manual_seed(0)
    assert torch.isfinite(module.training_step(next(iter(loader)), 0))
    with pytest.raises(AttributeError):
        AdapterModule(base, adapter, cond_attr="cond").training_step(next(iter(loader)), 0)


def test_the_adapter_checkpoint_does_not_carry_the_frozen_base():
    base, adapter, _ = build()
    module = AdapterModule(base, adapter)
    ckpt = {"state_dict": module.state_dict()}
    assert any(k.startswith("base.") for k in ckpt["state_dict"]), "premise: the module holds the base"
    module.on_save_checkpoint(ckpt)
    keys = list(ckpt["state_dict"])
    assert keys and not any(k.startswith("base.") for k in keys)
    assert all(k.startswith("adapter.") for k in keys)


def test_optimising_one_batch_lowers_the_loss():
    """With the noise pinned, repeated steps on one batch must reduce the base's own denoising
    loss: the adapter is learning something, through the path the trainer uses."""
    base, adapter, loader = build()
    module = AdapterModule(base, adapter, lr=2e-2)
    batch = next(iter(loader))
    opt = module.configure_optimizers()
    losses = []
    for _ in range(40):
        torch.manual_seed(123)                     # same noise level and same noised graphs
        opt.zero_grad()
        loss = module.training_step(batch, 0)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.9 * losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_it_trains_under_a_lightning_trainer(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    base, adapter, loader = build()
    module = AdapterModule(base, adapter, lr=1e-2)
    before = [p.detach().clone() for p in adapter.parameters()]
    trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_progress_bar=False,
                         enable_model_summary=False, default_root_dir=str(tmp_path),
                         enable_checkpointing=False)
    trainer.fit(module, loader)
    assert any(not torch.equal(b, p) for b, p in zip(before, adapter.parameters())), \
        "fit() did not move the adapter"
    assert all(not p.requires_grad for p in base.parameters())
