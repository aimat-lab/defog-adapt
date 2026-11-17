"""Parity cases: one run of each validated code path, returned as plain JSON data.

The same cases run against two packages, selected by name:

* ``defog``        the research repo's package, at the commit this port was taken from. Its
                   output is committed as ``golden/golden.json``.
* ``defog_adapt``  this package. The parity tests require it to reproduce the golden output.

So these cases may only use API that exists in BOTH packages, and everything that could change
under a default being re-tuned (learning rates, EMA decay, KL strength, ...) is passed explicitly.

Determinism: single-threaded CPU, float32, every stochastic call preceded by ``manual_seed``.
Real checkpoints are required; they are not distributed with the package (point
``DEFOG_ADAPT_CKPTS`` at a directory holding the files listed in :data:`CKPT_FILES`).

Run through ``generate_golden.py`` (to create the golden file) or ``test_parity.py`` (to check).
"""
from __future__ import annotations

import hashlib
import importlib
import platform
import time
from pathlib import Path

import numpy as np
import torch

#: logical name -> path relative to the checkpoint directory
CKPT_FILES = {
    "base": "zinc_kek_base.ckpt",
    "adapter": "logp_xattn_adapter.ckpt",
    "head": "heads_kek/logp_head.ckpt",
    "size": "heads_kek/logp_head_size.ckpt",
}

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _single_thread() -> None:
    """Reduction order in multi-threaded kernels can depend on the thread count."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:          # already fixed earlier in this process
        pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(t: torch.Tensor) -> dict:
    """Bit-exact hash plus tolerance-comparable statistics of a tensor."""
    a = t.detach().cpu().contiguous().to(torch.float32)
    d = a.double()
    flat = a.reshape(-1)
    return {
        "shape": list(a.shape),
        "sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
        "sum": float(d.sum()),
        "abs_sum": float(d.abs().sum()),
        "min": float(d.min()),
        "max": float(d.max()),
        "head": [round(float(x), 7) for x in flat[:6].tolist()],
    }


def state_digest(sd: dict) -> dict:
    """One digest over a whole state dict (sorted keys, float32 bytes)."""
    h = hashlib.sha256()
    total = 0.0
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous().to(torch.float32)
        h.update(k.encode())
        h.update(v.numpy().tobytes())
        total += float(v.double().abs().sum())
    return {"n_tensors": len(sd), "sha256": h.hexdigest(), "abs_sum": total}


class Ctx:
    """Everything the cases share, resolved from the package name."""

    def __init__(self, pkg: str, ckpt_dir):
        _single_thread()
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        self.Chem = Chem
        self.pkg = pkg
        self.dir = Path(ckpt_dir)
        self.core = importlib.import_module(f"{pkg}.core")
        self.mol = importlib.import_module(f"{pkg}.domains.molecule")
        zref = importlib.import_module(f"{pkg}.data.zinc_reference")
        self.atom_types = list(zref.ATOM_TYPES)
        self.bond_types = list(zref.BOND_TYPES)
        (self.atom_enc, self.atom_dec,
         self.bond_enc, self.bond_dec) = self.mol.build_encoders(self.atom_types, self.bond_types)
        self.domain = self.mol.MoleculeDomain(self.atom_dec, self.bond_dec)

    # -- checkpoints (loaded fresh per case: RL mutates the adapter) --------
    def path(self, key: str) -> Path:
        return self.dir / CKPT_FILES[key]

    def load_base(self):
        return self.core.DeFoGModel.load(str(self.path("base"))).to(CPU).eval()

    def load_adapter(self):
        base_adapter = self.core.AdaLNAdapter.load(str(self.path("adapter")), device="cpu")
        return base_adapter.eval()

    # -- decoding -----------------------------------------------------------
    def smiles(self, sample):
        mol = self.mol.pyg_data_to_mol(sample, self.atom_dec, self.bond_dec)
        smi = self.mol.mol_to_smiles(mol) if mol is not None else None
        if smi is not None and self.Chem.MolFromSmiles(smi) is None:
            return None
        return smi

    def summarise(self, samples) -> dict:
        from rdkit.Chem import Crippen
        smis = [self.smiles(s) for s in samples]
        logp = []
        for smi in smis:
            m = self.Chem.MolFromSmiles(smi) if smi else None
            logp.append(round(float(Crippen.MolLogP(m)), 6) if m is not None else None)
        return {
            "smiles": smis,
            "logp": logp,
            "n_valid": sum(s is not None for s in smis),
            "n_nodes": [int(s.x.shape[0]) for s in samples],
        }


class TargetedSize:
    """A size distribution with one target pinned in (see scripts/eval_adapter_ckpt.py of the
    research repo for why: the target rides in the adapter branch, not in the sampler's own
    ``condition``, so the size model would otherwise silently fall back to its marginal)."""

    def __init__(self, model, target: float, n: int):
        self.model = model
        self.condition = torch.full((max(int(n), 1), model.cond_dim), float(target))

    def sample(self, num_samples, condition=None, device=None, generator=None):
        c = (self.condition[:num_samples] if self.condition.size(0) >= num_samples
             else self.condition[:1].expand(num_samples, -1))
        return self.model.sample(num_samples, condition=c, device=device, generator=generator)

    @property
    def max_size(self) -> int:
        return self.model.max_size

    def log_prob(self, sizes, condition=None):
        return self.model.log_prob(sizes, condition=self.condition[:sizes.numel()])


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def case_base_logits(ctx: Ctx) -> dict:
    """One forward pass of the base model on a seeded noisy state: the finest-grained check."""
    base = ctx.load_base()
    torch.manual_seed(1234)
    sizes = torch.tensor([14, 18, 22, 25, 20, 16, 12, 28])
    node_mask, n_nodes, X, E, y, _y_raw, _cfg = base._prepare_generation(
        8, sizes, None, None, base.guidance_scale, CPU)
    noisy = {"X_t": X, "E_t": E, "y_t": y, "t": torch.full((8, 1), 0.5), "node_mask": node_mask}
    extra = base._compute_extra_data(noisy)
    with torch.no_grad():
        pred = base.forward(noisy, extra, node_mask)
    return {"n_nodes": n_nodes.tolist(), "X": digest(pred.X), "E": digest(pred.E)}


def case_base_sampling(ctx: Ctx) -> dict:
    base = ctx.load_base()
    sampler = ctx.core.Sampler(base, eta=25.0, omega=0.0, sample_steps=40,
                               time_distortion="polydec")
    torch.manual_seed(1)
    return ctx.summarise(sampler.sample(12, device=CPU, show_progress=False))


def case_adapted_sampling(ctx: Ctx) -> dict:
    core = ctx.core
    base, adapter = ctx.load_base(), ctx.load_adapter()
    adapter.check_compatible(base)
    comp = core.AdapterComposition(
        [core.ConditionBranch(adapter, torch.tensor([3.0]), 1.5)], base=base, mode="product")
    sampler = core.AdaptedSampler(base, comp, eta=25.0, omega=0.0, sample_steps=40,
                                  time_distortion="polydec")
    torch.manual_seed(2)
    return ctx.summarise(sampler.sample(12, device=CPU, show_progress=False))


def case_fk_sampling(ctx: Ctx) -> dict:
    """Adapter proposal + Feynman-Kac with a learned head energy and a learned size model."""
    core = ctx.core
    base, adapter = ctx.load_base(), ctx.load_adapter()
    head = core.PropertyHead.load(str(ctx.path("head")), device="cpu")
    size_model = core.LearnedSizeDistribution.load(str(ctx.path("size")))
    target = 3.0
    energy = core.LearnedPropertyEnergy(head, target, ctx.domain, ctx.atom_enc, ctx.bond_enc)
    comp = core.AdapterComposition(
        [core.ConditionBranch(adapter, torch.tensor([target]), 1.5)], base=base, mode="product")
    sampler = core.FeynmanKacSampler(
        base, energy, beta=2.0, composition=comp, eta=25.0, omega=0.0, sample_steps=40,
        time_distortion="polydec")
    torch.manual_seed(3)
    out = sampler.sample(8, size_dist=TargetedSize(size_model, target, 8), device=CPU,
                         show_progress=False)
    return ctx.summarise(out)


def case_gdpo_adapter_step(ctx: Ctx) -> dict:
    """Two RL iterations of the adapter trainer: rollout, reward, KL, update, EMA."""
    from rdkit.Chem import Crippen
    core = ctx.core
    base, adapter = ctx.load_base(), ctx.load_adapter()
    adapter.train()
    before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    reward = core.WeightedSanityPropertyReward(
        ctx.atom_dec, ctx.bond_dec, lambda m: float(Crippen.MolLogP(m)), scale=1.2)
    trainer = core.AdapterGDPOTrainer(
        base, adapter, reward,
        # everything that a default could change is explicit
        kl_coef=0.2, rollout_weight=1.5, rollout_mode="product", crn=True,
        lr=1e-4, weight_decay=1e-5, ema_decay=0.9,
        rollout_size=8, sample_steps=10, eta=25.0, omega=0.0, time_distortion="polydec",
        condition_sampler=core.make_condition_sampler(0.0, 5.0, 8, 2, 3),
        subsample_steps=4, minibatch_size=8, lambda_edge=1.0, reduction="mean",
        advantage_mode="mean", advantage_clip=5.0, advantage_eps=1e-4,
        grad_clip=1.0, device=CPU, seed=0)
    torch.manual_seed(7)
    history = trainer.fit(2)
    keep = ("loss", "kl", "grad_norm", "reward_mean", "reward_std", "adv_std", "pos_frac")
    after = adapter.state_dict()
    delta = sum(float((after[k].detach().double() - before[k].double()).pow(2).sum())
                for k in before) ** 0.5
    return {
        "history": [{k: round(float(m[k]), 7) for k in keep} for m in history],
        "adapter": state_digest(after),
        "delta_l2": delta,
    }


def case_inpainting(ctx: Ctx) -> dict:
    """Freeze a benzene core and grow 8 atoms around it, plain and with time-travel."""
    core = ctx.core
    base = ctx.load_base()
    mol = ctx.Chem.MolFromSmiles("c1ccccc1")
    ctx.Chem.Kekulize(mol, clearAromaticFlags=True)     # the base is a kekulized vocabulary
    X_core, E_core = ctx.domain.encode_subgraph(mol, range(6))
    constraint = core.SubgraphConstraint(X_core, E_core)
    out = {"core_classes": X_core.argmax(-1).tolist()}
    for name, kw in (("plain", {}),
                     ("time_travel", {"resample": True, "n_resample": 2, "jump_length": 3})):
        sampler = core.InpaintingSampler(base, constraint, eta=25.0, sample_steps=30,
                                         time_distortion="polydec", **kw)
        torch.manual_seed(5)
        samples = sampler.sample(4, n_free=8, device=CPU, show_progress=False)
        res = ctx.summarise(samples)
        # the frozen core must be exactly the benzene atoms, in slots 0..5
        res["core_kept"] = [bool((s.x[:6].argmax(-1) == X_core.argmax(-1)).all()) for s in samples]
        out[name] = res
    return out


def _synthetic_spectrum(mols, grid):
    """A deterministic stand-in for a learned IR predictor: Gaussians whose heights count
    functional groups. The package takes ANY black-box predictor; this one needs no data."""
    from rdkit import Chem
    carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
    hydroxyl = Chem.MolFromSmarts("[OX2H]")
    nitrile = Chem.MolFromSmarts("C#N")

    def g(c, w):
        return np.exp(-0.5 * ((grid - c) / w) ** 2)

    out = np.zeros((len(mols), len(grid)))
    for i, m in enumerate(mols):
        out[i] = (0.4 * m.GetNumHeavyAtoms() * g(2900.0, 80.0)
                  + 3.0 * len(m.GetSubstructMatches(carbonyl)) * g(1710.0, 20.0)
                  + 2.0 * len(m.GetSubstructMatches(hydroxyl)) * g(3400.0, 60.0)
                  + 2.5 * len(m.GetSubstructMatches(nitrile)) * g(2230.0, 15.0) + 0.05)
    return out


def case_spectrum_fk(ctx: Ctx) -> dict:
    core = ctx.core
    base = ctx.load_base()
    grid = np.linspace(500.0, 4000.0, 701)
    energy = core.SpectrumEnergy(
        ctx.domain, lambda mols: _synthetic_spectrum(mols, grid), grid,
        bands=[core.Band(1710.0, 30.0, target=0.07)])
    sampler = core.FeynmanKacSampler(base, energy, beta=2.5, eta=25.0, omega=0.0,
                                     sample_steps=40, time_distortion="polydec")
    torch.manual_seed(9)
    res = ctx.summarise(sampler.sample(8, device=CPU, show_progress=False))
    carbonyl = ctx.Chem.MolFromSmarts("[CX3]=[OX1]")
    res["n_carbonyl"] = [
        (len(ctx.Chem.MolFromSmiles(s).GetSubstructMatches(carbonyl)) if s else None)
        for s in res["smiles"]]
    return res


CASES = {
    "base_logits": case_base_logits,
    "base_sampling": case_base_sampling,
    "adapted_sampling": case_adapted_sampling,
    "fk_sampling": case_fk_sampling,
    "gdpo_adapter_step": case_gdpo_adapter_step,
    "inpainting": case_inpainting,
    "spectrum_fk": case_spectrum_fk,
}


def environment(ctx: Ctx) -> dict:
    import rdkit
    return {
        "package": ctx.pkg,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "threads": 1,
        "ckpt_sha256": {k: sha256_file(ctx.path(k)) for k in CKPT_FILES},
    }


def run_cases(ctx: Ctx, names=None, log=print) -> dict:
    """Run cases one after another in this process."""
    out = {}
    for name in (names or CASES):
        t0 = time.time()
        out[name] = CASES[name](ctx)
        log(f"  {name:<20s} {time.time() - t0:6.1f}s")
    return out


def _worker(job):
    pkg, ckpt_dir, name = job
    t0 = time.time()
    return name, CASES[name](Ctx(pkg, ckpt_dir)), time.time() - t0


def run_parallel(pkg: str, ckpt_dir, names=None, jobs: int = 7, log=print) -> dict:
    """Run each case in its own process. Every process is single-threaded, so the result of
    a case does not depend on how many run side by side."""
    import multiprocessing as mp
    names = list(names or CASES)
    out = {}
    with mp.get_context("spawn").Pool(min(jobs, len(names))) as pool:
        work = [(pkg, str(ckpt_dir), n) for n in names]
        for name, res, dt in pool.imap_unordered(_worker, work):
            out[name] = res
            log(f"  {name:<20s} {dt:6.1f}s")
    return {n: out[n] for n in names}
