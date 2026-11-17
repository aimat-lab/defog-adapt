"""Unconditional DeFoG on ZINC250k or MOSES: the recipe behind the recorded base models.

    python examples/training__e1.py --DATASET "'zinc'"
    python examples/training__e1.py --DATASET "'moses'"
    python examples/training__e1.py --DATASET "'zinc'" --__TESTING__ True     # CPU, minutes

(PyComex takes Python literals on the command line, hence the quoted strings.)

What this fixes, and why it is not just "train on the dataset":

split            The GDSS/GruM reference split for ZINC (24,887 held-out test rows fixed by
                 ``valid_idx_zinc250k.json``) and the official MOSES train/test/test-scaffolds
                 split. Validation is carved out of TRAIN and is the only split that sampling
                 sweeps and checkpoint selection may see. The test splits are hashed and counted by
                 the loader and then not read here: they exist for one evaluation pass, later,
                 with a sampling configuration already chosen on validation.
representation   KEKULIZED, three bond types. An AROMATIC bond class is a promise about a whole
                 ring system that RDKit checks by kekulizing, and a per-edge model cannot keep it:
                 on the aromatic MOSES base 118 of 120 hard validity failures were kekulization
                 errors and exactly one was a valence error. Dropping the class makes that failure
                 impossible by construction (validity 0.884 -> 0.991, FCD 1.478 -> 1.126 across
                 four seeds). MOSES defaults to ``kekulized_v2`` here; ``aromatic_v1`` is kept only
                 so that older aromatic checkpoints stay reproducible.
vocabulary       Frozen, in a fixed order, not derived from whatever the data happens to contain.
                 A checkpoint decodes correctly only under the vocabulary it was trained with, and
                 decoding with the wrong one yields plausible molecules made of the wrong elements:
                 it does not raise. The model's channel counts are checked against the vocabulary.
validity         Three conventions reported side by side. The headline is the relaxed reading
                 (largest fragment) that the published rows use.
recipe           ``defog_adapt.presets.E1_MODEL``: 9 layers, width 256, relative random-walk
                 features, marginal noise, polydec time distortion, edge-loss weight 5, cosine
                 learning-rate schedule from 2e-4, EMA 0.9999. 300 epochs at batch 256 for ZINC,
                 100 for MOSES.

The diagnostic evaluation at the end samples at the DEPLOYED configuration (500 steps, eta 25), not
at the eta 0 the model stores as its default; that choice is what the recorded numbers used. It
reports validity, uniqueness and novelty only. FCD, scaffold similarity and the other benchmark
metrics need packages this library does not depend on and are computed separately.

Chaining across wall-clock windows. A 300-epoch run does not fit in one cluster allocation. With
``CKPT_DIR`` set, the full training state (weights, optimiser, LR schedule, EMA shadow, epoch) is
written to ``<CKPT_DIR>/last.ckpt`` and resumed from there, so each link re-runs the same command
and picks up where the last stopped. ``EPOCHS`` stays fixed across links because it is the cosine
horizon; ``MAX_TIME_HOURS`` bounds each link (per link, not cumulatively), and ``SKIP_FINAL_EVAL``
makes the links train only. Give every parallel arm its own ``CKPT_DIR`` or they resume from each
other's state.
"""
import hashlib
import os

import pytorch_lightning as pl
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import RDLogger

from defog_adapt.core import (
    BestValLossCheckpoint, DeFoGModel, EMACallback, PerLinkTimer, SampleVisualizationCallback,
    TrainingMonitorCallback,
)
from defog_adapt.data import moses_reference, zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import build_encoders, validity_report
from defog_adapt.presets import E1_MODEL, E1_SAMPLING

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Parameters
# ============================================================================
# :param DATASET: "zinc" or "moses".
DATASET: str = "zinc"
# :param REPRESENTATION: MOSES only. "kekulized_v2" (7 atom types, 3 bond types) is the recorded
#     model; "aromatic_v1" (8 atom types including a never-used H, aromatic bonds) reproduces older
#     checkpoints. Changing it changes the channel count, so a checkpoint decodes correctly only
#     under the representation it was trained with.
REPRESENTATION: str = "kekulized_v2"

# --- Reference data ---
# :param DATA_ROOT: where the dataset files live; None uses the library's per-user cache (or
#     $DEFOG_ADAPT_HOME). Files are checked against pinned SHA256s and a mismatch fails the run.
DATA_ROOT: str = None
# :param DOWNLOAD: fetch the files if they are missing. Off by default: the library never
#     downloads silently.
DOWNLOAD: bool = False
# :param ALLOW_HASH_MISMATCH: escape hatch for a changed upstream file. It defeats the point of
#     pinning, so a run with it set must say so in whatever it reports.
ALLOW_HASH_MISMATCH: bool = False
VAL_SIZE: int = 5000
SPLIT_SEED: int = 42
# :param GRAPH_CACHE_DIR: encoding 1.58M MOSES molecules takes a while, and a chained run would
#     repeat it on every link. Keyed by a hash of the SMILES and the vocabulary, so a stale cache
#     cannot survive a change to either. None disables it.
GRAPH_CACHE_DIR: str = None

# --- Model (defog_adapt.presets.E1_MODEL) ---
N_LAYERS: int = E1_MODEL["n_layers"]
HIDDEN_DIM: int = E1_MODEL["hidden_dim"]
HIDDEN_MLP_DIM: int = E1_MODEL["hidden_mlp_dim"]
N_HEADS: int = E1_MODEL["n_heads"]
DROPOUT: float = E1_MODEL["dropout"]
NOISE_TYPE: str = E1_MODEL["noise_type"]
EXTRA_FEATURES_TYPE: str = E1_MODEL["extra_features_type"]
RRWP_STEPS: int = E1_MODEL["rrwp_steps"]
MOLECULAR_FEATURES: bool = E1_MODEL["molecular_features"]

# --- Training ---
# :param EPOCHS: the cosine horizon; None means the recorded value for the dataset (ZINC 300,
#     MOSES 100).
EPOCHS: int = None
BATCH_SIZE: int = 256
LEARNING_RATE: float = E1_MODEL["lr"]
LR_SCHEDULER: str = E1_MODEL["lr_scheduler"]
LR_MIN: float = E1_MODEL["lr_min"]
WEIGHT_DECAY: float = E1_MODEL["weight_decay"]
LAMBDA_EDGE: float = E1_MODEL["lambda_edge"]
TRAIN_TIME_DISTORTION: str = E1_MODEL["train_time_distortion"]
EMA_DECAY: float = 0.9999
# :param NUM_WORKERS: data-loading workers; None means 4 for ZINC and 8 for MOSES.
NUM_WORKERS: int = None

# --- Chaining across wall-clock windows ---
CKPT_DIR: str = None
RESUME_CKPT: str = None
CKPT_EVERY_N_STEPS: int = 2000
MAX_TIME_HOURS: float = None

# --- Evaluation ---
# The model STORES eta 0 as its sampling default (that is what the recorded checkpoints hold);
# the diagnostic evaluation samples at the deployed configuration.
EVAL_STEPS: int = E1_SAMPLING["sample_steps"]
EVAL_ETA: float = E1_SAMPLING["eta"]
EVAL_OMEGA: float = E1_SAMPLING["omega"]
# :param NUM_EVAL_SAMPLES: None means 10,000 for ZINC and 30,000 for MOSES. FCD scales roughly as
#     1/n, so conventions are not interchangeable and the number must be reported with the result.
NUM_EVAL_SAMPLES: int = None
EVAL_CHUNK: int = 250
# :param SKIP_FINAL_EVAL: chained links set this: training only, evaluate once the chain is done.
SKIP_FINAL_EVAL: bool = False

SAMPLE_VIS_EVERY_K: int = 10
GEN_PROBE_EVERY_K: int = 10
GEN_PROBE_SAMPLES: int = 64
GEN_PROBE_STEPS: int = 100

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False

# ZINC valencies and atomic weights, for the molecular extra features. (MOSES carries its own.)
ZINC_ATOM_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1}
ZINC_ATOM_WEIGHT = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "P": 30.974, "S": 32.06,
                    "Cl": 35.45, "Br": 79.904, "I": 126.904}
ZINC_MAX_ATOM_WEIGHT = 500.0
RECORDED = {  # per dataset: epochs, loader workers, evaluation samples
    "zinc": dict(epochs=300, num_workers=4, n_eval=10_000),
    "moses": dict(epochs=100, num_workers=8, n_eval=30_000),
}


def _cache_file(cache_dir, smiles, atom_types, bond_types, kekulize):
    """Cache name covering the molecules, the vocabulary AND the kekulize flag: a kekulized and an
    aromatic encoding of the same molecules are different tensors, and reusing one for the other
    would train on silently mis-encoded data."""
    digest = hashlib.sha256("\n".join(smiles).encode()).hexdigest()
    key = "|".join([digest, ",".join(atom_types), ",".join(bond_types), f"kekulize={kekulize}"])
    return os.path.join(cache_dir, f"graphs_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pt")


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    if e.DATASET not in RECORDED:
        raise ValueError(f"DATASET must be one of {sorted(RECORDED)}, got {e.DATASET!r}")
    recorded = RECORDED[e.DATASET]
    epochs = e.EPOCHS or recorded["epochs"]
    num_workers = recorded["num_workers"] if e.NUM_WORKERS is None else e.NUM_WORKERS
    n_eval = e.NUM_EVAL_SAMPLES or recorded["n_eval"]
    e.log(f"{e.DATASET.upper()} UNCONDITIONAL -- E1 recipe ({epochs} epochs)")
    pl.seed_everything(e.SEED, workers=True)

    # -- Vocabulary + reference split ------------------------------------------------------
    if e.DATASET == "zinc":
        reference = zinc_reference
        atom_types, bond_types, kekulize = (list(zinc_reference.ATOM_TYPES),
                                            list(zinc_reference.BOND_TYPES), True)
        valency, weight, max_weight = ZINC_ATOM_VALENCY, ZINC_ATOM_WEIGHT, ZINC_MAX_ATOM_WEIGHT
        rep = None
    else:
        reference = moses_reference
        rep = moses_reference.get_representation(e.REPRESENTATION)
        atom_types, bond_types, kekulize = list(rep.atom_types), list(rep.bond_types), rep.kekulize
        valency, weight, max_weight = (moses_reference.ATOM_VALENCY, moses_reference.ATOM_WEIGHT,
                                       moses_reference.MAX_ATOM_WEIGHT)
        e.log(f"representation '{rep.name}': {len(atom_types)} atom types {atom_types}, "
              f"bonds {bond_types}, kekulize={kekulize}" + (f" ({rep.note})" if rep.note else ""))

    # Hard-fails on a hash or count mismatch, which is the intended behaviour: with published-only
    # baselines a drifted split shows up as an unexplained performance gap, never as an error.
    split = reference.load_reference_split(
        e.DATA_ROOT, val_size=e.VAL_SIZE, split_seed=e.SPLIT_SEED, download=e.DOWNLOAD,
        allow_hash_mismatch=e.ALLOW_HASH_MISMATCH)
    e.log(split.summary())
    e["provenance/split"] = split.provenance

    # -- Encode (the test splits are not touched) -------------------------------------------
    def encode(smiles, **kw):
        if e.DATASET == "zinc":
            return zinc_reference.build_graphs(smiles, atom_types=atom_types, bond_types=bond_types,
                                               kekulize=kekulize, **kw)
        return moses_reference.build_graphs(smiles, representation=rep, **kw)

    cache = None
    if e.GRAPH_CACHE_DIR:
        os.makedirs(e.GRAPH_CACHE_DIR, exist_ok=True)
        cache = _cache_file(e.GRAPH_CACHE_DIR, split.train_smiles, atom_types, bond_types, kekulize)
    if cache and os.path.exists(cache):
        e.log(f"loading encoded graphs from cache {os.path.basename(cache)}")
        blob = torch.load(cache, weights_only=False)
        train_graphs, train_smiles = blob["train_graphs"], blob["train_smiles"]
        val_graphs, n_skip_train, n_skip_val = blob["val_graphs"], blob["n_skip_train"], blob["n_skip_val"]
    else:
        train_graphs, train_smiles, n_skip_train = encode(split.train_smiles, progress=True)
        val_graphs, _, n_skip_val = encode(split.val_smiles)
        if cache:
            tmp = cache + ".partial"
            torch.save({"train_graphs": train_graphs, "train_smiles": train_smiles,
                        "val_graphs": val_graphs, "n_skip_train": n_skip_train,
                        "n_skip_val": n_skip_val}, tmp)
            os.replace(tmp, cache)                        # rename only when complete
    e.log(f"encoded train {len(train_graphs)} (skipped {n_skip_train}) | "
          f"val {len(val_graphs)} (skipped {n_skip_val})")
    e["provenance/encoding"] = {
        "dataset": e.DATASET, "atom_types": atom_types, "bond_types": bond_types,
        "kekulize": kekulize, "n_train_graphs": len(train_graphs), "n_val_graphs": len(val_graphs),
        "n_skipped_train": n_skip_train, "n_skipped_val": n_skip_val,
    }
    # A frozen vocabulary should fit the data exactly. More than a rounding error means the two
    # disagree and the model would be trained on a quietly filtered dataset.
    if n_skip_train > 0.001 * max(1, len(split.train_smiles)):
        e.log(f"WARNING: {n_skip_train} train molecules did not encode; expected about 0 under "
              f"the frozen {e.DATASET} vocabulary.")

    from torch_geometric.loader import DataLoader

    # persistent_workers stays off on purpose: a lingering worker pool is a plausible reason for a
    # failed arm to sit on a GPU instead of exiting, and the respawn cost is noise.
    train_loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, persistent_workers=False)
    val_loader = DataLoader(val_graphs, batch_size=e.BATCH_SIZE, num_workers=num_workers,
                            persistent_workers=False)

    # -- Model ------------------------------------------------------------------------------
    model = DeFoGModel.from_dataloader(
        train_loader, n_layers=e.N_LAYERS, hidden_dim=e.HIDDEN_DIM, hidden_mlp_dim=e.HIDDEN_MLP_DIM,
        n_heads=e.N_HEADS, dropout=e.DROPOUT, noise_type=e.NOISE_TYPE,
        extra_features_type=e.EXTRA_FEATURES_TYPE, rrwp_steps=e.RRWP_STEPS,
        molecular_features=e.MOLECULAR_FEATURES,
        atom_valencies=[valency[a] for a in atom_types], atom_weights=[weight[a] for a in atom_types],
        max_atom_weight=max_weight, lr=e.LEARNING_RATE, weight_decay=e.WEIGHT_DECAY,
        lambda_edge=e.LAMBDA_EDGE, train_time_distortion=e.TRAIN_TIME_DISTORTION,
        lr_scheduler=e.LR_SCHEDULER, lr_min=e.LR_MIN, sample_time_distortion=E1_MODEL["sample_time_distortion"],
        sample_steps=E1_MODEL["sample_steps"], eta=E1_MODEL["eta"], omega=E1_MODEL["omega"])
    # The model infers its channel counts from the data, so a mismatch means the encoded graphs do
    # not match the declared vocabulary. Nothing downstream would raise: it would decode to the
    # wrong atoms for the rest of the checkpoint's life. Assert it while it is cheap.
    if rep is not None and not rep.matches_model(model):
        raise RuntimeError(
            f"model classes {getattr(model, 'output_dims', {})} do not match representation "
            f"'{rep.name}' ({len(rep.atom_types)} atom types, {len(rep.bond_types)} bond types + "
            f"one no-bond class). A stale graph cache is the usual cause.")
    e["model/num_params"] = sum(p.numel() for p in model.parameters())
    e.log(f"model: {e['model/num_params']:,} parameters")
    # Node counts come from the training loader, so graph sizes are drawn from the training
    # distribution rather than fixed or uniform.
    e["provenance/size_distribution"] = {
        "type": type(model.default_size_dist).__name__,
        "source": "train split node-count histogram",
        "max_size": int(model.default_size_dist.max_size),
    }

    # -- Train ------------------------------------------------------------------------------
    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, bond_types)

    def probe_metrics(samples):
        report = validity_report(samples, atom_decoder, bond_decoder, reference_smiles=train_smiles)
        return {"validity": report["validity_relaxed_largest_frag"],
                "uniqueness": report["uniqueness"], "novelty": report.get("novelty", 0.0)}

    monitor = TrainingMonitorCallback(
        smoothing_window=5, figure_callback=lambda fig: e.track("training_progress", fig),
        generation_metrics_fn=probe_metrics, gen_every_k=e.GEN_PROBE_EVERY_K,
        gen_num_samples=e.GEN_PROBE_SAMPLES, gen_sample_steps=e.GEN_PROBE_STEPS,
        # No checkpoint_dir: selection is on val/loss, not on a 64-sample probe.
    )
    visual = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=e.SAMPLE_VIS_EVERY_K, sample_steps=e.GEN_PROBE_STEPS,
        domain=MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_smiles),
        figure_callback=lambda fig: e.track("samples", fig))
    # The best-val-loss model goes in CKPT_DIR, not the run archive: each chained link gets its own
    # archive, so leaving it there would scatter one run's best checkpoint across N directories.
    best_dir = e.CKPT_DIR if e.CKPT_DIR else e.path
    best_ckpt = BestValLossCheckpoint(checkpoint_dir=best_dir, monitor="val/loss")

    callbacks = [monitor, visual, best_ckpt]
    if e.EMA_DECAY and e.EMA_DECAY > 0:
        callbacks = [EMACallback(decay=e.EMA_DECAY)] + callbacks
        e.log(f"EMA enabled (decay={e.EMA_DECAY})")
    # PerLinkTimer, NOT Trainer(max_time=...): Lightning's Timer restores its elapsed time from the
    # checkpoint, which makes the budget cumulative over the chain and leaves later links nothing.
    if e.MAX_TIME_HOURS:
        hours = int(e.MAX_TIME_HOURS)
        minutes = int(round((e.MAX_TIME_HOURS - hours) * 60))
        callbacks.append(PerLinkTimer(duration={"hours": hours, "minutes": minutes}))
        e.log(f"PerLinkTimer = {hours}h{minutes:02d}m for THIS link (per link, not cumulative)")

    resume_path, enable_ckpt = e.RESUME_CKPT, False
    if e.CKPT_DIR:
        os.makedirs(e.CKPT_DIR, exist_ok=True)
        from pytorch_lightning.callbacks import ModelCheckpoint

        callbacks.append(ModelCheckpoint(dirpath=e.CKPT_DIR, save_last=True, save_top_k=0,
                                         every_n_train_steps=e.CKPT_EVERY_N_STEPS))
        enable_ckpt = True
        auto = os.path.join(e.CKPT_DIR, "last.ckpt")
        if resume_path is None and os.path.exists(auto):
            resume_path = auto
        e.log(f"resumable checkpointing -> {e.CKPT_DIR} (every {e.CKPT_EVERY_N_STEPS} steps); "
              + (f"RESUMING from {resume_path}" if resume_path else "fresh start"))

    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1, enable_progress_bar=True,
                         enable_checkpointing=enable_ckpt, logger=False, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume_path)
    e.log(f"saved final model to {model.save(os.path.join(e.path, 'model'))}")
    e["training/best_val_loss"] = best_ckpt.best
    e["training/best_val_epoch"] = best_ckpt.best_epoch
    e["training/epochs_completed"] = int(trainer.current_epoch)
    e["training/finished_horizon"] = int(trainer.current_epoch) >= epochs
    e.log(f"epochs completed: {trainer.current_epoch}/{epochs}")

    if e.SKIP_FINAL_EVAL:
        e.log("SKIP_FINAL_EVAL: training link only; evaluate once the chain has finished.")
        return

    # -- Diagnostic evaluation (at the deployed sampling configuration) -----------------------
    e.log("=" * 60)
    e.log(f"DIAGNOSTIC EVAL: {n_eval} samples, {e.EVAL_STEPS} steps, eta {e.EVAL_ETA}, "
          f"omega {e.EVAL_OMEGA}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if best_ckpt.saved_path and os.path.exists(best_ckpt.saved_path):
        e.log(f"loading the best-val-loss checkpoint (val/loss {best_ckpt.best:.5f}, "
              f"epoch {best_ckpt.best_epoch})")
        model = DeFoGModel.load(os.path.join(best_dir, "best_model"))
    model = model.to(device).eval()

    samples, remaining = [], n_eval
    while remaining > 0:
        k = min(e.EVAL_CHUNK, remaining)
        samples += model.sample(num_samples=k, sample_steps=e.EVAL_STEPS, eta=e.EVAL_ETA,
                                omega=e.EVAL_OMEGA, device=device, show_progress=False)
        remaining -= k
    report = validity_report(samples, atom_decoder, bond_decoder, reference_smiles=train_smiles)
    e.commit_json("generated_smiles.json", report.pop("smiles"))

    e.log(f"validity (relaxed, largest fragment) = {report['validity_relaxed_largest_frag']:.4f}"
          "   <- the headline, comparable to published rows")
    e.log(f"validity (strict, no correction)     = {report['validity_strict_largest_frag']:.4f}")
    e.log(f"validity (whole molecule)            = {report['validity_whole_molecule']:.4f}")
    e.log(f"per valid:  uniqueness={report['uniqueness']:.4f} novelty={report.get('novelty', float('nan')):.4f}")
    e.log(f"cumulative: V={report['v']:.4f} V.U.={report['v_u']:.4f} "
          f"V.U.N.={report.get('v_u_n', float('nan')):.4f}")
    e.commit_json("e1_report.json", {
        "metrics": report,
        "provenance": {
            "split": split.provenance, "encoding": e["provenance/encoding"],
            "size_distribution": e["provenance/size_distribution"],
            "sampling": {"steps": e.EVAL_STEPS, "eta": e.EVAL_ETA, "omega": e.EVAL_OMEGA,
                         "n_samples": len(samples), "seed": e.SEED},
            "recipe": {"epochs_horizon": epochs, "epochs_completed": e["training/epochs_completed"],
                       "batch_size": e.BATCH_SIZE, "lr": e.LEARNING_RATE, "ema_decay": e.EMA_DECAY},
            "not_computed": ["fcd", "nspdk", "scaffold_similarity"],
            "novelty_reference": "train split",
        },
    })
    e.log("Done. FCD and the other benchmark metrics are not computed here.")


@experiment.testing
def testing(e: Experiment):
    """Smoke configuration: proves the pipeline executes, on a few hundred molecules.

    The reference split is truncated AFTER it has been loaded and verified, so the hash and count
    checks still run for real: a smoke test that skipped them would not exercise the part most
    likely to break."""
    # Only shrink the horizon if the caller did not set one: the testing hook runs after the
    # command line is parsed, and an unconditional assignment would discard `--EPOCHS N`.
    if e.EPOCHS is None:
        e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.NUM_WORKERS = 0
    e.VAL_SIZE = 100
    e.EVAL_STEPS = 5
    e.GEN_PROBE_STEPS = 5
    e.GEN_PROBE_EVERY_K = 1
    e.GEN_PROBE_SAMPLES = 8
    e.NUM_EVAL_SAMPLES = 32
    e.EVAL_CHUNK = 8
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    e.GRAPH_CACHE_DIR = None                              # never cache a truncated dataset

    for module, split_cls in ((zinc_reference, zinc_reference.ZincReferenceSplit),
                              (moses_reference, moses_reference.MosesReferenceSplit)):
        real = module.load_reference_split

        def small(*args, _real=real, _cls=split_cls, **kwargs):
            split = _real(*args, **kwargs)
            fields = dict(train_smiles=split.train_smiles[:400], val_smiles=split.val_smiles[:100],
                          test_smiles=split.test_smiles,
                          provenance={**split.provenance, "SMOKE_TEST_TRUNCATED": True})
            if hasattr(split, "test_scaffolds_smiles"):
                fields["test_scaffolds_smiles"] = split.test_scaffolds_smiles
            return _cls(**fields)

        module.load_reference_split = small


experiment.run_if_main()
