"""Policy-gradient fine-tuning (GDPO) of a ZINC base against its structural failure modes.

    python examples/rl_sanity__zinc.py --BASE_CKPT "'ckpts/zinc_e1_seed42/best_model'" \\
        --OUT_CKPT_DIR "'ckpts/zinc_rl1_seed42'"
    # round two starts from round one's best checkpoint:
    python examples/rl_sanity__zinc.py --BASE_CKPT "'ckpts/zinc_rl1_seed42/best_model'" \\
        --OUT_CKPT_DIR "'ckpts/zinc_rl2_seed42'"

The reward is graded, ``r = valid + connected + rings_ok``, each 0 or 1. A single "all three" flag
would be the exact target, but 95% of samples already satisfy it, which leaves a near-constant
reward and almost no group-relative advantage to learn from. The graded form separates the failing
5% and tells a one-fault sample from a two-fault one. Invalid scores 0 rather than being
decomposed, which also makes invalid strictly worse than any valid molecule, so there is no
incentive to trade validity for either term.

What it did on the recorded ZINC model (test split, n = 10,000, 500 steps, eta 25), after TWO
rounds at this recipe's defaults (25 iterations each): validity (relaxed) 0.9929 -> 0.9959,
disconnected 0.0267 -> 0.0180, wonky rings 0.0173 -> 0.0133, FCD 1.401 -> 1.346, scaffold
similarity 0.6035 -> 0.5464. Two rounds is the useful depth: round one gives most of the gain and a
one-time scaffold cost, round two gives consistent further gains and recovers some of it, a third
moves nothing outside the noise. Budget about an hour of RL per base, not an open-ended schedule.
Round two is a ratchet, not a reset: the KL reference is rebuilt from the model it is handed, so
the pull is toward round one, not back toward the original base.

THE FAILURE THIS GUARDS AGAINST is reward hacking: sanity can be raised by collapsing onto a
narrow, safe region of chemical space, which wrecks the distribution while every targeted metric
improves. The defences are a KL penalty to the frozen base (``KL_COEF``) and SMILES dumped before
and after, so that FCD and scaffold similarity can be scored externally against
``validation_reference.smi``. A run whose sanity improves while FCD degrades is a FAILED run, and
the dumped files are what make that judgeable. The same reward on an AROMATIC MOSES base did
exactly that (validity 0.885 -> 0.939 while FCD went 0.863 -> 1.706 on four of four seeds); the
cause was the aromatic representation, not the reward, and on a kekulized base it does not happen.

NEVER RELY ON THE FINAL ITERATION. A run peaked around iteration 20 and collapsed by iteration 59,
and because only the last model was saved the good policy was unrecoverable. EMA does not rescue
that: at decay 0.9 its memory is about ten iterations, so the final EMA reflected the collapsed
region. Periodic checkpoints plus a best-so-far (selected on a ROLLING mean, since a single batch of
128 carries about 2% noise and would reward a lucky iteration) make the peak survivable.

The E1 checkpoint is never touched: this writes to ``OUT_CKPT_DIR`` and refuses to overwrite the
directory the base came from, so the base's own results stay reproducible. The RL model is a
separate artifact and a separate claim.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem, RDLogger

from defog_adapt.core import DeFoGModel, GDPOTrainer, SanityReward
from defog_adapt.data import vocabulary, zinc_reference
from defog_adapt.domains import MoleculeDomain
from defog_adapt.domains.molecule import ring_sizes_ok, validity_report
from defog_adapt.presets import SANITY_RL_TRAINER

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Parameters
# ============================================================================
# :param BASE_CKPT: the base to fine-tune (a checkpoint path, with or without ".ckpt"). Required.
BASE_CKPT: str = None
OUT_CKPT_DIR: str = "ckpts/zinc_rl_seed42"
DATA_ROOT: str = None
SEED: int = 42
RING_LO: int = 3
RING_HI: int = 8

# --- RL (defog_adapt.presets.SANITY_RL_TRAINER) ---
# :param ITERATIONS: one round. The recorded model used two (the second starts from the first's
#     best checkpoint). A run that rolled out at eta 25 peaked near iteration 20 and had collapsed
#     by iteration 59, so 25 brackets the optimum without entering the degrading region, and the
#     checkpoint selection below captures the peak even if it moves.
ITERATIONS: int = 25
ROLLOUT_SIZE: int = SANITY_RL_TRAINER["rollout_size"]
SUBSAMPLE_STEPS: int = SANITY_RL_TRAINER["subsample_steps"]
MINIBATCH_SIZE: int = SANITY_RL_TRAINER["minibatch_size"]
LAMBDA_EDGE: float = SANITY_RL_TRAINER["lambda_edge"]
REDUCTION: str = SANITY_RL_TRAINER["reduction"]
ADVANTAGE_MODE: str = SANITY_RL_TRAINER["advantage_mode"]   # Dr. GRPO: mean baseline, no std
LR: float = SANITY_RL_TRAINER["lr"]
WEIGHT_DECAY: float = SANITY_RL_TRAINER["weight_decay"]
GRAD_CLIP: float = SANITY_RL_TRAINER["grad_clip"]
EMA_DECAY: float = SANITY_RL_TRAINER["ema_decay"]
# :param KL_COEF: strength of the pull toward the frozen base. Non-zero on purpose: it is the
#     primary defence against optimising sanity by collapsing the distribution.
KL_COEF: float = SANITY_RL_TRAINER["kl_coef"]

# --- Rollout sampling ---
# Rollouts run at 100 steps and eta 0 for cost; evaluation uses the deployed 500 steps and eta 25.
# That mismatch is a known approximation, which is why evaluation is done at the deployed
# configuration. (Rolling out at the deployed eta 25 diverged: noisier rollouts raise the
# gradient variance.)
ROLLOUT_SAMPLE_STEPS: int = SANITY_RL_TRAINER["sample_steps"]
ROLLOUT_ETA: float = SANITY_RL_TRAINER["eta"]
ROLLOUT_OMEGA: float = SANITY_RL_TRAINER["omega"]
TIME_DISTORTION: str = SANITY_RL_TRAINER["time_distortion"]

# --- Checkpointing ---
CKPT_EVERY: int = 5
SELECT_BEST: bool = True
SELECT_WINDOW: int = 3

# --- Evaluation (at the deployed configuration) ---
EVAL_SAMPLES: int = 2048
EVAL_STEPS: int = 500
EVAL_ETA: float = 25.0
EVAL_OMEGA: float = 0.0

__DEBUG__: bool = False
__TESTING__: bool = False


@torch.no_grad()
def evaluate(model, atom_decoder, bond_decoder, n_samples, steps, eta, omega, device, train_ref,
             chunk=256):
    """Sample at the DEPLOY configuration and report every targeted metric, plus the SMILES.

    The SMILES matter as much as the numbers: FCD and scaffold similarity are computed externally,
    and they are what decide whether an improvement in sanity came at the cost of the distribution.
    """
    samples, remaining = [], n_samples
    while remaining > 0:
        k = min(chunk, remaining)
        samples += model.sample(num_samples=k, sample_steps=steps, eta=eta, omega=omega,
                                device=device, show_progress=False)
        remaining -= k
    report = validity_report(samples, atom_decoder, bond_decoder, reference_smiles=train_ref)
    smiles = report.pop("smiles")
    wonky = sum(1 for s in smiles
                if (m := Chem.MolFromSmiles(s)) is not None and not ring_sizes_ok(m))
    report["wonky_ring_frac"] = wonky / max(1, len(smiles))
    report["sanity"] = sum(
        1 for s in smiles
        if "." not in s and (m := Chem.MolFromSmiles(s)) is not None and ring_sizes_ok(m)
    ) / max(1, n_samples)
    return report, smiles


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    if not e.BASE_CKPT:
        raise ValueError("BASE_CKPT is required: the E1 base to fine-tune")
    e.log("GDPO structural-sanity RL on a ZINC base")
    torch.manual_seed(e.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.dirname(os.path.abspath(e.BASE_CKPT))
    if os.path.abspath(e.OUT_CKPT_DIR).rstrip("/") == base_dir.rstrip("/"):
        raise RuntimeError("OUT_CKPT_DIR is the base's own directory; refusing to overwrite it.")
    os.makedirs(e.OUT_CKPT_DIR, exist_ok=True)

    split = zinc_reference.load_reference_split(e.DATA_ROOT)
    train_ref = set(split.train_smiles)
    # Validation only. The test split is not read here: the base's test pass already happened, and
    # this model is a separate artifact whose test pass is its own, later, one-shot event.
    with open(os.path.join(e.path, "validation_reference.smi"), "w") as fh:
        fh.write("\n".join(split.val_smiles) + "\n")

    model = DeFoGModel.load(e.BASE_CKPT).to(device)
    e.log(f"loaded base {e.BASE_CKPT}")
    # Decoding with the wrong vocabulary does not raise: it yields plausible molecules made of the
    # wrong elements, which would then flow into the reward, the checkpoint selection and every
    # reported number. So check the base against the vocabulary it is about to be decoded with.
    atom_types, bond_types = list(zinc_reference.ATOM_TYPES), list(zinc_reference.BOND_TYPES)
    e.log(vocabulary.check_model(model, atom_types, bond_types, what=f"base {e.BASE_CKPT}"))
    from defog_adapt.domains.molecule import build_encoders

    _, atom_decoder, _, bond_decoder = build_encoders(atom_types, bond_types)
    domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_ref)

    # -- BEFORE -----------------------------------------------------------------------------
    model.eval()
    before, before_smiles = evaluate(model, atom_decoder, bond_decoder, e.EVAL_SAMPLES, e.EVAL_STEPS,
                                     e.EVAL_ETA, e.EVAL_OMEGA, device, train_ref)
    with open(os.path.join(e.path, "before.smi"), "w") as fh:
        fh.write("\n".join(before_smiles) + "\n")
    e.log(f"BEFORE  validity={before['validity_relaxed_largest_frag']:.4f} "
          f"disconnected={before['disconnected']:.4f} wonky_rings={before['wonky_ring_frac']:.4f} "
          f"sanity={before['sanity']:.4f} uniq={before['uniqueness']:.4f}")
    e["before"] = before

    # -- TRAIN ------------------------------------------------------------------------------
    reward = SanityReward(domain, e.RING_LO, e.RING_HI)
    trainer = GDPOTrainer(
        model, reward, rollout_size=e.ROLLOUT_SIZE, sample_steps=e.ROLLOUT_SAMPLE_STEPS,
        eta=e.ROLLOUT_ETA, omega=e.ROLLOUT_OMEGA, time_distortion=e.TIME_DISTORTION,
        subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH_SIZE,
        lambda_edge=e.LAMBDA_EDGE, reduction=e.REDUCTION, advantage_mode=e.ADVANTAGE_MODE,
        kl_coef=e.KL_COEF, lr=e.LR, weight_decay=e.WEIGHT_DECAY, grad_clip=e.GRAD_CLIP,
        ema_decay=e.EMA_DECAY, device=device, seed=e.SEED)
    e.log(f"training {e.ITERATIONS} iterations (kl_coef={e.KL_COEF}, lr={e.LR})")

    history, best = [], {"score": -float("inf"), "iter": None, "path": None}

    def on_iter(i, stats):
        row = {"iter": i, **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
               **reward.last}
        history.append(row)
        if e.CKPT_EVERY and i % e.CKPT_EVERY == 0:
            trainer.save(os.path.join(e.OUT_CKPT_DIR, f"iter{i:03d}"), use_ema=True)
        if e.SELECT_BEST:
            window = [h.get("sanity_frac", 0.0) for h in history[-e.SELECT_WINDOW:]]
            score = sum(window) / max(1, len(window))
            # Require a full window so the first iteration cannot win on a one-batch fluke.
            if len(window) >= e.SELECT_WINDOW and score > best["score"]:
                best.update(score=score, iter=i,
                            path=trainer.save(os.path.join(e.OUT_CKPT_DIR, "best_model"), use_ema=True))
                e.log(f"  it {i:3d} new best smoothed sanity_frac={score:.3f} -> best_model")
        if i % 5 == 0 or i == e.ITERATIONS - 1:
            e.log(f"  it {i:3d} reward={row.get('reward_mean', float('nan')):.3f} "
                  f"sanity={reward.last.get('sanity_frac', float('nan')):.3f} "
                  f"disc={reward.last.get('disconnected_frac', float('nan')):.3f} "
                  f"wonky={reward.last.get('wonky_ring_frac', float('nan')):.3f}")

    trainer.fit(e.ITERATIONS, on_iter=on_iter)
    final_path = trainer.save(os.path.join(e.OUT_CKPT_DIR, "rl_model_final"), use_ema=True)
    e.log(f"saved the final-iteration model -> {final_path}")
    e.commit_json("history.json", history)

    # Evaluate the SELECTED model, not the last one: the last iteration is only the best policy if
    # the run never degraded, which is not the common case.
    if e.SELECT_BEST and best["path"]:
        e.log(f"loading the best checkpoint (iteration {best['iter']}, smoothed sanity "
              f"{best['score']:.3f})")
        model = DeFoGModel.load(os.path.join(e.OUT_CKPT_DIR, "best_model")).to(device)
        out_path = best["path"]
    else:
        if trainer.ema is not None:
            trainer.ema.copy_to(model)
        out_path = final_path
    e["selected"] = {"iter": best["iter"], "smoothed_score": best["score"], "path": out_path}

    # -- AFTER ------------------------------------------------------------------------------
    model.eval()
    after, after_smiles = evaluate(model, atom_decoder, bond_decoder, e.EVAL_SAMPLES, e.EVAL_STEPS,
                                   e.EVAL_ETA, e.EVAL_OMEGA, device, train_ref)
    with open(os.path.join(e.path, "after.smi"), "w") as fh:
        fh.write("\n".join(after_smiles) + "\n")
    e["after"] = after

    e.log("=" * 62)
    e.log(f"{'metric':<28}{'before':>10}{'after':>10}{'delta':>12}")
    for key, label in [("validity_relaxed_largest_frag", "validity (relaxed)"),
                       ("disconnected", "disconnected"), ("wonky_ring_frac", "wonky rings"),
                       ("sanity", "sanity (all three)"), ("uniqueness", "uniqueness")]:
        e.log(f"{label:<28}{before[key]:>10.4f}{after[key]:>10.4f}{after[key] - before[key]:>+12.4f}")
    e.log("=" * 62)
    e.log("FCD and scaffold similarity are NOT computed here. Score before.smi and after.smi "
          "against validation_reference.smi: a sanity gain with a degraded FCD is a FAILED run.")
    e.commit_json("summary.json", {
        "base_ckpt": e.BASE_CKPT, "out_ckpt": out_path, "seed": e.SEED,
        "selected_iter": best["iter"], "selected_smoothed_score": best["score"],
        "final_iteration_ckpt": final_path,
        "reward": "graded: valid + connected + rings_ok in [0, 3]",
        "kl_coef": e.KL_COEF, "iterations": e.ITERATIONS, "lr": e.LR,
        "rollout": {"steps": e.ROLLOUT_SAMPLE_STEPS, "eta": e.ROLLOUT_ETA, "omega": e.ROLLOUT_OMEGA},
        "eval": {"steps": e.EVAL_STEPS, "eta": e.EVAL_ETA, "omega": e.EVAL_OMEGA,
                 "n": e.EVAL_SAMPLES, "scored_against": "validation"},
        "before": before, "after": after,
        "delta": {k: after[k] - before[k] for k in
                  ("validity_relaxed_largest_frag", "disconnected", "wonky_ring_frac", "sanity",
                   "uniqueness")},
    })

    if history:
        it = [h["iter"] for h in history]
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
        left.plot(it, [h.get("reward_mean", np.nan) for h in history], lw=1.5)
        left.set(xlabel="iteration", ylabel="mean reward (0-3)", title="reward")
        left.grid(alpha=0.3)
        for key, label in [("sanity_frac", "sanity"), ("disconnected_frac", "disconnected"),
                           ("wonky_ring_frac", "wonky rings")]:
            right.plot(it, [h.get(key, np.nan) for h in history], lw=1.5, label=label)
        right.set(xlabel="iteration", ylabel="fraction of rollout", title="targeted failure modes")
        right.legend(fontsize=8)
        right.grid(alpha=0.3)
        fig.tight_layout()
        e.commit_fig("rl_progress.png", fig)


@experiment.testing
def testing(e: Experiment):
    e.ITERATIONS = 4
    e.CKPT_EVERY = 2
    e.SELECT_WINDOW = 2
    e.ROLLOUT_SIZE = 8
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH_SIZE = 4
    e.ROLLOUT_SAMPLE_STEPS = 5
    e.EVAL_SAMPLES = 16
    e.EVAL_STEPS = 5


experiment.run_if_main()
