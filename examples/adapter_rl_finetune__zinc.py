"""Fine-tune a trained property adapter by policy gradient (GDPO) to TIGHTEN its conditioning.

    python examples/adapter_rl_finetune__zinc.py --BASE_CKPT "'ckpts/zinc_rl2_seed42/best_model'" \\
        --ADAPTER_CKPT "'runs/logp_adapter.ckpt'" --PROPERTY "'logp'" \\
        --HEAD_CKPT "'ckpts/heads/logp_head.ckpt'" --SIZE_MODEL_CKPT "'ckpts/heads/logp_head_size.ckpt'"

The base stays frozen and only the adapter's parameters move, so composability is preserved. The
reward is a conditional property match, ``-|property(molecule) - target|``, with targets sampled
AMORTISED over the property's whole range; the advantage is grouped by target; and a KL term pulls
the policy back toward the pre-RL adapter, which is the guard against reward hacking.

What it did (E2 protocol, 100 validation targets x 10 generations, target-paired; the adapter on its
own best weight, with the learned size draw; four seeds, six hours each; the head as the reward):
logP error 0.3132 -> 0.2807 (-10%, p = 0.019, 0.3250 -> 0.2807 against the marginal-size starting
point, -13.6%) and QED 0.0782 -> 0.0698 (-10.7%, 0.0867 -> 0.0698, -19.5%), with validity rising
(0.982 -> 0.989 and 0.979 -> 0.996) and uniqueness unchanged at 1.0. Disconnected fragments on QED fell
from 12.5% to 3.5%, which the error does not show. Two seeds of the same arm do not differ (p = 0.95
and 0.89), so the magnitude is not seed luck.

Read the limits as carefully as the gain:

* The RL gain does NOT add to Feynman-Kac. On top of FK the increment is not significant for either
  of the two RL seeds (logP -0.0084 and -0.0146, p = 0.66 and 0.32; QED -0.0028 and -0.0010, p = 0.69
  and 0.69), where the same RL was worth -0.0324 at p = 0.019 without it. They correct the same
  error, so only one of them is paid for. All four point estimates favour RL; "indistinguishable
  from zero" is the honest claim, not "zero".
* One round is the useful depth. A second round of head-reward RL on QED was a significant REGRESSION
  on the 100-target protocol (+0.0114 and +0.0198, p = 0.004 and 0.00002) while its own internal
  evaluation claimed the opposite (the probe scored two fixed targets, and best-of-five selection
  overfit them), and a second round of weighted-reward RL on QED bought nothing.
* It does not reach the low end of logP: with an RDKit reward it tightened HIGH-logP targets on all
  four seeds (-0.06) and not LOW-logP ones.
* A head-reward run on SAScore regressed in its second round, and the head said so too.

THE OPERATING POINT MUST BE THE DEPLOYMENT POINT. Three knobs each had a default that quietly
differed from deployment, and all three were wrong at once in earlier runs, so that RL improved a
policy nobody runs. (1) ``ROLLOUT_WEIGHT`` is the guidance weight the rollouts are generated at AND
the gradient is scored at; it must be the weight the adapter is deployed at (1.5 for logP, TPSA and
SAScore, 2.0 for QED), which is why it defaults to the property's measured optimum and not to 1.
(2) ``SIZE_MODEL_CKPT`` draws rollout sizes from ``P(n | target)``; deploying with a learned size draw
but training under the marginal optimises against a distribution you do not use (rollouts averaged
+4.5 heavy atoms over the evaluation: macrocycles and fragments). (3) ``ETA`` is the evaluation
stochasticity (25), a different quantity from the exploration stochasticity of the rollouts
(``ROLLOUT_ETA``, 1.0, a swept winner under common random numbers). When an RL run and its benchmark
disagree, check that they are the same policy before anything else.

The early-stop probe scores 9 FIXED targets across the 5th-95th percentiles
(``PROBE_PERCENTILES``), not the two extremes the report uses, and ALWAYS computes the RDKit truth
on the molecules it selects on, logging ``sel_mae`` beside ``true_mae`` so that a divergence between
what the head likes and what is true shows up during the run, not in an evaluation days later. A
second best, chosen on the truth, is saved alongside. With a head reward the truth-chosen snapshot
beat the head-chosen one by 0.0084 (p = 0.043) on QED, where the head's ranking had inverted.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem, RDLogger

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    label_graphs, property_fn, require_matching_vocabulary, sample_in_chunks, zinc_vocabulary,
)

from defog_adapt.core import (  # noqa: E402
    CrossAttentionAdapter, AdaptedSampler, AdapterComposition, AdapterGDPOTrainer, ConditionBranch,
    DeFoGModel, HeadPropertyMatchReward, LearnedSizeDistribution, PropertyHead,
    PropertyMatchReward, Sampler, WeightedSanityPropertyReward, head_predict_batch,
    make_condition_sampler,
)
from defog_adapt.data import zinc_reference  # noqa: E402
from defog_adapt.domains.molecule import build_encoders, mol_to_smiles, pyg_data_to_mol  # noqa: E402
from defog_adapt.presets import ADAPTER_RL_TRAINER, ADAPTER_SAMPLING, ADAPTER_WEIGHT  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Parameters
# ============================================================================
# :param BASE_CKPT / ADAPTER_CKPT: the frozen base and the trained adapter to fine-tune. Required.
BASE_CKPT: str = None
ADAPTER_CKPT: str = None
PROPERTY: str = "logp"            # logp | tpsa | qed | sascore
# :param PROPERTY_FROM: which molecule the RL TARGETS describe. It must match what the adapter being
#     fine-tuned was trained on, or RL optimises toward a target scale the adapter does not speak
#     (a 1.6-log-unit gap at the low end of logP). Every recorded adapter used "decoded".
PROPERTY_FROM: str = "decoded"
DATA_ROOT: str = None

# --- Reward source ---
# :param REWARD_SOURCE: "head" (a learned PropertyHead scores the property term; the early-stop probe
#     and the best-snapshot selection use it too, while the evaluation reports RDKit truth as well,
#     to validate that optimising the head moved the real property) or "rdkit" (the closed-form
#     function). The confirmed runs used the head.
REWARD_SOURCE: str = "head"
HEAD_CKPT: str = None             # required for REWARD_SOURCE="head"
# :param SIZE_MODEL_CKPT: a LearnedSizeDistribution to draw rollout sizes from (empty = the dataset
#     marginal). Set it when the adapter will be DEPLOYED with a learned size draw: the size prior is
#     part of the environment the policy acts in.
SIZE_MODEL_CKPT: str = ""
# :param EVAL_SIZE_MODEL: draw evaluation and probe sizes from SIZE_MODEL_CKPT as deployment does,
#     instead of the marginal. Needs SIZE_MODEL_CKPT.
EVAL_SIZE_MODEL: bool = True

# :param REWARD_SHAPE: "tiered" (connectivity FIRST: invalid -10 < disconnected -4 < connected in
#     [-3, 0], so ANY connected molecule outranks ANY disconnected one and sanity vetoes targeting;
#     the head-reward runs used this) or "weighted" (RDKit reward only; targeting and sanity are each
#     normalised to [0, 1] and weighted 3:1, so targeting can outrank sanity. On QED that gave -26% in
#     the internal evaluation without costing sanity, but the same configuration on a second seed gave
#     only -9.8%: the direction is safe, the magnitude is seed-sensitive).
REWARD_SHAPE: str = "tiered"
INVALID_REWARD: float = -10.0
DISCONNECT_REWARD: float = -4.0
PROP_CLAMP: float = 3.0
W_PROP: float = 3.0
W_SANITY: float = 1.0
PROP_SPAN: float = 3.0            # weighted: targeting reaches 0 at PROP_SPAN standard deviations off target

# --- RL (defog_adapt.presets.ADAPTER_RL_TRAINER) ---
MAX_TIME_HOURS: float = 6.0       # the RL loop's wall budget (evaluations happen outside it)
MAX_ITERS: int = 100000           # an unreached cap; the time bounds the loop
ROLLOUT_SIZE: int = ADAPTER_RL_TRAINER["rollout_size"]        # K trajectories per iteration
N_GROUPS: int = 8                 # distinct targets per iteration (K / N_GROUPS rollouts each)
CRN: bool = ADAPTER_RL_TRAINER["crn"]   # common random numbers: shared start noise and size per group
ROLLOUT_STEPS: int = ADAPTER_RL_TRAINER["sample_steps"]
ROLLOUT_ETA: float = ADAPTER_RL_TRAINER["eta"]
ROLLOUT_OMEGA: float = ADAPTER_RL_TRAINER["omega"]
# :param ROLLOUT_WEIGHT: the guidance weight the rollouts are generated at and the gradient is scored
#     at; None means this property's deployment weight (presets.ADAPTER_WEIGHT).
ROLLOUT_WEIGHT: float = None
TIME_DISTORTION: str = ADAPTER_RL_TRAINER["time_distortion"]
SUBSAMPLE_STEPS: int = ADAPTER_RL_TRAINER["subsample_steps"]
MINIBATCH: int = ADAPTER_RL_TRAINER["minibatch_size"]
LR: float = ADAPTER_RL_TRAINER["lr"]
KL_COEF: float = ADAPTER_RL_TRAINER["kl_coef"]            # to the PRE-RL adapter
EMA_DECAY: float = ADAPTER_RL_TRAINER["ema_decay"]        # fast: it tracks the live weights
GRAD_CLIP: float = ADAPTER_RL_TRAINER["grad_clip"]
LAMBDA_EDGE: float = ADAPTER_RL_TRAINER["lambda_edge"]
TARGET_RANGE_PCT: list = [1, 99]  # amortised RL targets are sampled uniformly over this range
LOG_EVERY: int = 10

# --- Early stopping: keep the best EMA snapshot by a periodic, validity-gated probe ---
EARLY_STOP: bool = True
PROBE_EVERY: int = 40
PROBE_N_PER: int = 32             # samples per probe target
# :param PROBE_PERCENTILES: the targets the probe scores. Deliberately SEPARATE from
#     TARGET_PERCENTILES (which sizes the pre/post report) and deliberately WIDER, and FIXED across
#     probes so that probe error stays comparable and "keep the best snapshot" means something.
PROBE_PERCENTILES: list = [5, 16, 27, 38, 50, 61, 72, 83, 95]
# :param SAVE_PROBE_CKPTS: write the EMA weights at EVERY probe, not just the selected best. Worth it
#     whenever the selection metric is a proxy: it is the only way to ask afterwards whether the
#     truth-selected snapshot would have been better.
SAVE_PROBE_CKPTS: bool = False
VALIDITY_FLOOR_MARGIN: float = 0.05   # a snapshot must keep validity >= (pre-RL - this) to be eligible

# --- Evaluation (pre and post RL) ---
EVAL_STEPS: int = ADAPTER_SAMPLING["sample_steps"]
ETA: float = ADAPTER_SAMPLING["eta"]
OMEGA: float = ADAPTER_SAMPLING["omega"]
# :param GUIDANCE_WEIGHTS: None means [ROLLOUT_WEIGHT]: the optimised policy is evaluated at the
#     weight it was optimised and is deployed at.
GUIDANCE_WEIGHTS: list = None
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
N_PER_TARGET: int = 128
N_BASELINE: int = 256
EVAL_CHUNK: int = 32
EVAL_SEED: int = 1234             # PAIRED evaluation: one fixed rng so pre, post and probe sample the same draws
REF_SUBSAMPLE: int = 20000

SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ============================================================================ evaluation helpers
def gen_eval(samples, atom_decoder, bond_decoder):
    """``(connected valid RDKit Mols, n_total, n_valid, n_disconnected)``.

    The Mols are the SMILES-reparsed CONNECTED molecules, so that one generation can be scored under
    both RDKit and the head."""
    mols, n_valid, n_disc = [], 0, 0
    for graph in samples:
        mol = pyg_data_to_mol(graph, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        parsed = Chem.MolFromSmiles(smi) if smi is not None else None
        if parsed is None:
            continue
        n_valid += 1
        if "." in smi:
            n_disc += 1
            continue
        mols.append(parsed)
    return mols, len(samples), n_valid, n_disc


def rdkit_stats(mols, target, prop_fn):
    """``(error to target, mean, n)`` of the property over the molecules; skips failures."""
    values = []
    for mol in mols:
        try:
            v = float(prop_fn(mol))
        except Exception:                                  # noqa: BLE001
            continue
        if v == v:
            values.append(v)
    if not values:
        return None, None, 0
    a = np.asarray(values)
    return float(np.mean(np.abs(a - target))), float(a.mean()), len(a)


def steer_eval(base, adapter, atom_dec, bond_dec, prop_fn, targets, weights, args, device,
               head_fn=None, size_dist=None):
    """Per (level, weight): the error to the target (RDKit truth), disconnection and validity -- and,
    with ``head_fn``, ALSO the head's error (the selection metric). ``args.EVAL_SEED`` fixes the rng
    (saved and restored) so that pre and post evaluations sample the SAME draws: the delta is then the
    adapter, not sampling noise."""
    state = torch.get_rng_state()
    torch.manual_seed(args.EVAL_SEED)
    ctmc = dict(eta=args.ETA, omega=args.OMEGA, sample_steps=args.EVAL_STEPS,
                time_distortion=args.TIME_DISTORTION)
    base_mols, _, n_valid, n_disc = gen_eval(
        sample_in_chunks(Sampler(base, **ctmc), args.N_BASELINE, args.EVAL_CHUNK, device),
        atom_dec, bond_dec)
    out = {"baseline_mean": rdkit_stats(base_mols, 0.0, prop_fn)[1],
           "baseline_disc": (n_disc / n_valid) if n_valid else None, "levels": {}}
    for level, target in targets.items():
        out["levels"][level] = {"target": target, "per_w": {}}
        for w in weights:
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([target]), w)],
                                      base=base, mode="product")
            # The condition rides along only when a size model needs it; the base itself is
            # unconditional and ignores it either way.
            condition = torch.tensor([[target]], dtype=torch.float32) if size_dist is not None else None
            samples = sample_in_chunks(AdaptedSampler(base, comp, **ctmc), args.N_PER_TARGET,
                                       args.EVAL_CHUNK, device, size_dist=size_dist,
                                       condition=condition)
            mols, total, valid, disc = gen_eval(samples, atom_dec, bond_dec)
            mae, mean, n_conn = rdkit_stats(mols, target, prop_fn)
            rec = {"n": n_conn, "valid": (valid / total) if total else None,
                   "disc": (disc / valid) if valid else None, "mean": mean, "mae": mae}
            if head_fn is not None:
                preds = [v for v in head_fn(mols) if v is not None]
                if preds:
                    a = np.asarray(preds, dtype=float)
                    rec["head_mae"], rec["head_mean"] = float(np.mean(np.abs(a - target))), float(a.mean())
                else:
                    rec["head_mae"] = rec["head_mean"] = None
            out["levels"][level]["per_w"][str(w)] = rec
    torch.set_rng_state(state)
    return out


def format_eval(ev, weights):
    parts = []
    for level, d in ev["levels"].items():
        for w in weights:
            r = d["per_w"][str(w)]
            disc = f"{r['disc'] * 100:.0f}%" if r["disc"] is not None else "?"
            valid = f"{r['valid'] * 100:.0f}%" if r.get("valid") is not None else "?"
            head = f" head_mae={r['head_mae']}" if "head_mae" in r else ""
            parts.append(f"{level}->{d['target']:.1f}@w{w}: mae={r['mae']}{head} disc={disc} valid={valid}")
    return " | ".join(parts)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    for name in ("BASE_CKPT", "ADAPTER_CKPT"):
        if not getattr(e, name):
            raise ValueError(f"{name} is required")
    if e.REWARD_SOURCE not in ("head", "rdkit"):
        raise ValueError(f"REWARD_SOURCE must be 'head' or 'rdkit', got {e.REWARD_SOURCE!r}")
    if e.REWARD_SOURCE == "head" and not e.HEAD_CKPT:
        raise ValueError("REWARD_SOURCE='head' needs HEAD_CKPT (see examples/fit_property_head__zinc.py), "
                         "or use REWARD_SOURCE='rdkit'")
    if e.EVAL_SIZE_MODEL and not e.SIZE_MODEL_CKPT:
        e.log("EVAL_SIZE_MODEL needs SIZE_MODEL_CKPT; evaluating with the dataset marginal")
    e.log(f"adapter RL fine-tune (GDPO), property={e.PROPERTY}, kl_coef={e.KL_COEF}")
    import pytorch_lightning as pl

    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prop_fn = property_fn(e.PROPERTY)
    rollout_weight = e.ROLLOUT_WEIGHT if e.ROLLOUT_WEIGHT is not None else ADAPTER_WEIGHT[e.PROPERTY]
    weights = list(e.GUIDANCE_WEIGHTS) if e.GUIDANCE_WEIGHTS else [rollout_weight]
    e.log(f"operating point: rollout/scored/probe weight {rollout_weight}, evaluation weights {weights}, "
          f"rollout eta {e.ROLLOUT_ETA}, evaluation eta {e.ETA}")

    atom_types, bond_types, kekulize = zinc_vocabulary()
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(atom_types, bond_types)

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0, "expected an unconditional base"
    adapter = CrossAttentionAdapter.load(e.ADAPTER_CKPT, device=device)
    adapter.check_compatible(base)
    e.log(f"loaded the {e.PROPERTY} adapter from {e.ADAPTER_CKPT} (cond_dim={adapter.cond_dim})")

    # -- The property distribution: RL target range + evaluation targets ----------------------
    # Under the adapter's own label convention (property_from), or every target the policy ever
    # sees is shifted relative to what the adapter was trained on.
    pool = list(zinc_reference.load_reference_split(e.DATA_ROOT).train_smiles)
    rng = np.random.default_rng(e.SEED)
    sub = [pool[int(i)] for i in rng.permutation(len(pool))[: min(e.REF_SUBSAMPLE, len(pool))]]
    _, vals, n_skipped = label_graphs(
        sub, prop_fn, atom_encoder=atom_enc, bond_encoder=bond_enc, atom_decoder=atom_dec,
        bond_decoder=bond_dec, kekulize=kekulize, property_from=e.PROPERTY_FROM, keep_graphs=False)
    require_matching_vocabulary(n_skipped, len(sub), "reference molecules")
    e.log(f"reference values: {len(vals)} molecules ({n_skipped} skipped), {e.PROPERTY} "
          f"mean={vals.mean():.4f} std={vals.std():.4f}")
    p_lo, p_hi = [float(x) for x in np.percentile(vals, e.TARGET_RANGE_PCT)]
    prop_std = float(vals.std()) or 1.0
    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    probe_targets = {f"p{int(q)}": float(x)
                     for q, x in zip(e.PROBE_PERCENTILES, np.percentile(vals, e.PROBE_PERCENTILES))}
    e["eval/targets"], e["eval/probe_targets"] = targets, probe_targets
    e.log(f"RL target range [{p_lo:.2f}, {p_hi:.2f}]; evaluation targets {targets}; prop_std={prop_std:.3f}")

    # -- Reward --------------------------------------------------------------------------------
    head_fn = None
    if e.REWARD_SOURCE == "head":
        head = PropertyHead.load(e.HEAD_CKPT, device=device)
        head_fn = lambda mols: head_predict_batch(mols, head, atom_enc, bond_enc, device)  # noqa: E731
        reward = HeadPropertyMatchReward(
            head, atom_enc, bond_enc, atom_dec, bond_dec, device, scale=prop_std,
            invalid_reward=e.INVALID_REWARD, disconnect_reward=e.DISCONNECT_REWARD,
            prop_clamp=e.PROP_CLAMP)
        e.log(f"reward: head from {e.HEAD_CKPT}; the probe and the selection use the HEAD, the "
              f"evaluation reports RDKit truth too")
    elif e.REWARD_SHAPE == "weighted":
        reward = WeightedSanityPropertyReward(atom_dec, bond_dec, prop_fn, scale=prop_std,
                                              w_prop=e.W_PROP, w_sanity=e.W_SANITY,
                                              prop_span=e.PROP_SPAN)
        e.log(f"reward: weighted, {e.W_PROP}*targeting + {e.W_SANITY}*sanity, targeting reaching 0 "
              f"at {e.PROP_SPAN * prop_std:.3f} off target")
    else:
        reward = PropertyMatchReward(atom_dec, bond_dec, prop_fn, scale=prop_std,
                                     invalid_reward=e.INVALID_REWARD,
                                     disconnect_reward=e.DISCONNECT_REWARD, prop_clamp=e.PROP_CLAMP)
    cond_sampler = make_condition_sampler(p_lo, p_hi, e.ROLLOUT_SIZE, e.N_GROUPS, e.SEED)

    # -- Sizes: P(n | target) when a size model is supplied, else the marginal ----------------
    size_dist = None
    if e.SIZE_MODEL_CKPT:
        size_dist = LearnedSizeDistribution.load(e.SIZE_MODEL_CKPT, device=device)
        if getattr(size_dist, "property_from", None) not in (None, "", e.PROPERTY_FROM):
            raise RuntimeError(
                f"the size model was fit on property_from={size_dist.property_from!r} but this run "
                f"uses {e.PROPERTY_FROM!r}; the two disagree about what a target means")
        e.log(f"rollout sizes: P(n | target) from {e.SIZE_MODEL_CKPT}")
    else:
        e.log("rollout sizes: the dataset marginal (no size model supplied)")
    eval_size_dist = size_dist if e.EVAL_SIZE_MODEL else None
    e.log(f"evaluation and probe sizes: {'P(n | target)' if eval_size_dist else 'the dataset marginal'}")

    def eval_now(tag):
        ev = steer_eval(base, adapter, atom_dec, bond_dec, prop_fn, targets, weights, e, device,
                        head_fn=head_fn, size_dist=eval_size_dist)
        e.log(f"[{tag}] baseline={ev['baseline_mean']}  {format_eval(ev, weights)}")
        return ev

    e.log("=== PRE-RL evaluation ===")
    pre_ev = eval_now("pre-RL")
    # Snapshot the input weights so we can prove afterwards that RL actually moved them: never
    # ship an unchanged adapter as a "result".
    input_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}

    trainer = AdapterGDPOTrainer(
        base, adapter, reward, rollout_weight=rollout_weight, kl_coef=e.KL_COEF, lr=e.LR,
        ema_decay=e.EMA_DECAY, rollout_size=e.ROLLOUT_SIZE, sample_steps=e.ROLLOUT_STEPS,
        eta=e.ROLLOUT_ETA, omega=e.ROLLOUT_OMEGA, time_distortion=e.TIME_DISTORTION,
        condition_sampler=cond_sampler, subsample_steps=e.SUBSAMPLE_STEPS,
        minibatch_size=e.MINIBATCH, lambda_edge=e.LAMBDA_EDGE, crn=e.CRN, size_dist=size_dist,
        grad_clip=e.GRAD_CLIP, seed=e.SEED, device=device)
    e.log(f"=== RL: max_time={e.MAX_TIME_HOURS}h, K={e.ROLLOUT_SIZE} over {e.N_GROUPS} targets "
          f"(crn={e.CRN}), {e.ROLLOUT_STEPS} steps, lr={e.LR}, kl={e.KL_COEF}, early_stop={e.EARLY_STOP} ===")

    # -- Early-stop bookkeeping: the best validity-gated EMA snapshot by probe error ---------
    pre_valid_floor = 0.0
    if e.EARLY_STOP:
        valids = [pre_ev["levels"][level]["per_w"][str(weights[0])]["valid"] or 0.0
                  for level in e.LEVEL_NAMES]
        pre_valid_floor = max(0.0, min(valids) - e.VALIDITY_FLOOR_MARGIN)
    best = {"mae": float("inf"), "state": None, "iter": -1, "true_mae": float("nan")}
    # A SECOND best, chosen on RDKit truth rather than the selection metric. Free when the two agree
    # (an RDKit reward makes them identical), and the whole point when they do not: it turns "the
    # head picked a worse snapshot" from a postmortem into a checkpoint you can ship.
    best_true = {"mae": float("inf"), "state": None, "iter": -1, "sel_mae": float("nan")}

    def probe_eval():
        """PAIRED evaluation at the DEPLOYMENT policy (rollout weight, EVAL_STEPS, the deployment size
        draw, EMA weights) over the wide, fixed ``probe_targets``. Returns ``(selection error,
        minimum validity, truth error)``; the truth is computed on the same molecules the selection
        is, so a divergence is visible as it happens."""
        state = torch.get_rng_state()
        torch.manual_seed(e.EVAL_SEED)
        selection, truth, valids = [], [], []
        ctmc = dict(eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                    time_distortion=e.TIME_DISTORTION)
        for target in probe_targets.values():
            comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([target]), rollout_weight)],
                                      base=base, mode="product")
            condition = (torch.tensor([[target]], dtype=torch.float32)
                         if eval_size_dist is not None else None)
            samples = sample_in_chunks(AdaptedSampler(base, comp, **ctmc), e.PROBE_N_PER, e.EVAL_CHUNK,
                                       device, size_dist=eval_size_dist, condition=condition)
            mols, total, valid, _ = gen_eval(samples, atom_dec, bond_dec)
            valids.append((valid / total) if total else 0.0)
            if not mols:
                continue
            mae, _, _ = rdkit_stats(mols, target, prop_fn)
            if mae is not None:
                truth.append(mae)
            if head_fn is not None:
                preds = [v for v in head_fn(mols) if v is not None]
                if preds:
                    selection.append(float(np.mean(np.abs(np.asarray(preds, dtype=float) - target))))
            elif mae is not None:
                selection.append(mae)
        torch.set_rng_state(state)
        return (float(np.mean(selection)) if selection else float("inf"),
                min(valids) if valids else 0.0,
                float(np.mean(truth)) if truth else float("nan"))

    t0 = time.time()
    deadline = t0 + e.MAX_TIME_HOURS * 3600
    history, it = [], 0
    while it < e.MAX_ITERS and time.time() < deadline:
        m = trainer.step()
        history.append(m)
        if it % e.LOG_EVERY == 0:
            e.log(f"[iter {it}] reward={m['reward_mean']:+.3f}(min {m['reward_min']:+.2f}) "
                  f"kl={m['kl']:.4f} adv_std={m['adv_std']:.2f} grad={m['grad_norm']:.2f}")
        if e.EARLY_STOP and e.PROBE_EVERY and it > 0 and (it + 1) % e.PROBE_EVERY == 0:
            backup = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            if trainer.ema is not None:
                trainer.ema.copy_to(adapter)
            mae, valid, true_mae = probe_eval()
            # The adapter currently HOLDS the EMA weights, so any snapshot has to be taken here,
            # before the live training weights go back in.
            eligible = valid >= pre_valid_floor
            improved = eligible and (mae < best["mae"])
            improved_true = eligible and (true_mae < best_true["mae"])
            if e.SAVE_PROBE_CKPTS:
                adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_probe{it:04d}"))
            if improved or improved_true:
                snap = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                if improved:
                    best = {"mae": mae, "iter": it, "true_mae": true_mae, "state": snap}
                if improved_true:
                    best_true = {"mae": true_mae, "iter": it, "sel_mae": mae, "state": snap}
            adapter.load_state_dict(backup)                   # restore the live training weights
            e.track("probe_mae", float(mae)), e.track("probe_valid", float(valid))
            e.track("probe_true_mae", float(true_mae)), e.track("probe_iter", int(it))
            e.log(f"[probe iter{it}] sel_mae={mae:.3f} true_mae={true_mae:.3f} valid={valid:.1%} "
                  f"floor={pre_valid_floor:.1%} best={best['mae']:.3f}@{best['iter']} "
                  f"truthbest={best_true['mae']:.3f}@{best_true['iter']} "
                  f"{'*NEW*' if improved else ''}{' *NEW-TRUE*' if improved_true else ''}")
        it += 1
    wall = time.time() - t0
    e.log(f"RL done: {it} iterations in {wall / 60:.1f} min")
    e.commit_json("rl_history.json", history)

    # Deployment weights: the best early-stop snapshot, else the final EMA.
    if e.EARLY_STOP and best["state"] is not None:
        adapter.load_state_dict(best["state"])
        e.log(f"early stop: deploying the snapshot from iteration {best['iter']} "
              f"(probe sel_mae={best['mae']:.3f}, true_mae={best.get('true_mae', float('nan')):.3f})")
    elif trainer.ema is not None:
        trainer.ema.copy_to(adapter)
    # How far did the deployed adapter actually move? A tiny change means RL never learned (learning
    # rate too low, or the EMA lagging) and any pre -> post change is evaluation noise.
    deployed = adapter.state_dict()
    wdiff = max(float((deployed[k] - input_state[k]).abs().max())
                for k in input_state if deployed[k].dtype.is_floating_point)
    e["results/deployed_weight_diff"] = wdiff
    e.log(f"deployed adapter max|dweight| vs the input = {wdiff:.3e}"
          + ("   WARNING: NEAR ZERO, RL did not learn (pre -> post is noise)" if wdiff < 1e-4 else ""))
    ckpt = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_rl"))
    e.log(f"saved the RL adapter -> {ckpt}")

    # The truth-selected snapshot, whenever it is not the one just saved. With an RDKit reward the
    # two metrics are the same number and this never fires; with a learned head it is the difference
    # between shipping what the head liked and shipping what RDKit liked.
    e["results/selection_cost"] = None
    if e.EARLY_STOP and best_true["state"] is not None and best_true["iter"] != best["iter"]:
        cost = float(best["true_mae"]) - float(best_true["mae"])
        e["results/selection_cost"], e["results/truthpick_iter"] = cost, best_true["iter"]
        deployed_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        adapter.load_state_dict(best_true["state"])
        path = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_rl_truthpick"))
        adapter.load_state_dict(deployed_state)               # the post-RL evaluation scores the DEPLOYED one
        e.log(f"selection divergence: the metric picked iteration {best['iter']} "
              f"(true={best['true_mae']:.4f}); RDKit truth preferred iteration {best_true['iter']} "
              f"(true={best_true['mae']:.4f}). Selection cost {cost:+.4f}. Saved -> {path}")
    elif e.EARLY_STOP and best_true["state"] is not None:
        e["results/selection_cost"] = 0.0
        e.log(f"selection agreement: the metric and RDKit truth both picked iteration {best['iter']}")

    e.log("=== POST-RL evaluation ===")
    post_ev = eval_now("post-RL")

    summary = {"property": e.PROPERTY, "kl_coef": e.KL_COEF, "iterations": it,
               "wall_seconds": round(wall, 1), "reward_source": e.REWARD_SOURCE,
               "reward_shape": e.REWARD_SHAPE, "crn": e.CRN, "rollout_eta": e.ROLLOUT_ETA,
               "rollout_weight": rollout_weight, "eval_eta": e.ETA, "lr": e.LR,
               "ema_decay": e.EMA_DECAY, "probe_percentiles": list(e.PROBE_PERCENTILES),
               "probe_targets": probe_targets, "probe_n_per": e.PROBE_N_PER,
               "eval_size_model": bool(eval_size_dist is not None),
               "early_stop_best_iter": best["iter"] if e.EARLY_STOP else None,
               "early_stop_best_true_mae": best.get("true_mae") if e.EARLY_STOP else None,
               "truthpick_iter": best_true["iter"] if e.EARLY_STOP else None,
               "selection_cost": e["results/selection_cost"], "deployed_weight_diff": wdiff,
               "targets": targets, "pre": pre_ev, "post": post_ev, "mae_delta": {}, "disc_delta": {}}
    pct = lambda x: f"{x * 100:.0f}%" if x is not None else "?"      # noqa: E731
    e.log("=" * 60)
    e.log(f"pre -> post   (RDKit MAE = TRUTH; head MAE = the selection metric; reward={e.REWARD_SOURCE})")
    for level in e.LEVEL_NAMES:
        for w in weights:
            pr, po = pre_ev["levels"][level]["per_w"][str(w)], post_ev["levels"][level]["per_w"][str(w)]
            delta = (po["mae"] - pr["mae"]) if pr["mae"] is not None and po["mae"] is not None else None
            hpre, hpost = pr.get("head_mae"), po.get("head_mae")
            hdelta = (hpost - hpre) if hpre is not None and hpost is not None else None
            summary["mae_delta"][f"{level}_w{w}"] = {
                "pre": pr["mae"], "post": po["mae"], "delta": delta, "head_pre": hpre,
                "head_post": hpost, "head_delta": hdelta, "valid_pre": pr.get("valid"),
                "valid_post": po.get("valid")}
            summary["disc_delta"][f"{level}_w{w}"] = {"pre": pr["disc"], "post": po["disc"]}
            head = f" | head MAE {hpre} -> {hpost} (d{hdelta})" if hpre is not None else ""
            e.log(f"{level} w={w}: RDKit MAE {pr['mae']} -> {po['mae']} (d{delta}){head} | "
                  f"disconnected {pct(pr['disc'])}->{pct(po['disc'])} | "
                  f"valid {pct(pr.get('valid'))}->{pct(po.get('valid'))}")
    e.commit_json("rl_finetune_metrics.json", summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{level}\nw{w}" for level in e.LEVEL_NAMES for w in weights]
    pre = [pre_ev["levels"][lv]["per_w"][str(w)]["mae"] or np.nan for lv in e.LEVEL_NAMES for w in weights]
    post = [post_ev["levels"][lv]["per_w"][str(w)]["mae"] or np.nan for lv in e.LEVEL_NAMES for w in weights]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pre, 0.4, label="pre-RL", color="0.6")
    ax.bar(x + 0.2, post, 0.4, label="post-RL", color="#55a868")
    ax.set_xticks(x), ax.set_xticklabels(labels)
    ax.set(ylabel=f"{e.PROPERTY} error to target", title=f"Adapter RL: {e.PROPERTY} (kl={e.KL_COEF}); lower is better")
    ax.legend()
    fig.tight_layout()
    e.commit_fig("mae_pre_post.png", fig)
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.MAX_TIME_HOURS = 0.05
    e.MAX_ITERS = 2
    e.ROLLOUT_SIZE = 8
    e.N_GROUPS = 2
    e.ROLLOUT_STEPS = 5
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH = 4
    e.EVAL_STEPS = 5
    e.N_PER_TARGET = 8
    e.N_BASELINE = 8
    e.EVAL_CHUNK = 8
    e.REF_SUBSAMPLE = 300
    e.PROBE_EVERY = 1
    e.PROBE_N_PER = 8
    e.EARLY_STOP = True
    e.LOG_EVERY = 1


experiment.run_if_main()
