"""
Feynman-Kac / Sequential-Monte-Carlo steering for DeFoG (training-free).

Implements inference-time, reward-tilted sampling in the spirit of
*Discrete Feynman-Kac Correctors* (arXiv:2601.10403) and Feynman-Kac steering:
run a population of ``K`` particles through the denoising process (the frozen model's own
dynamics, or an adapter composition as the proposal), score each particle by a **potential**
derived from an external reward, and periodically **resample** (kill low-weight particles,
clone high-weight ones).

Because the resampling acts on whole trajectories, the ensemble self-corrects globally: a
particle that happens to be heading toward the target "takes over", which attacks both
failure modes of plug-in guidance (undershoot / within-mode reward-hacking, and
mode-selection). No training or fine-tuning; the reward is evaluated on each particle's
*predicted clean* graph. The potential plus resampling implement (approximate) reward-tilted
sampling; with an adapter composition as the proposal, the adapter proposes target-ish
molecules and the resampling concentrates them further (a stronger, heuristic search, not an
exact importance sampler).

Cost: ``K`` forward passes per step (the batch) + one extra forward per particle
at each resample checkpoint (to predict the clean graph for the reward), + reward
evaluation (e.g. RDKit decode) on ``K`` graphs per checkpoint.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import PlaceHolder, dense_to_pyg
from .sampler import Sampler


def _resample_indices(w: torch.Tensor, K: int, scheme: str) -> torch.Tensor:
    """Draw K parent indices from normalised weights ``w``.

    Every scheme here is UNBIASED -- each particle's expected offspring count is K*w_i.
    They differ only in the VARIANCE of that count, and that variance is what destroys
    diversity: a particle that deserves one child and randomly gets none is gone for good,
    while its neighbour gets three identical copies.

    multinomial
        K independent draws. The highest-variance scheme of the three, and the one this
        sampler used unconditionally.
    systematic
        One uniform offset, then K equally spaced points through the cumulative weights.
        A particle with weight w gets either floor(K*w) or ceil(K*w) offspring -- never
        zero if it deserves one -- so the offspring count is as close to deterministic as
        an unbiased scheme allows. Standard in the SMC literature and strictly lower
        variance than multinomial.
    residual
        Deterministically award floor(K*w_i) copies, then draw the remainder
        multinomially from the fractional parts. Also lower variance than multinomial;
        systematic is usually at least as good, which is why both are offered rather than
        one being assumed.
    """
    if scheme == "multinomial":
        return torch.multinomial(w, K, replacement=True)
    if scheme == "systematic":
        u = torch.rand(1, device=w.device) / K
        positions = u + torch.arange(K, device=w.device, dtype=w.dtype) / K
        return torch.searchsorted(w.cumsum(0).contiguous(), positions).clamp_(max=K - 1)
    if scheme == "residual":
        counts = torch.floor(K * w)
        deterministic = torch.repeat_interleave(
            torch.arange(K, device=w.device), counts.long())
        n_left = K - int(counts.sum().item())
        if n_left <= 0:
            return deterministic[:K]
        resid = K * w - counts
        total = resid.sum()
        # All weights exactly on multiples of 1/K leaves no fractional mass to draw from;
        # fall back to uniform rather than dividing by zero.
        resid = (resid / total) if float(total) > 0 else torch.full_like(w, 1.0 / K)
        extra = torch.multinomial(resid, n_left, replacement=True)
        return torch.cat([deterministic, extra])
    raise ValueError(f"unknown resample scheme {scheme!r}; "
                     f"have 'multinomial', 'systematic', 'residual'")


@torch.no_grad()
def predict_clean(model, X_t, E_t, y_t, t, node_mask):
    """MAP prediction of the clean graph from the current noisy state: run the
    model, softmax, argmax per node/edge. Returns masked one-hot (X1, E1)."""
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
    extra = model._compute_extra_data(noisy)
    pred = model.forward(noisy, extra, node_mask)
    pX = F.softmax(pred.X, dim=-1)
    pE = F.softmax(pred.E, dim=-1)
    X1 = F.one_hot(pX.argmax(-1), pX.size(-1)).float()
    E1 = F.one_hot(pE.argmax(-1), pE.size(-1)).float()
    ph = PlaceHolder(X=X1, E=E1, y=y_t).mask(node_mask)
    return ph.X, ph.E


class FeynmanKacSampler(Sampler):
    """
    Reward-tilted SMC steering over a (frozen) DeFoG model.

    Args:
        model:   the base DeFoGModel.
        energy_fn: callable ``(X1, E1, node_mask) -> (K,)`` energy per graph
            (lower = better), evaluated on the predicted CLEAN graph. For molecular
            property steering use :class:`MoleculePropertyEnergy` (returns
            ``(prop - target)^2``; invalid graphs get a large energy so they are
            culled). The FK potential is ``phi = -beta * energy``.
        beta:    tilt strength (inverse temperature of the reward).
        resample_interval: resample every this many steps (default ~steps/8).
        warmup_frac: fraction of steps to run before the FIRST resample. This
            matters a lot: early in denoising the predicted clean graph is
            unreliable AND size-biased -- large graphs are transiently invalid or
            overshoot the target, so early resampling culls them prematurely and
            collapses the sizes toward small "reliable" molecules. Resample only
            LATE (default 0.7; 0.8+ is tighter), once predictions are valid and
            shaped, so selection reflects final quality, not transient states.
        ess_frac: if set, only resample when the effective sample size drops below
            ``ess_frac * K`` (adaptive resampling; preserves diversity). If None,
            resample at every checkpoint.
    """

    def __init__(self, model, energy_fn: Callable, *, beta=1.0,
                 resample_interval: Optional[int] = None, warmup_frac: float = 0.7,
                 ess_frac: Optional[float] = None,
                 resample_scheme: str = "residual", composition=None, **kwargs):
        super().__init__(model, **kwargs)
        self.energy_fn = energy_fn
        # Optional AdapterComposition as the PROPOSAL: the frozen-base dynamics are
        # conditioned by the adapter(s) -- the inherited Sampler._advance threads
        # self.composition into denoise_step -- while FK reward-tilts + resamples on
        # top. The FK energy is scored on the ADAPTER-conditioned predicted-clean graph
        # (target-aware), so "FK refinement over adapter conditioning" works end-to-end.
        if composition is not None:
            self.composition = composition
        self.beta = float(beta)
        self.resample_interval = resample_interval
        self.warmup_frac = warmup_frac
        self.ess_frac = ess_frac
        if resample_scheme not in ("multinomial", "systematic", "residual"):
            raise ValueError(f"unknown resample_scheme {resample_scheme!r}")
        # The default is `residual`. All three schemes are unbiased and the measured
        # differences were inside the noise floor on both properties, so this is a choice on
        # theoretical grounds -- lower offspring-count variance is never worse in expectation
        # -- NOT an empirical result. The earliest recorded results used "multinomial"; pin
        # that scheme to reproduce them, and write the scheme into any result you keep.
        self.resample_scheme = resample_scheme
        #: One record per resample checkpoint of the LAST ``sample`` call (empty before the first):
        #: ``step``, ``ess_over_k`` (effective sample size as a fraction of the particles),
        #: ``logw_range`` (spread of the accumulated log-weights: the steering pressure) and
        #: ``resampled``. A run that never resamples is the unguided sampler with extra forward
        #: passes, and no error metric reveals it, so read this before concluding that Feynman-Kac
        #: "did not help": a ``beta`` that is too small shows up as ``ess_over_k`` near 1 and a
        #: ``logw_range`` near 0 at every checkpoint.
        self.trace = []

    def _desc(self) -> str:
        tag = ""
        if getattr(self, "composition", None) is not None and len(self.composition) > 0:
            tag += "+adapter"
        return "FK-SMC" + tag

    @torch.no_grad()
    def _predict_clean(self, X, E, y, t_norm, node_mask):
        """MAP clean-graph prediction for the reward. If an adapter ``composition`` is
        the proposal, predict WITH the adapter modulation (target-aware) -- averaging
        the branch predictions for a multi-branch composition -- so the energy reflects
        the conditioned model rather than the unconditional base."""
        comp = getattr(self, "composition", None)
        if comp is None or len(comp) == 0:
            return predict_clean(self.model, X, E, y, t_norm, node_mask)
        K = X.size(0)
        noisy = {"X_t": X, "E_t": E, "y_t": y, "t": t_norm, "node_mask": node_mask}
        extra = self.model._compute_extra_data(noisy)
        pXs, pEs = [], []
        for br in comp.branches:
            c = torch.as_tensor(br.condition, dtype=torch.float32, device=X.device)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.size(0) == 1 and K > 1:
                c = c.expand(K, -1)
            pred = self.model.forward(noisy, extra, node_mask, cond_modulation=br.adapter(c, t=t_norm))
            pXs.append(F.softmax(pred.X, dim=-1)); pEs.append(F.softmax(pred.E, dim=-1))
        pX = torch.stack(pXs).mean(0); pE = torch.stack(pEs).mean(0)
        X1 = F.one_hot(pX.argmax(-1), pX.size(-1)).float()
        E1 = F.one_hot(pE.argmax(-1), pE.size(-1)).float()
        ph = PlaceHolder(X=X1, E=E1, y=y).mask(node_mask)
        return ph.X, ph.E

    @torch.no_grad()
    def _potential(self, X, E, y, t_norm, node_mask):
        X1, E1 = self._predict_clean(X, E, y, t_norm, node_mask)
        energy = self.energy_fn(X1, E1, node_mask).to(X.device).reshape(-1)
        return -self.beta * energy

    @torch.no_grad()
    def sample(self, num_samples, num_nodes=None, size_dist=None, condition=None,
               device=None, show_progress=True, on_step=None):
        model = self.model
        was_training = model.training
        model.eval()
        device = device if device is not None else model.device

        # ESS is >= 1 by construction, so a threshold of `ess_frac * num_samples` at or below 1 can
        # never be crossed and FK silently degenerates to plain (adapter) sampling -- same
        # molecules, same MAE, uniqueness 1.0, and nothing raised. Both ends of this have
        # already cost real jobs: K=4 with ess_frac=0.25, and K=10 with ess_frac=0.10.
        if self.ess_frac is not None and self.ess_frac * num_samples <= 1.0:
            raise ValueError(
                f"ess_frac={self.ess_frac} with K={num_samples} puts the resample threshold at "
                f"{self.ess_frac * num_samples:.3f} particles, and the effective sample "
                f"size is never below 1, so Feynman-Kac would never resample and this run "
                f"would silently "
                f"be plain sampling. Raise ess_frac above {1.0 / num_samples:.3g}, or raise K."
            )

        node_mask, n_nodes, X, E, y, y_raw, use_cfg = model._prepare_generation(
            num_samples, num_nodes, size_dist, condition, self.guidance_scale, device
        )
        K = num_samples
        logw = torch.zeros(K, device=device)
        prev_phi = None
        self.trace = []
        ri = self.resample_interval or max(1, self.sample_steps // 8)
        warmup = int(self.warmup_frac * self.sample_steps)

        iterator = range(self.sample_steps)
        if show_progress:
            iterator = tqdm(iterator, desc=self._desc())

        for t_int in iterator:
            X, E, y = self._advance(t_int, X, E, y, node_mask, use_cfg)
            if on_step is not None:
                on_step(t_int + 1, self.sample_steps,
                        "fk_warmup" if t_int < warmup else "sampling")

            is_checkpoint = (
                t_int >= warmup
                and (t_int - warmup) % ri == 0
                and t_int < self.sample_steps - 1
            )
            if not is_checkpoint:
                continue

            t_norm = ((t_int + 1) / self.sample_steps) * torch.ones(K, 1, device=device)
            phi = self._potential(X, E, y, t_norm, node_mask)
            # incremental weight (telescoping): reward *gained* since last checkpoint
            logw = logw + (phi if prev_phi is None else phi - prev_phi)
            prev_phi = phi

            w = torch.softmax(logw, dim=0)
            ess = 1.0 / w.pow(2).sum().clamp_min(1e-12)
            will_resample = (self.ess_frac is None) or (ess < self.ess_frac * K)
            self.trace.append({"step": t_int + 1, "ess_over_k": float(ess) / K,
                               "logw_range": float(logw.max() - logw.min()),
                               "resampled": bool(will_resample)})
            if will_resample:
                idx = _resample_indices(w, K, self.resample_scheme)
                X, E, node_mask, y = X[idx], E[idx], node_mask[idx], y[idx]
                prev_phi = prev_phi[idx]
                if y_raw is not None:
                    y_raw = y_raw[idx]
                logw = torch.zeros(K, device=device)
                if on_step is not None:
                    on_step(t_int + 1, self.sample_steps, "fk_resample")

        # Hook: constrained samplers install their exact target here, after the last denoise step
        # (the same hook, at the same place, as in `Sampler.sample`). No-op in the base sampler.
        X, E = self._post_loop(X, E, node_mask)

        X, E, _ = model.limit_dist.ignore_virtual_classes(X, E)
        n_final = node_mask.sum(-1)
        samples = dense_to_pyg(X, E, y_raw, node_mask, n_final)

        if was_training:
            model.train()
        return samples
