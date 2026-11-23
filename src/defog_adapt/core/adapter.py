"""
Frozen-base cross-attention / gated-FiLM (adaLN-Zero style) CFG-adapters for DeFoG.

Localize classifier-free guidance to a small, swappable, stackable adapter on a
FROZEN unconditional base (ControlNet / T2I-Adapter / IP-Adapter family), instead
of retraining the base per condition. See ``docs/concepts/adapters.md``.

Mechanism
---------
* The base ``DeFoGModel`` (``cond_dim=0``) is frozen. A :class:`CrossAttentionAdapter`
  reads a condition ``c`` and emits, per transformer layer, a gated-FiLM
  modulation ``{scale, shift, gate}`` for the node (X), edge (E) and global (y)
  hidden streams. The modulation is a **gated residual** applied to each frozen
  layer's *output*: ``h' = h + mask ⊙ (gate ⊙ (scale·h + shift))``.
* The **gate** head is zero-initialized, so at init (and for the unconditional
  branch, which applies no modulation at all) the base is reproduced **exactly**
  -- the property N-branch product-of-experts composition relies on.
* Training (:class:`AdapterModule`) optimizes ONLY the adapter with the base's own
  denoising cross-entropy loss -> a conditional denoiser ``p(x1|x_t,c)`` (NOT the
  Bregman/positive-pairing objective that flattened the earlier guidance adapter).
* Composition (:class:`AdapterComposition`) stacks N conditions as product-of-
  experts on the rate matrices (generalizing the shipped 2-branch CFG blend to
  N+1 branches, run as one batched ``(N+1)·B`` forward). Consumed by
  :class:`~defog_adapt.core.sampler.AdaptedSampler`.

Adapters are EXTERNAL objects (never attached to ``DeFoGModel``) so the base's
``save``/``load`` and every existing checkpoint are untouched.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .data import to_dense
from .layers import timestep_embedding

_STREAMS = ("X", "E", "y")
_DIMKEY = {"X": "dx", "E": "de", "y": "dy"}

# --- the default adapter architecture ---------------------------------------------
# Measured on ZINC logP under the E2 protocol (100 targets x 10 generations): the
# FiLM-only adapter reached MAE 0.5420, and adding node->condition cross-attention plus
# Fourier condition bands took it to 0.3250 at equal or better validity. Attribution at
# 60 epochs: cross-attention -0.153, Fourier -0.024, both together -0.192 (superadditive).
#
# 64/128/16 comes from the capacity sweep, but READ THAT SWEEP CAREFULLY: seed-to-seed
# spread on an identical configuration was 0.058 MAE, which is larger than nearly every
# difference in it. These are a defensible point in a flat region, not a measured optimum,
# and re-tuning them on one seed will mostly measure the seed.
#
# Applied by `for_base`, NOT by `__init__` -- see for_base's docstring for why that
# distinction is load-bearing for old checkpoints.
DEFAULT_XATTN_TOKENS = 64
DEFAULT_XATTN_DIM = 128
DEFAULT_XATTN_HEADS = 16
DEFAULT_COND_FOURIER = 3

# Accepted keys of CrossAttentionAdapter.__init__, used by from_config to ignore extras rather
# than raise on a config written by a newer version.
_CONFIG_KEYS = frozenset({
    "cond_dim", "n_layers", "dims", "hidden", "time_conditioned", "streams",
    "time_emb_dim", "cond_mean", "cond_std", "name", "cond_type",
    "base_token", "cond_encoder",
    "cond_fourier", "xattn_tokens", "xattn_dim", "xattn_heads",
})

# Flags that adapter configs written by the research package carry, for features this package
# does not include (interior conditioning was superseded by cross-attention). A saved config
# that has them switched OFF is exactly a plain adapter, so it loads; one that has them ON
# needs heads this class does not have, and must fail loudly rather than load a different model.
_REMOVED_FLAGS = ("interior_ff", "interior_attn")


def _drop_removed_features(config: dict, where: str) -> dict:
    """Return ``config`` without the removed-feature flags, refusing any that are enabled."""
    cfg = dict(config)
    for key in _REMOVED_FLAGS:
        if cfg.pop(key, False):
            raise ValueError(
                f"{where} enables {key!r}, an adapter feature that is not part of defog-adapt "
                f"(interior conditioning was superseded by node cross-attention). Load this "
                f"adapter with the research package it was trained in.")
    return cfg


# ===========================================================================
# Modulation: per-layer FiLM params for a (possibly stacked) batch
# ===========================================================================
class Modulation:
    """Per-layer gated-FiLM parameters for one forward pass.

    ``layers`` is a list (len == n_layers) of dicts; each holds, for the streams
    present, keys ``scale{X,E,y}``, ``shift{X,E,y}``, ``gate{X,E,y}`` each of shape
    ``(B, channel)``. A zero ``gate`` makes that stream's delta exactly 0 (bypass).
    """

    def __init__(self, layers: List[Dict[str, torch.Tensor]], xattn=None):
        self.layers = layers
        #: Optional per-layer node cross-attention, as a list (len == n_layers) of
        #: ``[(row_slice, fn), ...]``. ``fn(X_slice, x_mask_slice) -> delta`` is a
        #: closure over the adapter's own layer module and its condition tokens, which
        #: is why this cannot be a plain tensor like the FiLM parameters: the module has
        #: to run INSIDE the frozen base's stack, on that group's rows only.
        #:
        #: The (slice, fn) list-per-layer form is what "keeps the door open" for
        #: N-branch stacking: a second branch is one more entry at a different offset.
        #: It is refused today (untested, see AdapterComposition) rather than absent.
        self.xattn = xattn

    def apply(self, i, X, E, y, x_mask, e_mask):
        """Apply layer ``i``'s modulation to the block outputs (gated residual).
        ``x_mask`` (B,n,1) / ``e_mask`` (B,n,n,1) mask the *delta* (never the
        hidden state) so gate=0 reproduces the base's own padding behavior."""
        m = self.layers[i]
        if "gateX" in m:
            X = X + x_mask * (m["gateX"][:, None] * (m["scaleX"][:, None] * X + m["shiftX"][:, None]))
        if "gateE" in m:
            E = E + e_mask * (m["gateE"][:, None, None] * (m["scaleE"][:, None, None] * E + m["shiftE"][:, None, None]))
        if "gatey" in m:
            y = y + m["gatey"] * (m["scaley"] * y + m["shifty"])
        if self.xattn and self.xattn[i]:
            # Built as a full-size delta and added, rather than assigned into X in
            # place: X is needed for the backward pass, and an in-place write on it
            # would either error or silently corrupt the gradient.
            delta = torch.zeros_like(X)
            for sl, fn in self.xattn[i]:
                delta[sl] = fn(X[sl], x_mask[sl])
            X = X + delta
        return X, E, y

    @staticmethod
    def stack_groups(mods: Sequence["Modulation"], bs: int, device) -> "Modulation":
        """Build a ``(N+1)·bs`` modulation: group 0 = the frozen base (all-zero rows, an exact
        bypass -- the unconditional branch of classifier-free guidance), groups 1..N = each
        adapter's modulation, concatenated along the batch dim.

        Robust to HETEROGENEOUS key sets across branches: an adapter that lacks a key another
        one has (e.g. one that modulates only the node stream next to one that modulates all
        three) is treated as gate=0 -> exact bypass at that site. Takes the UNION of keys over
        all branches and zero-fills any key a branch does not define."""
        n_layers = len(mods[0].layers)
        for m in mods:
            assert len(m.layers) == n_layers, (
                f"modulation layer-count mismatch: {len(m.layers)} != {n_layers}")
        combined = []
        for L in range(n_layers):
            keys = set()
            for m in mods:
                keys |= set(m.layers[L].keys())
            d = {}
            for k in sorted(keys):
                ch = next(m.layers[L][k].shape[-1] for m in mods if k in m.layers[L])
                # group 0 first, then one row per branch. A missing key becomes zeros,
                # which is an exact bypass at that site.
                rows = [torch.zeros(bs, ch, device=device)]
                for m in mods:
                    t = m.layers[L].get(k)
                    rows.append(t if t is not None else torch.zeros(bs, ch, device=device))
                d[k] = torch.cat(rows, dim=0)
            combined.append(d)

        # Cross-attention closures are re-homed from "all rows" onto this group's
        # rows. Group g occupies [g*bs, (g+1)*bs); group 0 is the frozen base.
        xa = None
        if any(m.xattn for m in mods):
            xa = [[] for _ in range(n_layers)]
            for g, m in enumerate(mods, start=1):
                if not m.xattn:
                    continue
                off = g * bs
                for L in range(n_layers):
                    for sl, fn in m.xattn[L]:
                        if sl != slice(None):
                            # An already-stacked Modulation being re-stacked: the offsets
                            # would compose wrongly and silently. `raise`, not `assert` --
                            # python -O strips asserts and this must survive it.
                            raise ValueError(
                                "stack_groups expects each source modulation to carry "
                                f"whole-batch cross-attention entries; got {sl}. This "
                                "Modulation has already been stacked.")
                        xa[L].append((slice(off, off + bs), fn))
        return Modulation(combined, xattn=xa)


def _bind_xattn(module, tokens: torch.Tensor):
    """Bind one layer's cross-attention module to this call's condition tokens.

    The result is what ``Modulation`` carries: ``fn(X, x_mask) -> masked delta``. The
    mask is applied to the DELTA, never to X, so a padded row is left exactly as the
    frozen base left it -- the same contract the FiLM path uses."""
    def fn(X, x_mask):
        return x_mask * module(X, tokens)
    return fn


class NodeConditionCrossAttention(nn.Module):
    """One frozen-base layer's cross-attention from NODES to condition tokens.

    Why this exists. The FiLM path computes every scale/shift/gate from ``c`` alone and
    broadcasts the same numbers to every atom: the research log calls this an
    open-loop controller and argues it is why property targeting plateaus -- logP is a
    sum of per-atom environment contributions, so the right action at a given atom
    depends on that atom, and a uniform push cannot express it.

    Here each atom forms a QUERY from its own current representation and pulls a weighted
    mixture of condition tokens.

    BE PRECISE ABOUT WHAT THIS BUYS, because the obvious claim is wrong. "Two atoms get
    different deltas" is ALREADY true of FiLM -- its delta at node i is
    ``g(c) * (s(c) * X_i + b(c))``, which varies with X_i. Both mechanisms are node-local
    functions of ``(X_i, c)``. The difference is the FORM of that function: FiLM applies
    one DIAGONAL AFFINE map, identical for every atom, whereas cross-attention is a
    content-addressed nonlinear lookup -- the atom's own representation selects which
    condition tokens it reads, so different kinds of atom can receive qualitatively
    different corrections rather than the same correction scaled per channel.

    Because those representations come out of the frozen base's own message passing, by
    the middle layers a query already reflects the atom's neighbourhood, not just its
    element. It is still NOT the global running total that a closed-loop readout
    (not part of this package) would provide: nothing here tells the adapter what the partial
    molecule's logP currently is.

    Structure follows IP-Adapter: a SEPARATE attention path added to the frozen block's
    output, with a ZERO-INITIALISED output projection, so at init the delta is exactly
    zero and the base is reproduced bit-for-bit. That is the same invariant the zero-init
    FiLM gate provides, and the product-of-experts composition depends on it.

    Applied at the layer OUTPUT (via :meth:`Modulation.apply`) rather than spliced inside
    the block: the frozen base's own attention is untouched, so nothing about its
    computation changes and the only new tensor is an additive residual.
    """

    def __init__(self, dx: int, d_tok: int, n_heads: int = 8):
        super().__init__()
        if dx % n_heads:
            raise ValueError(f"dx={dx} not divisible by n_heads={n_heads}")
        self.dx, self.d_tok, self.n_heads = int(dx), int(d_tok), int(n_heads)
        # Pre-norm on the query side only. The frozen base's activation scale is not
        # ours to assume, and an unnormalised query would make the attention logits
        # depend on it; the keys/values come from our own tokens and are already tame.
        self.norm = nn.LayerNorm(dx)
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(d_tok, dx)
        self.v = nn.Linear(d_tok, dx)
        self.out = nn.Linear(dx, dx)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # zero-init => exact no-op at init

    def forward(self, X: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``X`` (B,n,dx), ``tokens`` (B,m,d_tok) -> delta (B,n,dx)."""
        B, n, _ = X.shape
        if tokens.dim() == 3 and tokens.size(1) < 2:
            # The CrossAttentionAdapter-level guard covers the normal path, but this module can be
            # constructed directly: softmax over a single key is identically 1.0, so every
            # node gets the same value and the query path receives no gradient.
            raise ValueError(
                f"cross-attention needs >= 2 condition tokens, got {tokens.size(1)}: "
                f"softmax over one key is constant, so this degenerates to a broadcast.")
        if tokens.dim() != 3 or tokens.size(0) != B:
            # `.view(B, -1, h, dh)` would SUCCEED on a mismatched batch by silently
            # reinterpreting one graph's tokens as several graphs' -- no error, a wrong
            # delta. The FiLM path broadcasts a 1-row condition correctly, so the two
            # paths would disagree and only one of them would say so.
            raise ValueError(
                f"tokens must be (B, m, d_tok) with B={B} matching the queries; got "
                f"{tuple(tokens.shape)}. Expand the condition to the full batch first "
                f"(AdapterComposition._broadcast_condition does this).")
        h, dh = self.n_heads, self.dx // self.n_heads
        q = self.q(self.norm(X)).view(B, n, h, dh).transpose(1, 2)          # B,h,n,dh
        k = self.k(tokens).view(B, -1, h, dh).transpose(1, 2)               # B,h,m,dh
        v = self.v(tokens).view(B, -1, h, dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, n, self.dx)
        return self.out(o)
        # NOTE: no mask over the KEYS -- condition tokens are always present. Padded
        # QUERY rows are zeroed by the caller's x_mask, same as the FiLM delta.

# ===========================================================================
# CrossAttentionAdapter: c -> per-layer modulation
# ===========================================================================
def _base_token(base) -> float:
    """A cheap, stable identity token for a frozen base (to catch hot-swapping an
    adapter onto a different base with matching dims). Sum of a fixed weight."""
    with torch.no_grad():
        return float(base.model.mlp_in_X[0].weight.detach().double().sum().cpu())


class FingerprintEncoder(nn.Module):
    """Residual MLP encoder for a wide, flat condition such as a Morgan fingerprint.

    Why this exists: without an encoder the trunk's first layer is the *only* thing
    between the condition and every FiLM head, and it is ``Linear(cond_dim + t, hidden)``
    with ``hidden`` fixed at 256. Widening the fingerprint therefore does not widen the
    path — 512 and 1024 bits are both compressed to 256 in one step, and 2048 would be an
    8:1 squeeze. Measured: 512 -> 1024 bits bought +0.015 Tanimoto lift, which is what
    pouring more information into an unchanged bottleneck looks like. This encoder moves
    the narrowest point off the first layer so that "more bits" and "wider path" become
    separable choices rather than one confounded one.

    There is no spatial structure to exploit: Morgan bits are hash buckets, so adjacency is
    meaningless and a convolution would be modelling an artifact of the hash. Depth with
    residual connections is the right prior instead — it lets the encoder learn bit
    *co-occurrence* (substructures that appear together) without assuming anything about
    bit ordering.

    The adapter's exact-no-op-at-init property is unaffected: it comes from the zero-init
    gate heads, which sit downstream of this module.
    """

    def __init__(self, in_dim: int, out_dim: int = 512, hidden: int = 1024,
                 n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.proj_in = nn.Linear(self.in_dim, self.hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden),
                *( [nn.Dropout(self.dropout)] if self.dropout > 0 else [] ),
            )
            for _ in range(self.n_blocks)
        ])
        self.proj_out = nn.Sequential(nn.LayerNorm(self.hidden), nn.SiLU(),
                                      nn.Linear(self.hidden, self.out_dim))

    def _config(self) -> dict:
        """Enough to rebuild this encoder. Without it, anything that reconstructs an adapter
        from ``_config()`` — notably ``AdapterGDPOTrainer``'s frozen KL reference — would
        rebuild it *without* the encoder and silently compare against a different model."""
        return dict(kind="mlp", in_dim=self.in_dim, out_dim=self.out_dim,
                    hidden=self.hidden, n_blocks=self.n_blocks, dropout=self.dropout)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.size(-1) != self.in_dim:
            raise ValueError(f"FingerprintEncoder expects {self.in_dim} inputs, got {c.size(-1)}")
        h = self.proj_in(c)
        for blk in self.blocks:
            h = h + blk(h)          # pre-norm residual: identity path stays clean
        return self.proj_out(h)


#: Condition encoders addressable from a saved config's ``kind`` field. Keeping the
#: registry next to the classes means adding one is a single edit, and an unknown kind
#: fails loudly at load rather than silently rebuilding the wrong architecture.
_COND_ENCODERS = {"mlp": FingerprintEncoder}


class CrossAttentionAdapter(nn.Module):
    """Zero-init gated-FiLM adapter, with node-to-condition cross-attention, over a FROZEN base's transformer stack.

    Maps a condition ``c`` (+ optionally the flow-time ``t``) to per-layer
    ``{scale, shift, gate}`` for the enabled streams. Exact no-op at init (gate
    heads zero-initialized) -> the base is reproduced bit-for-bit.

    The name is the recorded architecture (``defog_adapt.presets.ADAPTER_ARCHITECTURE``): a gated-FiLM trunk, plus
    cross-attention from every node to the condition (the half that carries the effect), plus Fourier bands on the
    condition. ``xattn_tokens=0`` and ``cond_fourier=0`` switch the last two off and give the original FiLM-only
    adapter, which ``docs/about/results-and-caveats.md`` measures as an arm of its own. The research fork called this class
    ``AdaLNAdapter``; that name is kept as an alias (below), and no checkpoint stores a class name.
    """

    def __init__(self, cond_dim: int, n_layers: int, dims: Dict[str, int],
                 hidden: int = 256, time_conditioned: bool = True,
                 streams: Sequence[str] = _STREAMS, time_emb_dim: int = 64,
                 cond_mean=None, cond_std=None, name: str = "", cond_type: str = "",
                 base_token: Optional[float] = None,
                 cond_encoder: Optional[nn.Module] = None,
                 cond_fourier: int = 0, xattn_tokens: int = 0,
                 xattn_dim: int = 128, xattn_heads: int = 8):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.dims = dict(dims)
        self.hidden = hidden
        self.time_conditioned = time_conditioned
        self.time_emb_dim = time_emb_dim
        self.streams = tuple(streams)
        self.name, self.cond_type = name, cond_type
        self.base_token = base_token
        self.cond_fourier = int(cond_fourier)     # Fourier bands on the condition (0 = off)
        self.xattn_tokens = int(xattn_tokens)     # node->condition cross-attention (0 = off)
        self.xattn_dim = int(xattn_dim)
        self.xattn_heads = int(xattn_heads)

        # An encoder, when present, sits between the (already normalised) condition and the
        # trunk, so everything downstream — state_dict, ConditionBranch, the package format —
        # still sees exactly one module with one cond_dim.
        if isinstance(cond_encoder, dict):
            spec = dict(cond_encoder)
            # Configs written before `kind` existed are all spectrum encoders, so that
            # stays the default for them: a kind-less config must be reported as what it
            # is, not rebuilt as something else.
            kind = spec.pop("kind", "spectrum_cnn")
            if kind == "spectrum_cnn":
                raise ValueError(
                    "this adapter uses the spectrum condition encoder ('spectrum_cnn'), which "
                    "is not part of defog-adapt. Load it with the research package it was "
                    "trained in.")
            if kind not in _COND_ENCODERS:
                raise ValueError(f"unknown cond_encoder kind {kind!r} in {cond_encoder!r}; "
                                 f"known: {sorted(_COND_ENCODERS)}")
            cond_encoder = _COND_ENCODERS[kind](**spec)
        self.cond_encoder = cond_encoder
        encoded_dim = cond_dim if cond_encoder is None else int(cond_encoder.out_dim)

        # --- Fourier features on the (normalised) condition ---------------------
        # The trunk otherwise sees the property as ONE raw float while the flow-time
        # gets a 64-dim sinusoidal embedding. Tancik et al. (arXiv:2006.10739): an MLP
        # over a low-dimensional input converges to the high-frequency part of its
        # target impractically slowly, so the learned map c -> modulation gets the
        # smooth part long before the part that distinguishes 3.5 from 4.2. That is
        # the measured failure signature -- bias without tracking.
        #
        # LOW-DIMENSIONAL ONLY, which is the regime the result is about. A fingerprint
        # condition goes through its own encoder instead; expanding 512 hash bits into
        # bands would be both meaningless and enormous.
        if self.cond_fourier < 0:
            raise ValueError(f"cond_fourier must be >= 0, got {self.cond_fourier}")
        if self.cond_fourier:
            if cond_encoder is not None:
                raise ValueError("cond_fourier is for a raw low-dimensional condition; "
                                 "this adapter already has a cond_encoder")
            if cond_dim > 8:
                raise ValueError(f"cond_fourier with cond_dim={cond_dim}: Fourier features "
                                 f"are for LOW-dimensional conditions (<= 8)")
            # Frequencies 0.5, 1, 2, ... cycles per z-unit on the z-scored condition.
            #
            # THE BANDWIDTH IS A CEILING, NOT A KNOB TO RAISE. The band vector must stay
            # SIMILAR for nearby targets (so the adapter interpolates to targets it never
            # saw, which is most of them) while being clearly DIFFERENT for far ones.
            # Measured cosine similarity of the feature vector on ZINC logP
            # (cond_std 1.158, targets spanning ~9.3 units, i.e. ~8 z):
            #
            #     n   f_max   cycles/range   cos(0.1 logP)   cos(2.5 logP)
            #     2     1.0        8            0.910            0.710     under-resolved
            #     3     2.0       16            0.762            0.335     <- default
            #     4     4.0       32            0.431            0.087
            #     6    16.0      128            0.104           -0.125     aliased
            #     8    64.0      512           -0.035            0.075     noise
            #
            # n=3 keeps neighbours correlated while separating the ends; by n=6 a
            # 0.1-logP step has already decorrelated the features and the mapping is
            # memorisation, not interpolation. The raw normalised scalar rides alongside
            # and carries the smooth component, so the bands only have to add resolution.
            # `test_fourier_bandwidth_keeps_neighbours_correlated` fails if this aliases.
            self.register_buffer("fourier_freqs",
                                 0.5 * (2.0 ** torch.arange(self.cond_fourier, dtype=torch.float32)))
        fourier_dim = 2 * cond_dim * self.cond_fourier
        cond_in = encoded_dim + fourier_dim + (time_emb_dim if time_conditioned else 0)
        self.trunk = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # per (layer, stream): a scale/shift head (normal init) + a zero-init gate.
        self.ss = nn.ModuleList()
        self.gate = nn.ModuleList()
        for _ in range(n_layers):
            ss_l, gate_l = nn.ModuleDict(), nn.ModuleDict()
            for s in self.streams:
                ch = dims[_DIMKEY[s]]
                ss_l[s] = nn.Linear(hidden, 2 * ch)
                g = nn.Linear(hidden, ch)
                nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)   # zero-init gate => exact no-op
                gate_l[s] = g
            self.ss.append(ss_l); self.gate.append(gate_l)

        # --- node -> condition cross-attention (one module per layer) -----------
        if self.xattn_tokens == 1:
            # softmax over ONE key is identically 1.0, so every node receives the same
            # value vector and q.weight never gets gradient -- the mechanism degenerates
            # into exactly the per-graph broadcast it exists to avoid, silently.
            raise ValueError(
                "xattn_tokens=1 degenerates: softmax over a single token is constant, so "
                "the delta is identical for every node and the query path receives no "
                "gradient. Use >= 2 tokens, or 0 to disable cross-attention.")
        if self.xattn_tokens < 0:
            raise ValueError(f"xattn_tokens must be >= 0, got {self.xattn_tokens}")
        if self.xattn_tokens:
            self.tok = nn.Linear(hidden, self.xattn_tokens * self.xattn_dim)
            self.xattn = nn.ModuleList([
                NodeConditionCrossAttention(dims["dx"], self.xattn_dim, self.xattn_heads)
                for _ in range(n_layers)
            ])

        m = torch.zeros(cond_dim) if cond_mean is None else torch.as_tensor(cond_mean, dtype=torch.float32).reshape(-1)
        s = torch.ones(cond_dim) if cond_std is None else torch.as_tensor(cond_std, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
        self.register_buffer("cond_mean", m)   # buffer -> follows .to(device) and is in state_dict
        self.register_buffer("cond_std", s)

    # --- construction helper ------------------------------------------------
    @classmethod
    def for_base(cls, base, cond_dim: int, **kw) -> "CrossAttentionAdapter":
        """Build an adapter matching ``base``'s transformer dims (read from the live
        module) and layer count, in the CURRENT DEFAULT ARCHITECTURE.

        The defaults live here and deliberately not in ``__init__``. ``load`` and
        ``from_config`` rebuild a saved adapter by calling ``__init__`` with the stored
        config, and a checkpoint written before cross-attention existed has no
        ``xattn_tokens`` key in it at all. Defaulting the mechanism ON in ``__init__``
        would reconstruct every one of those adapters with a cross-attention path their
        state_dict does not contain -- and for an encoder adapter (fingerprint) it would
        hit the Fourier guard and raise outright. So ``__init__`` stays
        conservative and means "exactly what the config says", while ``for_base`` means
        "what we would build today".

        Pass ``xattn_tokens=0, cond_fourier=0`` for the FiLM-only adapter that every
        result before 2026-08-28 was measured on.
        """
        attn = base.model.tf_layers[0].self_attn
        dims = {"dx": attn.dx, "de": attn.de, "dy": attn.dy}
        n_layers = len(base.model.tf_layers)
        kw.setdefault("base_token", _base_token(base))
        kw.setdefault("xattn_tokens", DEFAULT_XATTN_TOKENS)
        kw.setdefault("xattn_dim", DEFAULT_XATTN_DIM)
        kw.setdefault("xattn_heads", DEFAULT_XATTN_HEADS)
        if "cond_fourier" not in kw:
            # Fourier features are a result about LOW-dimensional inputs, and __init__
            # refuses them next to an encoder or a wide condition. An unconditional
            # default would therefore make for_base() raise for every fingerprint
            # adapter, so the default is "on where it is legal", not "on".
            legal = kw.get("cond_encoder") is None and cond_dim <= 8
            kw["cond_fourier"] = DEFAULT_COND_FOURIER if legal else 0
        return cls(cond_dim, n_layers, dims, **kw)

    # --- forward ------------------------------------------------------------
    def normalize(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.cond_mean) / self.cond_std

    def _fourier(self, c_norm: torch.Tensor) -> torch.Tensor:
        """(B, D) normalised condition -> (B, 2*D*n_bands) sin/cos features."""
        ang = 2.0 * math.pi * c_norm.unsqueeze(-1) * self.fourier_freqs   # (B, D, F)
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)

    def forward(self, c: torch.Tensor, t: Optional[torch.Tensor] = None) -> Modulation:
        c_norm = self.normalize(c.float())
        # The encoder replaces the raw condition; Fourier features ACCOMPANY it. The raw
        # normalised scalar is kept alongside the bands so nothing the trunk could learn
        # before is taken away -- turning the feature on cannot lose information.
        parts = [self.cond_encoder(c_norm) if self.cond_encoder is not None else c_norm]
        if self.cond_fourier:
            parts.append(self._fourier(c_norm))
        if self.time_conditioned:
            assert t is not None, "time_conditioned adapter requires t"
            parts.append(timestep_embedding(t.reshape(-1, 1), self.time_emb_dim))
        h = self.trunk(torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0])

        xattn = None
        if self.xattn_tokens:
            tokens = self.tok(h).view(h.size(0), self.xattn_tokens, self.xattn_dim)
            # slice(None) here; Modulation.stack_groups re-homes each closure onto its
            # group's rows when the branches are batched together.
            xattn = [[(slice(None), _bind_xattn(mod, tokens))] for mod in self.xattn]

        layers = []
        for L in range(self.n_layers):
            d = {}
            for s in self.streams:
                scale, shift = self.ss[L][s](h).chunk(2, dim=-1)
                gate = self.gate[L][s](h)
                d[f"scale{s}"], d[f"shift{s}"], d[f"gate{s}"] = scale, shift, gate
            layers.append(d)
        return Modulation(layers, xattn=xattn)

    # --- compatibility / io -------------------------------------------------
    def check_compatible(self, base):
        attn = base.model.tf_layers[0].self_attn
        assert self.dims["dx"] == attn.dx and self.dims["de"] == attn.de and self.dims["dy"] == attn.dy, \
            f"adapter dims {self.dims} != base ({attn.dx},{attn.de},{attn.dy})"
        assert self.n_layers == len(base.model.tf_layers), \
            f"adapter n_layers {self.n_layers} != base {len(base.model.tf_layers)}"
        if self.base_token is not None:
            tok = _base_token(base)
            if abs(tok - self.base_token) > 1e-3 * (1 + abs(self.base_token)):
                warnings.warn(
                    f"adapter '{self.name}' was trained on a different base "
                    f"(token {self.base_token:.4g} != {tok:.4g}); steering may be meaningless.",
                    RuntimeWarning)

    def _config(self):
        encoder = None if self.cond_encoder is None else self.cond_encoder._config()
        return dict(cond_encoder=encoder,
                    cond_dim=self.cond_dim, n_layers=self.n_layers, dims=self.dims,
                    hidden=self.hidden, time_conditioned=self.time_conditioned,
                    streams=list(self.streams), time_emb_dim=self.time_emb_dim,
                    name=self.name, cond_type=self.cond_type,
                    base_token=self.base_token,
                    cond_fourier=self.cond_fourier, xattn_tokens=self.xattn_tokens,
                    xattn_dim=self.xattn_dim, xattn_heads=self.xattn_heads)

    def config(self) -> dict:
        """The architecture config needed to rebuild this adapter (public alias).

        Exposed for out-of-band serialization -- e.g. a container that stores tensors
        separately from declarations (safetensors + a metadata file) and therefore
        cannot rely on ``save``/``load`` round-tripping a pickled dict.
        """
        return self._config()

    @classmethod
    def from_config(cls, config: dict, state_dict: dict, device="cpu") -> "CrossAttentionAdapter":
        """Rebuild an adapter from a ``config()`` dict and a separately-stored state dict.

        The counterpart to :meth:`config`, for callers that keep tensors and
        declarations in different files. ``load`` remains the path for ``.ckpt``
        checkpoints written by :meth:`save`.

        Unknown config keys are ignored rather than raising, so a config written by a
        newer version stays loadable as long as the tensors match. The one exception is a
        feature this package does not include: a config that switches it on is refused (see
        ``_REMOVED_FLAGS``), because ignoring the key would load a different model.
        """
        config = _drop_removed_features(config, "adapter config")
        cfg = {k: v for k, v in config.items() if k in _CONFIG_KEYS}
        if "streams" in cfg:
            cfg["streams"] = tuple(cfg["streams"])
        a = cls(**cfg)
        a.load_state_dict(state_dict)   # includes the cond_mean/cond_std buffers
        return a.to(device)

    def save(self, path):
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"
        torch.save({"state_dict": self.state_dict(), "config": self._config()}, path)
        return path

    @classmethod
    def load(cls, path, device="cpu") -> "CrossAttentionAdapter":
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = _drop_removed_features(ck["config"], f"adapter checkpoint {path}")
        cfg["streams"] = tuple(cfg["streams"])
        # Filter through _CONFIG_KEYS exactly as from_config does. The .ckpt path is the
        # one the run scripts use, and it was the stricter of the two: a checkpoint
        # written by a newer version raised TypeError here while from_config accepted it.
        dropped = sorted(set(cfg) - _CONFIG_KEYS)
        if dropped:
            warnings.warn(f"ignoring unknown config keys in {path}: {dropped}",
                          RuntimeWarning)
        a = cls(**{k: v for k, v in cfg.items() if k in _CONFIG_KEYS})
        a.load_state_dict(ck["state_dict"])   # includes cond_mean/cond_std buffers
        return a.to(device)


#: The research fork's name for :class:`CrossAttentionAdapter`. Kept as an alias so that code written against the
#: fork, and its migration guide, keep working: it is the same class, not a subclass or a copy.
AdaLNAdapter = CrossAttentionAdapter


# ===========================================================================
# Composition
# ===========================================================================
@dataclass
class ConditionBranch:
    """One condition in a composition: an adapter, a RAW target (the adapter
    normalizes internally), and its CFG weight ``w``.

    The default is 2.0, the measured optimum for logP and QED. Branches are blended in
    probability space (see :class:`AdapterComposition`), which is what makes w>1 usable at all.
    """
    adapter: CrossAttentionAdapter
    condition: torch.Tensor
    weight: float = 2.0


class AdapterComposition:
    """N-branch product-of-experts spec consumed by ``denoise_step`` /
    ``AdaptedSampler``. ``mode='product'`` sums the log-ratios; ``'mean'`` averages
    (recommended for N>1 to keep the effective uncond coefficient bounded)."""

    def __init__(self, branches: Sequence[ConditionBranch], base=None, mode: str = "product"):
        """
        ``mode`` is the FORM of the blend (geometric product vs its per-branch mean).

        The blend is applied to the predicted clean-graph MARGINALS, and ONE rate matrix is
        derived from the result. That is where FreeGress applies guidance (Eq. 10/11); it
        spends a single ``X_1`` draw, and it keeps the trajectory consistent with the
        terminal step, which blends the same clean log-probs. Blending a rate matrix per
        branch instead breaks above w=1: at w=2 it collapsed logP validity to 0.526 and MAE
        to 5.59, against 0.982 and 0.542 here, so that placement is not offered.
        """
        assert mode in ("product", "mean")
        self.branches = list(branches)
        self.mode = mode
        # Cross-attention branches stack. Modulation carries them as (row_slice, closure)
        # per layer and stack_groups re-homes each closure onto its own group's rows, so a
        # second branch is one more entry at a different offset. This was refused while
        # nothing tested it; test_two_xattn_adapters_stack_without_crosstalk now checks
        # that each group's output equals that adapter's SOLO forward, which is the
        # property that makes composition mean anything.
        if base is not None:
            for b in self.branches:
                b.adapter.check_compatible(base)

    def __len__(self):
        return len(self.branches)

    @staticmethod
    def _broadcast_condition(condition, bs: int, device) -> torch.Tensor:
        c = torch.as_tensor(condition, dtype=torch.float32, device=device)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.size(0) == 1 and bs > 1:
            c = c.expand(bs, -1)
        assert c.size(0) == bs, f"branch condition batch {c.size(0)} != {bs}"
        return c

    @torch.no_grad()
    def build_modulation(self, bs: int, t: torch.Tensor) -> Modulation:
        """Combined ``(N+1)·bs`` modulation: group 0 = the frozen base (all zeros, an exact
        bypass), group i = adapter_i."""
        device = t.device
        mods = [br.adapter(self._broadcast_condition(br.condition, bs, device), t=t)
                for br in self.branches]
        return Modulation.stack_groups(mods, bs, device)

    def weights(self, device, dtype=torch.float32) -> torch.Tensor:
        """``(N,)`` tensor of the branches' scalar CFG weights."""
        return torch.tensor([float(b.weight) for b in self.branches],
                            device=device, dtype=dtype)


# ===========================================================================
# Training module
# ===========================================================================
class _FrozenBaseModule(pl.LightningModule):
    """Shared plumbing for training an add-on over a FROZEN base: freeze the base, and keep
    it out of the add-on's checkpoints."""

    def _freeze_base(self, base):
        self.base = base.requires_grad_(False)

    def on_save_checkpoint(self, checkpoint):
        # Don't serialize the frozen base inside every adapter checkpoint.
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("base.")
        }

    @staticmethod
    def _dense(batch):
        d, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        d = d.mask(node_mask)
        return d.X, d.E, node_mask


class AdapterModule(_FrozenBaseModule):
    """Train ONLY the adapter (frozen base) with the base's own denoising CE loss:
    a direct conditional denoiser ``p(x1|x_t,c)``. The base is frozen and kept out of the
    checkpoints by :class:`_FrozenBaseModule`."""

    def __init__(self, base, adapter: CrossAttentionAdapter, cond_attr: str = "cond", lr: float = 2e-4):
        super().__init__()
        self._freeze_base(base)
        self.adapter = adapter
        self.cond_attr = cond_attr
        self.lr = float(lr)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.adapter.parameters(), lr=self.lr, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs, device = X1.size(0), X1.device
        c = getattr(batch, self.cond_attr).to(device).view(bs, -1).float()   # RAW; adapter normalizes
        y0 = torch.zeros(bs, 0, device=device)
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask)
            extra = self.base._compute_extra_data(noisy)
        mod = self.adapter(c, t=noisy["t"])
        pred = self.base.forward(noisy, extra, node_mask, cond_modulation=mod)
        loss = self.base.train_loss(pred_X=pred.X, pred_E=pred.E, pred_y=pred.y,
                                    true_X=X1, true_E=E1, true_y=y0, node_mask=node_mask)
        self.log("adapter/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss
