"""The recorded recipes, as plain data.

These are NOT defaults. ``DeFoGModel`` keeps its generic constructor defaults, so behaviour does
not shift underneath existing checkpoints; the values below are the settings the recorded
molecular results were produced with, kept in one place so nobody has to reverse-engineer them
from an experiment script.

``E1_MODEL`` is the unconditional distribution-quality (E1) recipe. It is verified against the
hyperparameters stored in the checkpoints that produced the recorded ZINC and MOSES results
(``tests/test_presets.py`` re-checks it when those checkpoints are available): a 9-layer, 256-wide
graph transformer with relative random-walk features, marginal noise, polydec time distortion in
training and sampling, an edge-loss weight of 5, and a cosine learning-rate schedule from 2e-4.

Dataset-specific values are not in the preset: the atom valencies and weights
(``atom_valencies``, ``atom_weights``, ``max_atom_weight``) come from the dataset's vocabulary, and
``molecular_features`` needs them, so pass them alongside::

    from defog_adapt.presets import E1_MODEL
    model = DeFoGModel.from_dataloader(loader, **E1_MODEL,
                                       atom_valencies=..., atom_weights=..., max_atom_weight=...)

The adapters were measured on this lineage: an E1-recipe base, then two rounds of structural RL
(``examples/rl_sanity__zinc.py``). One trap when reading the checkpoint they were trained against: it was
re-packaged, so the hyperparameters stored in the file (identity time distortion, edge-loss weight 1, 100
sampling steps) are defaults filled in at export, not its training configuration; its weights are identical,
tensor for tensor, to the RL model, which carries ``E1_MODEL``. Do not rely on a loaded model's stored
sampling defaults: pass the sampling arguments explicitly, as every recipe here does.
"""

#: Model and optimiser settings behind the recorded unconditional results.
E1_MODEL = dict(
    n_layers=9,
    hidden_dim=256,
    hidden_mlp_dim=512,
    n_heads=8,
    dropout=0.1,
    noise_type="marginal",
    extra_features_type="rrwp",
    rrwp_steps=20,
    molecular_features=True,
    lambda_edge=5.0,
    train_time_distortion="polydec",
    sample_time_distortion="polydec",
    lr=2e-4,
    lr_scheduler="cosine",
    lr_min=1e-6,
    weight_decay=1e-5,
    sample_steps=500,
    eta=0.0,
    omega=0.0,
)

#: The sampling configuration the E1 results were reported at (ZINC and MOSES). The model's own
#: ``eta`` above is the stored default; this is what to sample with. For ZINC, 250 steps is
#: statistically equivalent at half the cost (recorded in the research log).
E1_SAMPLING = dict(
    sample_steps=500,
    eta=25.0,
    omega=0.0,
    time_distortion="polydec",
)

#: Unconditional structural RL (``GDPOTrainer``): reward = valid + connected + rings in [3, 8],
#: a KL pull to the frozen base, Dr. GRPO advantages. Rollouts at eta 0 and 100 steps are a known
#: approximation of the deployed 500-step, eta 25 configuration, which is why evaluation is done at
#: the latter. The recorded ZINC model used TWO rounds of 25 iterations (the second starts from the
#: first round's best checkpoint); a third round moved nothing outside the noise.
SANITY_RL_TRAINER = dict(
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

#: The adapter architecture behind every recorded adapter result: node-to-condition
#: cross-attention (64 tokens, width 128, 16 heads) plus Fourier bands on the condition, over the
#: FiLM trunk. ``cond_fourier`` is for LOW-dimensional conditions (one scalar property); a
#: high-dimensional condition such as a fingerprint uses a ``cond_encoder`` instead and no bands.
ADAPTER_ARCHITECTURE = dict(
    hidden=256,
    time_conditioned=True,
    streams=("X", "E", "y"),
    cond_fourier=3,
    xattn_tokens=64,
    xattn_dim=128,
    xattn_heads=16,
)

#: Adapter training (``AdapterModule``). Epochs, learning rate and batch size are one measured
#: recipe: 20 epochs was badly under-trained for this architecture, 60-80 is saturated, and every
#: reported number is at 4e-4. Gradient clipping is applied by the Lightning ``Trainer``.
ADAPTER_TRAINING = dict(
    epochs=80,
    batch_size=24,
    lr=4e-4,
    grad_clip=1.0,
)

#: Adapter sampling as the conditional benchmark was run: 500 steps, eta 25, polydec.
ADAPTER_SAMPLING = dict(
    sample_steps=500,
    eta=25.0,
    omega=0.0,
    time_distortion="polydec",
)

#: The classifier-free guidance weight each property's adapter measured best at (probability-space
#: product blending). Guiding beyond ~2 degrades validity; the optimum differs by property, and
#: reporting or fine-tuning at another property's weight is how a real effect gets missed.
ADAPTER_WEIGHT = {"logp": 1.5, "tpsa": 1.5, "sascore": 1.5, "qed": 2.0}

#: Feynman-Kac steering as recorded (ZINC logP and QED; 10 particles per target, 500 steps).
#: ``beta`` is DIMENSIONLESS: it multiplies an energy standardised by the property variance
#: (:class:`~defog_adapt.core.StandardisedEnergy`), so a given value means the same for logP and
#: for QED. 20 is what the headline logP and QED rows used (QED was also run at 8); the value
#: matters far more than the particle count. At 2.5 the particle system was starved (logP resampled
#: on 16% of checkpoints) and FK looked useless at 500 steps; at 20 it cut the logP error from
#: 0.3132 to 0.2299 on validation, at a cost in diversity (uniqueness about 0.76). Raising beta
#: lowers the error further and the diversity with it. Particles resample only after 60% of the
#: trajectory, and only when the effective sample size drops below half of them; the interval is
#: the sampler's own steps // 8. Tune beta by sweeping it and watching the resample rate: a run
#: that never resamples is the unguided sampler with extra forward passes.
FK_STEERING = dict(
    beta=20.0,
    warmup_frac=0.6,
    ess_frac=0.5,
    resample_scheme="residual",
)

#: Adapter fine-tuning by policy gradient (``AdapterGDPOTrainer``), head reward. ``rollout_weight``
#: is deliberately absent: it is required, and it must be the weight the adapter is DEPLOYED at
#: (``ADAPTER_WEIGHT``) or the run improves a policy nobody uses. Rollouts at eta 1 with common
#: random numbers are a different quantity from the eta 25 the evaluation is measured at.
ADAPTER_RL_TRAINER = dict(
    rollout_size=128,
    sample_steps=250,
    eta=1.0,
    omega=0.0,
    time_distortion="polydec",
    subsample_steps=16,
    minibatch_size=16,
    lambda_edge=1.0,
    reduction="mean",
    crn=True,
    kl_coef=0.05,
    lr=1e-4,
    weight_decay=1e-5,
    grad_clip=1.0,
    ema_decay=0.9,
)

__all__ = [
    "E1_MODEL", "E1_SAMPLING", "SANITY_RL_TRAINER", "ADAPTER_ARCHITECTURE", "ADAPTER_TRAINING",
    "ADAPTER_SAMPLING", "ADAPTER_WEIGHT", "FK_STEERING", "ADAPTER_RL_TRAINER",
]
