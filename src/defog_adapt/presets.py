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

A caveat that matters for the adapter results: they were measured on a ZINC base that was trained
under an EARLIER configuration (edge-loss weight 1, identity time distortion, no LR schedule) and
then fine-tuned with reinforcement learning, not on a base trained with this recipe. There is no
recorded adapter run on an E1-recipe base.
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

__all__ = ["E1_MODEL", "E1_SAMPLING"]
