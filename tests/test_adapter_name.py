"""The adapter class is ``CrossAttentionAdapter``; the research fork's name, ``AdaLNAdapter``, is an alias.

The rename is safe only because of three facts, and each is pinned here: the old name is the SAME object (so fork code,
the migration guide and the parity gate, which calls the research package's class by its old name on both sides, keep
working); a checkpoint stores no class name (so every existing adapter, and the ones being trained now, loads under
either); and the FiLM-only adapter the comparison in docs/about/results-and-caveats.md calls an arm of its own is this class too, with
the cross-attention and the Fourier bands switched off.
"""

import torch

import defog_adapt.core as core
from defog_adapt.core import AdaLNAdapter, CrossAttentionAdapter
from defog_adapt.core.adapter import AdaLNAdapter as AdaLNAdapterFromTheModule


def test_the_research_forks_name_is_the_same_class_not_a_copy():
    assert AdaLNAdapter is CrossAttentionAdapter is AdaLNAdapterFromTheModule


def test_both_names_are_exported():
    assert "CrossAttentionAdapter" in core.__all__ and "AdaLNAdapter" in core.__all__


def test_a_checkpoint_stores_no_class_name_and_loads_through_either_name(tmp_path, small_model):
    adapter = CrossAttentionAdapter.for_base(small_model, cond_dim=1, hidden=16, cond_mean=[2.0], cond_std=[1.5])
    path = adapter.save(str(tmp_path / "adapter"))
    stored = torch.load(path, weights_only=False)
    assert set(stored) == {"state_dict", "config"}, "a stored class name or a pickled object would tie the file to a name"

    via_new, via_old = CrossAttentionAdapter.load(path), AdaLNAdapter.load(path)
    assert type(via_new) is type(via_old) is CrossAttentionAdapter
    for (name, a), (other, b) in zip(via_new.state_dict().items(), via_old.state_dict().items()):
        assert name == other and torch.equal(a, b)


def test_the_film_only_adapter_is_this_class_with_the_cross_attention_switched_off(small_model):
    film = CrossAttentionAdapter.for_base(small_model, cond_dim=1, hidden=16, xattn_tokens=0, cond_fourier=0)
    full = CrossAttentionAdapter.for_base(small_model, cond_dim=1, hidden=16)
    assert type(film) is type(full) is CrossAttentionAdapter
    assert film.xattn_tokens == 0 and film.cond_fourier == 0
    assert full.xattn_tokens > 0 and full.cond_fourier > 0, "the default is the recorded architecture"
    assert sum(p.numel() for p in film.parameters()) < sum(p.numel() for p in full.parameters())
