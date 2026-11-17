"""Where reference data lives, and that loading it never silently reaches for the network.

The research repo found its data next to the source tree (``Path(__file__).parents[2]``), which
inside an installed package is ``site-packages``: not writable and wiped on upgrade. The root is a
per-user cache now, movable with ``DEFOG_ADAPT_HOME``. And ``load_reference_split`` used to
download on its own by default; fetching ~100 MB because a function was called is not something a
loader should do unasked.
"""
from pathlib import Path

import pytest

import defog_adapt
from defog_adapt.data import _paths
from defog_adapt.data import chembl_reference as cr
from defog_adapt.data import moses_reference as mr
from defog_adapt.data import zinc_reference as zr


def test_the_env_var_moves_every_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFOG_ADAPT_HOME", str(tmp_path))
    assert zr.default_root() == tmp_path / "zinc250k"
    assert mr.default_root() == tmp_path / "moses"
    assert Path(cr.default_root()) == tmp_path / "chembl"


def test_the_default_is_a_user_cache_and_never_the_package_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("DEFOG_ADAPT_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = zr.default_root()
    assert root == tmp_path / "defog_adapt" / "zinc250k"
    package = Path(defog_adapt.__file__).resolve().parent
    assert root.resolve() != package and package not in root.resolve().parents


def test_without_xdg_the_default_is_under_the_home_directory(monkeypatch):
    monkeypatch.delenv("DEFOG_ADAPT_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert _paths.data_home() == Path.home() / ".cache" / "defog_adapt"


def test_the_root_is_read_when_called_not_when_imported(monkeypatch, tmp_path):
    monkeypatch.delenv("DEFOG_ADAPT_HOME", raising=False)
    before = zr.default_root()
    monkeypatch.setenv("DEFOG_ADAPT_HOME", str(tmp_path))
    assert zr.default_root() == tmp_path / "zinc250k" != before


def _forbid_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("loading reference data reached for the network")
    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    monkeypatch.setattr("urllib.request.urlopen", boom)


@pytest.mark.parametrize("mod", [zr, mr], ids=["zinc", "moses"])
def test_loading_missing_data_does_not_touch_the_network(mod, monkeypatch, tmp_path):
    """The default is download=False: with nothing on disk the loader says how to get the data
    (``download_reference()``) instead of fetching it."""
    monkeypatch.setenv("DEFOG_ADAPT_HOME", str(tmp_path))
    _forbid_the_network(monkeypatch)
    with pytest.raises(mod.ReferenceDataError) as exc:
        mod.load_reference_split()
    assert "download_reference" in str(exc.value)


@pytest.mark.parametrize("mod", [zr, mr], ids=["zinc", "moses"])
def test_download_true_still_asks_for_the_download(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFOG_ADAPT_HOME", str(tmp_path))
    seen = []

    def fake(root=None, **kwargs):
        seen.append(root)
        raise RuntimeError("stop here")

    monkeypatch.setattr(mod, "download_reference", fake)
    with pytest.raises(RuntimeError, match="stop here"):
        mod.load_reference_split(download=True)
    assert len(seen) == 1
