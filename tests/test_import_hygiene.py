"""What importing and using the library does to the process around it.

A library should not make global decisions for its user. Two of them used to be made here:
``callbacks.py`` called ``matplotlib.use("Agg")`` at import time (silently breaking interactive
and notebook sessions), and every figure went through pyplot's global registry (which leaked one
per epoch). These tests pin the opposite behaviour.

(``matplotlib.pyplot`` itself IS imported by ``import defog_adapt``: PyTorch Lightning's
``torchmetrics`` imports it. That is not this package's doing, so it is not asserted against.)
"""

import subprocess
import sys
import textwrap

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=300)


def test_importing_the_library_never_selects_a_matplotlib_backend():
    """matplotlib.use() is the call that makes the decision; record it instead of inferring
    from the resulting backend name (which, headless, is 'agg' anyway)."""
    proc = _run("""
        import matplotlib
        calls = []
        real = matplotlib.use
        matplotlib.use = lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1]
        import defog_adapt.core, defog_adapt.data
        try:
            import defog_adapt.domains
        except ImportError:
            pass                      # RDKit may be absent; the backend question is the same
        assert not calls, f"the library called matplotlib.use: {calls}"
        print("ok")
    """)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stdout + proc.stderr


def test_the_training_figure_is_not_held_by_pyplot(tmp_path):
    """A Figure built by the callbacks must not appear in pyplot's registry (so it cannot leak
    one per epoch), and must still be a real, savable matplotlib Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from defog_adapt.core.callbacks import TrainingMonitorCallback

    cb = TrainingMonitorCallback()
    for key in cb.history:               # a plausible two-epoch history of the right shape
        cb.history[key] = [0.5, 0.4]
    before = plt.get_fignums()
    fig = cb._generate_figure()
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == before, "the callback registered a figure with pyplot"
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


# ============================================================ optional dependencies
def test_the_core_and_the_data_loaders_import_without_rdkit():
    """RDKit is an optional dependency (the ``chem`` extra). Everything that is not chemistry
    has to import and work without it."""
    proc = _run("""
        import sys
        sys.modules["rdkit"] = None            # `import rdkit` now raises ImportError
        import defog_adapt.core, defog_adapt.data
        from defog_adapt.core import DeFoGModel, FeynmanKacSampler, AdaLNAdapter
        print("ok")
    """)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stdout + proc.stderr


def test_the_chemistry_domain_names_the_extra_to_install_when_rdkit_is_missing():
    proc = _run("""
        import sys
        sys.modules["rdkit"] = None
        try:
            import defog_adapt.domains
        except ImportError as exc:
            assert "defog-adapt[chem]" in str(exc), f"unhelpful message: {exc}"
            print("ok")
        else:
            raise SystemExit("defog_adapt.domains imported without RDKit?")
    """)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stdout + proc.stderr
