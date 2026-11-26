"""The CI and release workflows do what they promise, and nothing publishes by accident.

Workflows cannot be run here, so what is checked is the part that can be: that they parse, that the
floors job is generated from ``pyproject.toml`` (and so cannot drift from it), and that the release
workflow has the properties that make publishing safe: a tag is required to reach the index, a manual
run publishes nothing, no API token exists anywhere, and the publishing job stays dormant until
Trusted Publishing has been set up.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10: pytest itself depends on the backport
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SCRIPT = ROOT / ".github" / "scripts" / "floor_requirements.py"

pytestmark = pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=".github/ is not part of an sdist")


def workflow(name):
    data = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text())
    data["triggers"] = data.pop(True, data.pop("on", None))        # YAML reads a bare `on` as True
    return data


def floor_module():
    spec = importlib.util.spec_from_file_location("floor_requirements", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================ CI
def test_ci_runs_the_suite_on_every_supported_python_and_at_the_floors():
    ci = workflow("ci")
    assert {"test", "floors", "build"} <= set(ci["jobs"])
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    supported = sorted(c.split("::")[-1].strip() for c in pyproject["project"]["classifiers"]
                       if c.startswith("Programming Language :: Python :: 3."))
    assert sorted(ci["jobs"]["test"]["strategy"]["matrix"]["python"]) == supported
    assert pyproject["project"]["requires-python"] == ">=" + supported[0]


def test_the_floors_job_is_generated_from_pyproject_not_written_out():
    steps = yaml.dump(workflow("ci")["jobs"]["floors"])
    assert "floor_requirements.py" in steps and "3.10" in steps
    assert not re.search(r"torch==\d", steps), "the pins must come from pyproject, not be copied here"


def test_ci_installs_cpu_pytorch_before_everything_else():
    text = (WORKFLOWS / "ci.yml").read_text()
    assert text.count("download.pytorch.org/whl/cpu") >= 3, "every job that installs torch must use the CPU index"


# ============================================================ the floors script
def test_every_floor_pin_is_the_lower_bound_in_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = floor_module().floors(pyproject)
    lowers, unpinned = {}, set()
    for requirement in pyproject["project"]["dependencies"] + pyproject["project"]["optional-dependencies"]["chem"]:
        match = re.match(r"([\w.\-]+)>=([\w.]+)", requirement)
        if match:
            lowers[match.group(1)] = match.group(2)
        else:
            unpinned.add(re.split(r"[<>=!~ ]", requirement)[0])     # no tested floor: installed unpinned
    assert dict(p.split("==") for p in pins if "==" in p) == lowers
    assert {p for p in pins if "==" not in p} == unpinned


def test_the_only_extra_is_chem_and_the_rest_is_installed_by_default():
    """The user-facing install is one line: everything but RDKit comes with the package."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert list(pyproject["project"]["optional-dependencies"]) == ["chem"]
    core = {re.split(r"[<>=!~ ]", r)[0] for r in pyproject["project"]["dependencies"]}
    assert {"pandas", "psutil", "nvidia-ml-py"} <= core
    assert "pycomex" not in core and "pynvml" not in core, \
        "PyComex caps numpy below 2, and the PyPI package pynvml is deprecated and warns on import"
    assert "dev" in pyproject["dependency-groups"], "contributor tools belong in a dependency group, not an extra"


def test_the_floors_script_converts_lower_bounds_and_ignores_the_rest():
    module = floor_module()
    fake = {"project": {
        "dependencies": ["alpha>=1.2", "beta>=3.4.5,<4", "gamma"],
        "optional-dependencies": {"chem": ["delta>=2023.3.1"], "all": ["defog-adapt[chem]"]}}}
    assert module.floors(fake, ("core", "chem")) == ["alpha==1.2", "beta==3.4.5", "gamma", "delta==2023.3.1"]
    assert module.floors(fake, ("all",)) == [], "a self-reference to other extras is not a requirement"


# ============================================================ release
def test_a_release_needs_a_tag_and_a_manual_run_publishes_nothing():
    release = workflow("release")
    assert release["triggers"]["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in release["triggers"], "a manual run is the dry run of the build"
    assert set(release["jobs"]) == {"build", "publish-pypi"}, "there is no test index: only a PyPI account exists"
    publish = release["jobs"]["publish-pypi"]
    assert "startsWith(github.ref, 'refs/tags/v')" in publish["if"], \
        "only a pushed tag may reach the index: a manual run is on a branch and must never publish"
    assert publish["needs"] == "build"


def test_publishing_is_dormant_until_trusted_publishing_is_set_up():
    condition = workflow("release")["jobs"]["publish-pypi"]["if"]
    assert "vars.PYPI_TRUSTED_PUBLISHING == 'true'" in condition, \
        "until a trusted publisher exists on PyPI the job could only fail, so it must not run before the variable says so"


def test_publishing_uses_trusted_publishing_and_no_stored_credentials():
    release = workflow("release")
    publish = release["jobs"]["publish-pypi"]
    assert publish["permissions"] == {"id-token": "write"}
    assert "environment" in publish, "protect the environment with a required reviewer"
    assert release["permissions"] == {"contents": "read"}, "everything but the publishing job gets read access only"
    action = next(s for s in publish["steps"] if "gh-action-pypi-publish" in s.get("uses", ""))
    assert "repository-url" not in action.get("with", {}), "the real index is the default: no other index is configured"
    body = yaml.dump(release)                    # the parsed workflow: comments cannot trip or hide this
    assert "secrets." not in body and "password" not in body and "PYPI_API_TOKEN" not in body, \
        "there must be no API token: Trusted Publishing needs none"
    assert "test.pypi" not in body, "there is no test index"


def test_the_build_checks_the_distributions_and_keeps_them_for_the_upload():
    steps = workflow("release")["jobs"]["build"]["steps"]
    assert any("twine check dist/*" in s.get("run", "") for s in steps), "a manual run is only a dry run if it checks the build"
    kept = next(s for s in steps if "upload-artifact" in s.get("uses", ""))
    assert kept["with"]["name"] == "dist", "the upload, by hand or by the publishing job, takes the artifact called dist"
