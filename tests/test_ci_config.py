"""The CI and release workflows do what they promise, and nothing publishes by accident.

Workflows cannot be run here, so what is checked is the part that can be: that they parse, that the
floors job is generated from ``pyproject.toml`` (and so cannot drift from it), and that the release
workflow has the properties that make publishing safe: a tag is required for the real index, no API
token exists anywhere, and a manual run can only reach the test index.
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
    lowers = {}
    for requirement in pyproject["project"]["dependencies"] + pyproject["project"]["optional-dependencies"]["chem"] \
            + pyproject["project"]["optional-dependencies"]["data"]:
        name, version = re.match(r"([\w.\-]+)>=([\w.]+)", requirement).groups()
        lowers[name] = version
    assert {p.split("==")[0] for p in pins} == set(lowers)
    assert all(p.split("==")[1] == lowers[p.split("==")[0]] for p in pins)


def test_the_floors_script_converts_lower_bounds_and_ignores_the_rest():
    module = floor_module()
    fake = {"project": {
        "dependencies": ["alpha>=1.2", "beta>=3.4.5,<4", "gamma"],
        "optional-dependencies": {"chem": ["delta>=2023.3.1"], "all": ["defog-adapt[chem]"]}}}
    assert module.floors(fake, ("core", "chem")) == ["alpha==1.2", "beta==3.4.5", "gamma", "delta==2023.3.1"]
    assert module.floors(fake, ("all",)) == [], "a self-reference to other extras is not a requirement"


# ============================================================ release
def test_a_release_needs_a_tag_and_a_manual_run_can_only_reach_the_test_index():
    release = workflow("release")
    assert release["triggers"]["push"]["tags"] == ["v*"]
    manual = release["triggers"]["workflow_dispatch"]["inputs"]["target"]
    assert manual["options"] == ["testpypi"], "a manual run must never be able to publish for real"
    assert release["jobs"]["publish-pypi"]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert release["jobs"]["publish-pypi"]["needs"] == "publish-testpypi", "the test index goes first"


def test_publishing_uses_trusted_publishing_and_no_stored_credentials():
    release = workflow("release")
    for job in ("publish-testpypi", "publish-pypi"):
        assert release["jobs"][job]["permissions"] == {"id-token": "write"}
        assert "environment" in release["jobs"][job], "protect each environment with a required reviewer"
    body = yaml.dump(release)                    # the parsed workflow: comments cannot trip or hide this
    assert "secrets." not in body and "password" not in body and "PYPI_API_TOKEN" not in body, \
        "there must be no API token: Trusted Publishing needs none"


def test_the_test_index_step_tolerates_a_version_that_already_exists():
    steps = workflow("release")["jobs"]["publish-testpypi"]["steps"]
    publish = next(s for s in steps if "gh-action-pypi-publish" in s.get("uses", ""))
    assert publish["with"]["skip-existing"] is True
