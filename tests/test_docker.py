"""Tests that exercise the built container image over a real socket.

**Why these exist when `test_api.py` already covers the endpoints.** Those tests
drive the application in-process, which is the right way to test *behaviour* --
but by construction they cannot catch anything the container adds: a missing
runtime dependency that this laptop happens to have installed, a checkpoint that
was never copied into the image, a `MODEL_PATH` pointing somewhere that does not
exist inside the filesystem the container actually sees. Every one of those
failures looks identical from the outside -- a service that starts and then
answers 503 -- and none of them is reachable without running the image.

**Why they skip instead of fail when Docker is absent.** A fresh clone has no
image and CI may have no daemon; a test suite that goes red for that reason
teaches people to ignore red. The skip message says exactly which command
produces what is missing. The cost of this choice is real and worth naming:
skipped tests protect nobody, so the image has to be built for these to mean
anything.

Run them with the image built first::

    docker build -t molecular-property-predictor:latest .
    pytest -m docker
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from molecular_property_predictor.api.schemas import METHANE_EXAMPLE

#: Every test in this module is marked, so `pytest -m "not docker"` gives the
#: fast in-process suite and `pytest -m docker` gives only these.
pytestmark = pytest.mark.docker

IMAGE = "molecular-property-predictor:latest"

#: How long to wait for the container to answer. Start-up is dominated by
#: importing torch and reading the checkpoint, which is seconds on a warm
#: machine and noticeably longer on a cold one.
STARTUP_TIMEOUT_S = 90


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a docker command and capture its output."""
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


def _require_image() -> None:
    """Skip the whole module unless a daemon and the image are both present."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    if _docker("info", check=False).returncode != 0:
        pytest.skip("the docker daemon is not running")

    if _docker("image", "inspect", IMAGE, check=False).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run: docker build -t {IMAGE} .")


def _get(url: str, timeout: float = 5.0) -> dict:
    """GET a JSON document."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post(url: str, payload: dict, timeout: float = 10.0) -> dict:
    """POST a JSON document and read the JSON reply."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_until_serving(base_url: str, container_id: str) -> None:
    """Poll /health until the service answers, or fail with the container log.

    Printing the log on timeout matters more here than in an ordinary test: the
    failure happens inside a process with its own filesystem, and without its
    stdout there is nothing to debug from.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_S

    while time.monotonic() < deadline:
        try:
            _get(f"{base_url}/health", timeout=2.0)
            return
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)

    logs = _docker("logs", container_id, check=False)
    pytest.fail(
        f"container did not serve within {STARTUP_TIMEOUT_S}s\n"
        f"--- stdout ---\n{logs.stdout}\n--- stderr ---\n{logs.stderr}"
    )


@pytest.fixture(scope="module")
def container() -> tuple[str, str]:
    """Start the image, yield ``(base_url, container_id)``, then remove it.

    Module-scoped: start-up costs seconds and nothing here mutates the
    container, so one is enough for every test below.

    Published on ``127.0.0.1:0`` -- port 0 asks the kernel for a free port,
    which is then read back with `docker port`. Hardcoding 8000 would make the
    suite fail whenever a development server is already running, and binding to
    127.0.0.1 rather than all interfaces keeps a test fixture off the network.
    """
    _require_image()

    started = _docker("run", "--detach", "--publish", "127.0.0.1:0:8000", IMAGE)
    container_id = started.stdout.strip()

    try:
        published = _docker("port", container_id, "8000/tcp").stdout.strip()
        # e.g. "127.0.0.1:49160", possibly several lines for several bindings.
        host_port = published.splitlines()[0].rsplit(":", 1)[1]
        base_url = f"http://127.0.0.1:{host_port}"

        _wait_until_serving(base_url, container_id)
        yield base_url, container_id
    finally:
        # --force because the container is still running; the removal is what
        # keeps a failed run from leaving a process holding a port.
        _docker("rm", "--force", container_id, check=False)


def test_container_serves_a_loaded_model(container):
    """The baked-in checkpoint is found and loaded inside the image.

    This is the assertion the whole module exists for: `model_loaded` is false
    exactly when COPY, MODEL_PATH and the artifact's real location disagree.
    """
    base_url, _ = container

    health = _get(f"{base_url}/health")

    assert health["status"] == "ok"
    assert health["model_loaded"] is True


def test_container_reports_the_served_checkpoint(container):
    """/model answers from inside the container, with real provenance."""
    base_url, _ = container

    info = _get(f"{base_url}/model")

    assert info["target"] == "lumo"
    assert info["units"] == "eV"
    assert info["representation"] in {"sorted_coulomb", "eigenspectrum", "composition"}
    assert info["n_parameters"] > 0


def test_container_predicts_for_a_real_molecule(container):
    """A prediction crosses a socket, a process boundary and a filesystem.

    Methane's published LUMO is 0.1171 Hartree, about 3.19 eV. The bound here
    is deliberately loose -- this is a test of the delivery path, not of model
    accuracy, which Phase 6 measured on the held-out test set.
    """
    base_url, _ = container

    prediction = _post(f"{base_url}/predict", METHANE_EXAMPLE)

    assert prediction["n_atoms"] == 5
    assert isinstance(prediction["lumo_ev"], float)
    assert -20.0 < prediction["lumo_ev"] < 20.0


def test_container_rejects_a_bad_request(container):
    """Validation survives the trip: an unsupported element gets a 422.

    Worth checking through HTTP rather than only in-process, because a 422 body
    is produced by pydantic and a 500 would mean the exception escaped instead.
    """
    base_url, _ = container
    sulfur = {"atomic_numbers": [16], "coordinates": [[0.0, 0.0, 0.0]]}

    with pytest.raises(urllib.error.HTTPError) as raised:
        _post(f"{base_url}/predict", sulfur)

    assert raised.value.code == 422


def test_container_does_not_run_as_root(container):
    """The Dockerfile's USER instruction actually took effect.

    A promise made in a comment is not a control. `id -u` inside the running
    container is, and it is the cheapest possible check of it.
    """
    _, container_id = container

    uid = _docker("exec", container_id, "id", "-u").stdout.strip()

    assert uid == "10001"
