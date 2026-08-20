# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# A container image for the prediction API.
#
# Why not just "run the Python file" on the server? Because the file is the
# small half of what makes it work. The service also needs Python 3.12, a
# specific torch build, a C runtime those wheels were compiled against, and a
# checkpoint at a known path. "Install those on the server" is a procedure that
# drifts: it succeeds in a different order, against different system libraries,
# months apart. An image is that procedure executed *once* and frozen as a
# filesystem, so the thing tested locally is the thing that runs in the cloud.
#
# What the result actually is: not a virtual machine. There is no guest kernel
# and nothing is emulated. The container is one ordinary Linux process that the
# host kernel has given a private view of the filesystem, network and process
# table (namespaces), with limits on what it may consume (cgroups). That is why
# it starts in milliseconds rather than in a minute.
#
# Two stages below. The build stage needs pip's machinery and the space it
# consumes; the running service needs neither. Copying just the finished
# virtualenv into a clean second stage means none of that reaches production --
# a smaller image, and a smaller attack surface.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1: build the virtualenv
# ---------------------------------------------------------------------------
# -slim is Debian with the documentation, dev headers and extra locales removed
# (~50 MB against ~350 MB for the full tag). The tag is pinned to a minor
# version so a rebuild next year does not silently move to Python 3.14; pinning
# the digest as well would be stricter still, at the cost of never picking up
# security patches without an edit.
FROM python:3.12-slim-bookworm AS builder

# PIP_NO_CACHE_DIR: pip's download cache is worthless in a build stage that is
#   thrown away, and would add several hundred megabytes to this layer.
# PIP_DISABLE_PIP_VERSION_CHECK: silences a network call and a warning that
#   would otherwise appear in every build log.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# A virtualenv inside a container looks redundant -- the container is already
# isolated. It is here for one practical reason: it makes the entire installed
# dependency set a single directory that stage 2 can copy in one instruction.
# Installing into the system Python would scatter files across /usr/local and
# leave no clean thing to hand over.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# torch is installed first, on its own, from PyTorch's CPU index. This is the
# single most consequential line in the file.
#
# `pip install torch` on Linux resolves, by default, to the CUDA build: torch
# plus roughly a dozen nvidia-* wheels, about 2.5 GB, all of it dead weight on
# a machine with no GPU. The CPU index serves the same torch compiled without
# CUDA, around 200 MB. On a free-tier host that is the difference between an
# image that deploys and one that does not.
#
# It is a separate step from the rest of the install, rather than an
# --extra-index-url on one command, because with two indexes pip is free to
# pick either copy of the same version number and the choice is not
# reproducible. Installing torch alone here means the project install below
# finds the requirement already satisfied and never reconsiders it.
#
# This is also the layer that makes rebuilds bearable: it depends on nothing in
# the source tree, so editing the API never re-downloads it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

# The metadata files the build backend reads (pyproject declares README.md and
# LICENSE), then the package itself.
#
# Honest limitation: because setuptools needs src/ present to build the
# package, dependencies and source arrive in the same layer, so editing a
# source file re-resolves the small dependencies -- fastapi, numpy, pandas,
# scikit-learn -- on the next build. Roughly a minute. Splitting them would
# mean maintaining a second, generated list of dependencies alongside
# pyproject.toml, and a list that can disagree with the source of truth is a
# worse problem than a slow rebuild. torch, the expensive one, is cached above.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# No -e: an editable install leaves the venv pointing at /build/src, which does
# not exist in the next stage. A regular install copies the package into
# site-packages, so /opt/venv is self-contained.
RUN pip install .

# ---------------------------------------------------------------------------
# Stage 2: the image that actually runs
# ---------------------------------------------------------------------------
# Same base as the builder, deliberately: the wheels installed above were
# linked against this image's glibc, and a different runtime base can produce
# an import error that appears only in production.
FROM python:3.12-slim-bookworm AS runtime

# PYTHONDONTWRITEBYTECODE: .pyc files written at runtime land in the
#   container's writable layer, where they are discarded on every restart.
# PYTHONUNBUFFERED: without it Python block-buffers stdout when it is a pipe --
#   which it always is under Docker -- so logs appear in bursts, or not at all
#   if the process is killed. Nearly every containerised Python image sets this.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as a normal user, not root. A container is not a security boundary in the
# way a VM is: root inside the container is uid 0 on the host kernel, so an
# escape starts from a much stronger position. The uid is fixed rather than
# auto-assigned so that file ownership stays predictable if a volume is ever
# mounted in.
RUN useradd --create-home --uid 10001 appuser

# The finished environment, in one layer, owned by the user that will run it.
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

WORKDIR /app

# The checkpoint is baked into the image rather than mounted at runtime. That
# makes the image self-contained: what was tested is what deploys, and the host
# needs no persistent disk -- which most free tiers do not offer anyway. The
# cost is worth stating plainly: retraining means rebuilding the image, and it
# is why this one artifact is committed to git as an explicit exception to the
# rule that build outputs are not.
COPY --chown=appuser:appuser models/served.pt /app/models/served.pt

# Configuration by environment variable, read by api.service.resolve_model_path.
# The same image can therefore be pointed at a different checkpoint without a
# rebuild, which is what keeps "build once, deploy anywhere" true.
ENV MODEL_PATH=/app/models/served.pt

USER appuser

# EXPOSE publishes nothing. It is metadata: a note to a human reader and to
# tooling that this image listens on 8000. Reaching it still requires
# `docker run -p 8000:8000`, which is the host's decision, not the image's.
EXPOSE 8000

# Docker's own liveness probe, so `docker ps` reports health rather than merely
# "up". curl is not installed in -slim and adding it for this would be another
# package to keep patched, so the check uses the interpreter already here.
# start-period covers the few seconds torch and the checkpoint take to load;
# without it the container would be marked unhealthy while it is still
# legitimately starting.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# Two details in one line.
#
# ${PORT:-8000}: hosts including Azure Container Apps and Cloud Run inject the
# port they want the container to listen on. Honouring PORT now means Phase 8
# does not require a code change. Environment variables are expanded by a
# shell, not by Docker, which is why this is `sh -c` and not the plain exec
# form used elsewhere.
#
# exec: without it, sh stays alive as PID 1 with uvicorn as its child, and PID 1
# does not forward signals by default. `docker stop` would send SIGTERM to a
# shell that ignores it, wait ten seconds, then SIGKILL everything -- turning
# every routine restart into a hard kill mid-request. `exec` replaces the shell
# with uvicorn so the signal arrives where it can be handled and connections
# drain.
#
# One worker, no gunicorn. A second process would double resident memory for a
# second copy of torch and the model, on a host sized in fractions of a CPU.
# The endpoints already run inference in a threadpool (see api.main), so a
# single worker does not serialise requests; adding a process manager is a
# scaling decision to make with a measurement, not in advance.
CMD ["sh", "-c", "exec uvicorn molecular_property_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
