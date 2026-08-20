# CLAUDE.md

This file is instructions **for Claude Code**, read automatically when working in this repo in PyCharm. It tells you how to behave on this project, not just what to build.

---

## Background and working requirements

Author: Rahul, computational scientist (chemistry/physics, 8+ years). Established ground:
- Python (NumPy, Pandas, SciPy, scikit-learn, OOP/class-hierarchy package design)
- FORTRAN90, Bash, Linux, HPC/compute-cluster automation
- Git, GitLab CI/CD, pytest, formal code review
- Statistical modeling

This project deliberately moves into adjacent territory: the MLOps toolchain — PyTorch as a production artifact, FastAPI, MLflow, Docker, cloud deployment — rather than the scientific computing and statistical modeling that's already familiar. Working in the unfamiliar half is the point of building it.

**Working requirement:** when introducing a tool or pattern from that toolchain, explain the *why* alongside the *how* — what problem it solves, and why the industry converged on this approach rather than an obvious simpler one. Don't treat the rationale as self-evident or skip it as jargon. Reading code is not the constraint; the design intent behind unfamiliar tooling is.

## Project goal

Build a small, honest, end-to-end ML project: predict a molecular property from the QM9 dataset, serve the trained model as an API, track experiments properly, containerize it, and deploy it somewhere reachable by a URL.

This is a **portfolio/CV project**. Two things matter more than raw model accuracy:
1. **I can defend every part of it in an interview.** Don't add a tool or technique I can't explain in my own words by the time we're done.
2. **It's honest.** No inflated claims in code comments, README, or commit messages. If something is a simplified/toy version of a real production pattern, say so.

## How to work with me — process rules

1. **Work in phases, in order** (see Roadmap below). Do not jump ahead or scaffold later phases early, even if it seems efficient — I need to actually absorb each phase.
2. **Before starting a phase**, give me a short plan: what we're building, which new concepts/tools appear, and why this phase exists. Wait for me to confirm before writing code.
3. **After finishing a phase**, stop and summarize:
   - What we built
   - The 2–3 core concepts I should now understand
   - What I should be able to explain if asked about it in an interview
   Then wait for me to say "continue" before moving to the next phase.
4. **Checkpoint with git** at the end of every phase: help me write a clear commit message, commit, and (from phase 1 onward) push. Small, honest commit history is part of the point of this project.
5. **Prefer explaining over doing when I ask "why."** If I ask why we're using a tool a certain way, answer in plain language first; only show code after.
6. **Write tests as we go**, not as an afterthought — I already work this way (pytest), keep that habit here.
7. **Idiomatic, not clever.** Real-world code — type hints, docstrings, clean structure — but don't reach for advanced or unusual patterns without explaining them first.
8. **No secrets or credentials in code or commits.** Cloud keys, API tokens, etc. go in `.env` (gitignored) — explain the convention the first time it comes up.

## Tech stack (target)

- **Data**: QM9 dataset (public, ~134K small organic molecules)
- **Features**: molecular descriptors (we'll reuse the conceptual approach I already know from hand-coding Coulomb matrix / bag-of-bonds / ACSF representations previously — decide together whether to hand-roll a simplified version again or use a library, and document the choice)
- **Model**: PyTorch (start with a simple feed-forward network; a baseline scikit-learn model comes first for comparison — I already have real experience there)
- **Experiment tracking**: MLflow
- **Serving**: FastAPI
- **Containerization**: Docker
- **Deployment**: out of scope — see Phase 8 in the roadmap below
- **Repo structure**: standard Python package layout, `pytest` for tests, `README.md` as the public front door

## Roadmap (build in this order)

- **Phase 0 — Environment setup**: PyCharm project, virtual environment, git init, GitHub repo, `.gitignore`, dependency management approach (explain `requirements.txt` vs `pyproject.toml` and pick one, with reasoning).
- **Phase 1 — Data**: download/load QM9 (or a manageable subset), explore it, understand what we're predicting and why it's a reasonable target property. Explain train/validation/test split from first principles.
- **Phase 2 — Features**: turn raw molecule data into model-ready features. Explain what a "feature" means in ML terms and how it maps to what I already know from Coulomb matrices etc.
- **Phase 3 — Baseline model**: simple scikit-learn regression as a sanity-check baseline before touching PyTorch. Explain why a baseline matters.
- **Phase 4 — PyTorch model**: build and train a real (small) neural network. Explain the training loop concept-by-concept (forward pass, loss, backward pass, optimizer step) — don't assume this is obvious.
- **Phase 5 — MLflow**: add experiment tracking to Phase 4's training loop. Explain what problem MLflow solves that just printing results to console doesn't.
- **Phase 6 — FastAPI**: wrap the trained model in a REST API with at least one prediction endpoint. Explain REST basics as needed.
- **Phase 7 — Docker**: containerize the API. Explain what a container actually is and why we don't just "run the Python file" in production.
- **Phase 8 — Cloud deployment: dropped, deliberately.** The original plan was to deploy the container to a free tier and expose a public URL. Every option that can host a 2 GB PyTorch container now requires either a credit card on file (Azure, AWS, GCP) or a paid subscription (Hugging Face Docker Spaces). Render's free tier would have worked — the container was measured at 373–386 MB against a 512 MB cap — but by then the more useful question had been answered: this repository is meant to be *read and reproduced*, not consumed as a hosted service. The container is the deliverable, and `docker build && docker run` is the reproduction path. Recorded here rather than quietly skipped.
- **Phase 9 — Polish for reading**: make the repo work for someone who lands on it and wants to do the same thing for their own problem. Clone-and-reproduce instructions, a guide to adapting it to a different target property, a README that orients a reader in two minutes, and a figure or two.

## Definition of done (whole project)

- Repo is clean, documented, and something I could screen-share in an interview without embarrassment
- Every tool in the stack is one I can explain the purpose of, unprompted
- README.md is understandable by someone with no ML background
- No fabricated metrics, no copy-pasted boilerplate I can't explain
