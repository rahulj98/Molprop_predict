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

Build a small, honest, end-to-end ML project: predict a molecular property from the QM9 dataset, serve the trained model as an API, track experiments properly, and containerize it. Hosting it behind a public URL was the original last step and was **dropped deliberately** — the reasoning is under Phase 8 below. The container is the deliverable, and `docker build && docker run` is the reproduction path.

**Where the project is now.** Part I — a *fixed* molecular representation feeding a small feed-forward network — is complete and tagged `v1.0.0`. Part II asks whether a model that **learns** the representation does better, and is in progress. Both roadmaps are below; `git log` is the authoritative record, one commit per phase.

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
- **Linting**: Ruff — one tool in place of the flake8 + isort + black + pyupgrade + bandit stack it replaces. Same rules, one dependency, one config block, and fast enough (~0.15 s over the whole repo) that it runs on save and in CI without anyone waiting for it. The rule selection and every exception to it are commented in `pyproject.toml`; `notebooks/` is excluded, because those are committed *with their outputs* and satisfying a linter would mean re-executing them and rewriting the numbers the prose quotes.
- **CI**: GitHub Actions — runs lint and tests on a clean Ubuntu checkout for every push to `main` and every pull request. The point is not the checks themselves, which run locally too, but that they run somewhere that is *not* this laptop: no editable install left over from last week, no package installed once and never declared. This project has already been bitten by exactly that — `fastapi` and `uvicorn` were importable only via `mlflow`'s dependency tree until they were declared properly.
- **Part II model**: a message-passing graph neural network, hand-rolled in `src/molecular_gnn/` rather than taken from PyTorch Geometric — the point is to understand the mechanism, and a library that hides message passing behind one call defeats that
- **Repo structure**: standard Python package layout, `pytest` for tests, `README.md` as the public front door

## Roadmap (build in this order)

### Part I — a fixed representation ✅ complete, tagged `v1.0.0`

All of the below shipped except Phase 8, which was dropped on purpose. One commit per phase; the commit messages carry the reasoning and are the record. Kept here because the *why* of each phase is still the best short description of what the code does and should not be deleted just because the work is finished.

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

**Landed outside the roadmap, after v1.0.0.** Ruff, a GitHub Actions CI job, and two performance fixes (commit `c898a00`), then a housekeeping pass (`11d24f9`). Neither commit was a numbered phase — both follow from the Definition of Done below rather than from the plan, and are recorded here so the roadmap is not quietly contradicted by the history.

### Part II — a learned representation (in progress)

Everything in Part I hands the model a **fixed** description of a molecule: a 2012-era sorted Coulomb matrix, decided once and never adjusted. Part I's best is 0.244 eV on the sealed test set; published work reaches roughly 0.02–0.04 eV, about ten times better, using graph neural networks that **learn** the description instead. The remaining gap is the representation, not the model size — a wider feed-forward network would not close it. Part II is that experiment.

Two structural decisions already made, and worth holding to:

- **`src/molecular_gnn/` is a second package that imports nothing from the first.** A comparison between two representations is worth little if both sides lean on the same loader, split or featuriser — a difference could then come from the helper rather than the model. The duplication that buys (its own parser, its own split) is made safe by `tests/gnn/test_equivalence.py`, the one module that imports both and asserts they agree molecule for molecule and select identical splits.
- **The same sealed-test discipline as Part I.** Every number reported while choosing anything is a *validation* number. The test split opens once, on one frozen model, at the end.

- **Phase 10 — Graph data** ✅ done (`c626e60`). Its own QM9 parser, a random split and a **scaffold split** (Murcko scaffolds via RDKit, keeping whole ring systems on one side so the test set contains skeletons no model has seen), a radius graph, a Gaussian radial basis, a cosine cutoff envelope, and batching by concatenation with a `batch` index for pooling. 53 tests. No model consumes any of it yet.
- **Phase 11 — The network**: a message-passing model over those graphs. Explain the mechanism concept by concept — what a message is, why it is a function of the neighbour and the edge, why aggregation must be permutation-invariant, and how pooling turns per-atom vectors back into one number per molecule. Hand-rolled rather than PyTorch Geometric, for the reason in the tech stack above.
- **Phase 12 — Training**: reuse the loop discipline from Phase 4, not the code. Score against Part I's validation number on the **random** split first, since that is the only like-for-like comparison available.
- **Phase 13 — The comparison**: score both models under the scaffold split. This is the honest test of whether either generalises to genuinely new chemistry rather than interpolating within a chemical space, and it is expected to make both look worse. Report it anyway.

## Definition of done (whole project)

- Repo is clean, documented, and something I could screen-share in an interview without embarrassment
- Every tool in the stack is one I can explain the purpose of, unprompted
- README.md is understandable by someone with no ML background
- No fabricated metrics, no copy-pasted boilerplate I can't explain
- `ruff check` and `pytest` pass, and pass **in CI on a clean checkout** — not only on this laptop
- This file agrees with the repository. It is the instructions, so a stale roadmap here is worse than a stale note anywhere else: it misdirects the next session's work. It has drifted twice — claiming a deployment that was dropped, and stopping at Phase 9 while Part II was underway — so check it when a phase closes.
