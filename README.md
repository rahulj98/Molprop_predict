# Molecular Property Predictor

A small, end-to-end machine learning project that predicts a physical property of a molecule (for example, how much energy sits in its lowest empty electron orbital) directly from its structure — and serves that prediction through a live web API.

You do **not** need a machine learning or chemistry background to follow this README. Every section explains what's happening and why, in plain language, before showing anything technical.

---

## What does this project actually do?

1. It takes a public dataset of about 134,000 small molecules, each with known physical properties (these were originally calculated using expensive quantum chemistry software).
2. It trains a model that learns the relationship between a molecule's structure and one of those properties.
3. Once trained, the model can predict that property for a *new* molecule it has never seen, in a fraction of a second — instead of running an hours-long quantum chemistry calculation.
4. That trained model is wrapped in a small web service, so anyone (or any other piece of software) can send it a molecule and get a prediction back over the internet.

This mirrors a real pattern used in drug discovery and materials science: expensive simulations are slow, so a trained model is used to screen thousands of candidates cheaply, and only the most promising ones get the expensive, exact calculation.

## Why build this?

This project exists to demonstrate, in one small self-contained repo, the full path from "raw scientific data" to "a model anyone can query over the web" — the same shape of work used in applied data science and ML engineering roles, just at a scale one person can build and fully understand.

## The dataset in plain terms

The project uses **QM9**, a well-known public dataset in computational chemistry. It contains around 134,000 small organic molecules (think: molecules with up to 9 heavy atoms — carbon, oxygen, nitrogen, fluorine), each with a set of physical properties that were calculated using quantum chemistry simulations. Those simulations are accurate but slow. This project trains a model to approximate one of those properties almost instantly.

## The moving parts, explained without jargon

| Piece | What it's for, in plain terms |
|---|---|
| **Feature extraction** | Turning a molecule's structure into a list of numbers a model can actually work with. Models can't read a chemical structure directly — they need it translated into a consistent numerical format first. |
| **Model (PyTorch)** | The part that learns the pattern between "numbers describing a molecule" and "the property we want to predict." Think of it as a very flexible curve-fitting function that adjusts itself as it sees more examples. |
| **Experiment tracking (MLflow)** | A logbook for every training attempt — what settings were used, how accurate the result was. Without it, you lose track of which version of the model actually worked best after a few dozen attempts. |
| **API (FastAPI)** | A small web service that takes a request ("here's a molecule") and returns a response ("here's the predicted property"), so the model is usable by anything that can make a web request — a script, a browser, another application. |
| **Container (Docker)** | A way of packaging the code, the trained model, and everything it needs to run, into one predictable unit — so it behaves the same on any machine, instead of "works on my laptop." |
| **Cloud deployment** | Putting that container on a server so it has a real, public web address instead of only running on one person's computer. |

## Architecture, top to bottom

```
QM9 dataset
    │
    ▼
Feature extraction  (molecule → numbers)
    │
    ▼
Model training  (PyTorch, tracked with MLflow)
    │
    ▼
Trained model saved to disk
    │
    ▼
FastAPI service loads the model
    │
    ▼
Docker container packages the service
    │
    ▼
Deployed to the cloud → public URL
    │
    ▼
Anyone sends a molecule → gets a prediction back
```

## Try it yourself

> This section gets filled in as the project is built — instructions will appear here for running it locally and for hitting the live demo endpoint once deployed.

```bash
# Example of what this will look like once built:
git clone <this-repo>
cd molecular-property-predictor
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Then send a request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"molecule": "<example input format goes here>"}'
```

## Project status

Actively being built, in public, step by step. See commit history for progress — each phase (data → features → baseline model → PyTorch model → tracking → API → containerization → deployment) is committed separately so the build process itself is visible, not just the finished result.

## What this project is *not*

To keep this honest: this is a learning and portfolio project, not a production system. The model is intentionally small and the dataset intentionally scoped down for buildability. The value of the repo is in demonstrating the *full pipeline* correctly and understandably, not in state-of-the-art accuracy.

## Tech stack

Python · PyTorch · scikit-learn · FastAPI · MLflow · Docker · (Azure/AWS — TBD)

## Author

Rahul Kumar Jingar — [github.com/rahulj98](https://github.com/rahulj98)
