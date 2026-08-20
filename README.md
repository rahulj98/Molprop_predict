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

> **One caveat, stated up front rather than buried.** "Send it a molecule" means sending 3D atomic coordinates, and the model expects the *relaxed* geometry that the original quantum chemistry calculation produced. So this replaces the property calculation, not the geometry optimisation that comes before it. Feeding it a rough or approximate geometry will make the prediction worse, and nothing in the response will warn you. The speed-up is real, but it is a speed-up of one step, not of the whole pipeline.

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

## How well does it work?

Errors are in electronvolts (eV). For scale, the property being predicted spans roughly −4.8 to +5.3 eV across the dataset, and always guessing the average would be wrong by 1.05 eV.

| Model | Error (MAE) | Variance explained (R²) |
|---|---|---|
| Always guess the average | 1.055 eV | 0.00 |
| Ridge regression on atom counts | 0.769 eV | 0.46 |
| Gradient boosting on geometry | 0.369 eV | 0.86 |
| **Neural network (this project)** | **0.248 eV** | **0.92** |

All four rows are scored on the same *validation* set, so they are directly comparable — that set is what every modelling choice in the project was made against.

The final model was then scored once on a **test** set it had never touched, and which no decision was ever based on: **0.244 eV**, R² 0.92. That is the honest estimate of how it performs on unseen molecules, and the fact that it is close to the validation figure indicates the choices made along the way were not overfitted to the validation set.

**Where it falls short, honestly:**

- Published models reach roughly 0.02–0.04 eV on this task, about ten times better. They use graph neural networks that learn how to represent a molecule instead of being handed a fixed 2012-era recipe, as here. A bigger version of this network would not close that gap.
- Predictions at the extremes get pulled toward the average. The model is most reliable on typical molecules and least reliable on unusual ones — which, in a screening application, are the interesting ones.
- Test molecules are drawn randomly, so most have close relatives in the training data. Performance on a genuinely novel class of molecule would be worse.

## Try it yourself

The quickest route needs only Docker — no Python, no virtual environment, no downloading the dataset. The trained model is inside the image.

```bash
git clone https://github.com/rahulj98/Molprop_predict
cd Molprop_predict

docker build -t molecular-property-predictor .
docker run --rm -p 8000:8000 molecular-property-predictor
```

The service is then at `http://localhost:8000`, with `docker ps` reporting `healthy` once the model has loaded. `pytest -m docker` runs a small suite against the built image itself — that the checkpoint is really inside it, that a prediction survives the trip over a socket, and that the process is not running as root. The other 217 tests run in-process and need no Docker (`pytest -m "not docker"`).

<details>
<summary>Or run it directly with Python</summary>

```bash
pip install -e ".[dev]"

# Point the service at a trained checkpoint and start it
export MODEL_PATH=models/served.pt      # Windows: set MODEL_PATH=models\served.pt
uvicorn molecular_property_predictor.api.main:app --reload
```

</details>

Then send it methane:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "atomic_numbers": [6, 1, 1, 1, 1],
        "coordinates": [
          [-0.0127,  1.0858,  0.0080],
          [ 0.0022, -0.0060,  0.0020],
          [ 1.0117,  1.4638,  0.0003],
          [-0.5408,  1.4475, -0.8766],
          [-0.5238,  1.4379,  0.9064]
        ]
      }'
```

```json
{"lumo_ev":2.003439426422119,"units":"eV","n_atoms":5}
```

(The reference value for methane is +3.19 eV. That is an unusually large error, and deliberately so as an example — methane sits at the very edge of the dataset's range, where the model is at its weakest. See the "where it falls short" note above, and `notebooks/06_api.ipynb` for the full explanation.)

Interactive documentation is generated automatically at `http://localhost:8000/docs`, and `GET /model` reports exactly which checkpoint is loaded and how accurate it measured.

Trained models are build outputs — reproducible from the code and the recorded random seeds — so `models/` is not in this repository, with one deliberate exception: `models/served.pt`, the 4.2 MB checkpoint the API serves, is committed so that `docker build` works from a clone. Every other checkpoint is produced by running the notebooks in order.

### About the image

It is **2.09 GB**, which is large for a web service and worth explaining rather than hiding. PyTorch alone is 750 MB of that; the scientific Python stack underneath it (SciPy, pandas, PyArrow, scikit-learn, NumPy) is another 450 MB. The Dockerfile already takes the single biggest saving available — installing the CPU-only build of PyTorch instead of the default one, which would drag in ~2.5 GB of CUDA libraries for a machine with no GPU — and the remainder is essentially the cost of shipping PyTorch at all. Getting substantially below this would mean exporting the model to a lighter inference runtime such as ONNX Runtime, which is a real option, and a different project.

## Project status

Phases 0–7 complete: data, features, baseline models, neural network, experiment tracking, a working API, and a container image. Still to come: cloud deployment (Phase 8) and final polish (Phase 9).

Each phase is committed separately so the build process itself is visible, not just the finished result. The `notebooks/` directory contains the executed analysis for every phase, including the negative results.

## What this project is *not*

To keep this honest: this is a learning and portfolio project, not a production system. The model is intentionally small and the dataset intentionally scoped down for buildability. The value of the repo is in demonstrating the *full pipeline* correctly and understandably, not in state-of-the-art accuracy.

## Tech stack

Python · PyTorch · scikit-learn · FastAPI · MLflow · Docker · (Azure/AWS — TBD)

## Author

Rahul Kumar Jingar — [github.com/rahulj98](https://github.com/rahulj98)
