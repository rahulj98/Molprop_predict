"""Download, parse, and split the QM9 dataset.

QM9 (Ramakrishnan et al., *Scientific Data* **1**, 140022, 2014) contains
133,885 small organic molecules built from H, C, N, O and F, with up to nine
heavy atoms. Each molecule carries 15 scalar properties computed at the
B3LYP/6-31G(2df,p) level of theory, plus the relaxed 3D geometry.

We read the *original* published tarball from figshare rather than a
preprocessed CSV mirror. That costs us a parser, and buys two things: the 3D
coordinates (which a SMILES-only mirror discards, and which any
geometry-based representation needs), and a data provenance we can point at
by DOI.

Source: https://doi.org/10.6084/m9.figshare.978904
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# --- Locations --------------------------------------------------------------

#: Repository root, derived from this file's location rather than from the
#: current working directory. A notebook launched from ``notebooks/`` and a
#: test launched from the repo root must resolve ``data/`` to the same place --
#: anchoring on the working directory instead silently gives them two separate
#: copies of the dataset.
#:
#: This holds because the package is installed editable (``pip install -e .``),
#: so ``__file__`` still points into ``src/``. Phase 7 will pass container
#: paths explicitly rather than relying on it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Provenance -------------------------------------------------------------

QM9_XYZ_URL = "https://ndownloader.figshare.com/files/3195389"
QM9_UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"

#: Molecules in the published tarball, before any exclusions.
N_MOLECULES_PUBLISHED = 133_885

#: Molecules the authors flagged as failing a geometric consistency check:
#: the relaxed B3LYP geometry and the original Corina geometry parse to
#: different SMILES. We drop these rather than quietly keeping them.
N_UNCHARACTERIZED_PUBLISHED = 3_054

# --- Units ------------------------------------------------------------------

#: CODATA 2018. Several QM9 properties are published in Hartree; we convert
#: the energetic ones to eV so errors are comparable to the published
#: literature, which quotes eV almost universally.
HARTREE_TO_EV = 27.211386245988

#: The 15 scalar properties, in the order they appear on line 2 of each file.
PROPERTY_NAMES: tuple[str, ...] = (
    "A",      # GHz       rotational constant
    "B",      # GHz       rotational constant
    "C",      # GHz       rotational constant
    "mu",     # Debye     dipole moment
    "alpha",  # a0^3      isotropic polarizability
    "homo",   # Hartree   highest occupied molecular orbital energy
    "lumo",   # Hartree   lowest unoccupied molecular orbital energy
    "gap",    # Hartree   lumo - homo
    "r2",     # a0^2      electronic spatial extent
    "zpve",   # Hartree   zero point vibrational energy
    "U0",     # Hartree   internal energy at 0 K
    "U",      # Hartree   internal energy at 298.15 K
    "H",      # Hartree   enthalpy at 298.15 K
    "G",      # Hartree   free energy at 298.15 K
    "Cv",     # cal/mol/K heat capacity at 298.15 K
)

#: Properties published in Hartree, which we convert to eV on load.
HARTREE_PROPERTIES = frozenset({"homo", "lumo", "gap", "zpve", "U0", "U", "H", "G"})

#: The property this project predicts, chosen in Phase 1 with the evidence in
#: ``notebooks/01_explore_qm9.ipynb``.
#:
#: The choice is not arbitrary. Fitting ordinary least squares on *atom counts
#: alone* -- a model that knows no chemistry at all -- already reaches
#: R^2 = 1.0000 on U0, U, H and G, and 0.9971 on zpve. Those targets leak: a
#: model can score brilliantly on them while having learned only how to count
#: atoms. The same chemistry-free model reaches R^2 = 0.4555 on ``lumo``, so
#: more than half its variance genuinely requires structural information.
TARGET_PROPERTY = "lumo"

#: Mean absolute error of always predicting the mean of ``TARGET_PROPERTY``,
#: in eV, over the full filtered dataset. Any model that cannot beat this has
#: contributed nothing.
TARGET_MEAN_BASELINE_MAE_EV = 1.0510

#: QM9 contains only these five elements.
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def _parse_float(token: str) -> float:
    """Parse one numeric token from a QM9 ``.xyz`` file.

    The files were generated with Mathematica, which writes exponents as
    ``1.234*^-6`` instead of the ``1.234e-6`` that :func:`float` understands.
    Only some columns are ever affected, but it is cheaper to normalise every
    token than to guess which.
    """
    return float(token.replace("*^", "e"))


def parse_xyz(text: str) -> dict:
    """Parse a single QM9 ``.xyz`` file into a flat record.

    The format, per the paper's Table 2:

    ==========  ================================================
    Line        Contents
    ==========  ================================================
    1           number of atoms, ``na``
    2           ``gdb``, molecule index, then the 15 properties
    3..na+2     element, x, y, z, Mulliken partial charge
    na+3        harmonic vibrational frequencies
    na+4        SMILES, from GDB-17 and from the relaxed geometry
    na+5        InChI, from both geometries
    ==========  ================================================

    Vibrational frequencies are deliberately not retained: nothing in this
    project consumes them, and they are variable-length per molecule.
    """
    lines = text.splitlines()
    n_atoms = int(lines[0])

    header = lines[1].split()
    # header[0] is the literal tag "gdb"; header[1] is the molecule index.
    record: dict = {"index": int(header[1]), "n_atoms": n_atoms}

    for name, token in zip(PROPERTY_NAMES, header[2:], strict=True):
        value = _parse_float(token)
        record[name] = value * HARTREE_TO_EV if name in HARTREE_PROPERTIES else value

    elements: list[str] = []
    coordinates: list[float] = []
    charges: list[float] = []
    for line in lines[2 : 2 + n_atoms]:
        element, x, y, z, charge = line.split()
        elements.append(element)
        coordinates.extend((_parse_float(x), _parse_float(y), _parse_float(z)))
        charges.append(_parse_float(charge))

    record["elements"] = elements
    record["atomic_numbers"] = [ATOMIC_NUMBERS[e] for e in elements]
    # Flattened row-major; reshape to (n_atoms, 3) via `geometry()`. Parquet
    # stores this as a list<double> column without losing the nesting.
    record["coordinates"] = coordinates
    record["mulliken_charges"] = charges

    # Two tab-separated SMILES: the original GDB-17 string, and the one
    # obtained by re-perceiving the relaxed B3LYP geometry. They disagree
    # exactly for the "uncharacterized" molecules we exclude.
    smiles = lines[2 + n_atoms + 1].split()
    record["smiles_gdb17"] = smiles[0]
    record["smiles_relaxed"] = smiles[-1]

    inchi = lines[2 + n_atoms + 2].split()
    record["inchi_gdb17"] = inchi[0]
    record["inchi_relaxed"] = inchi[-1]

    return record


def geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(atomic_numbers, coordinates)`` for one molecule.

    Coordinates come back shaped ``(n_atoms, 3)`` in Angstrom, which is the
    layout any geometry-based representation will expect.
    """
    atomic_numbers = np.asarray(row["atomic_numbers"], dtype=np.int64)
    coordinates = np.asarray(row["coordinates"], dtype=np.float64).reshape(-1, 3)
    return atomic_numbers, coordinates


def _download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination``, skipping if it already exists.

    Raw downloads are gitignored, so this is what makes a fresh clone
    reproducible without committing 82 MB of data.
    """
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary name first, so an interrupted download cannot
    # leave a truncated file that later runs would mistake for complete.
    partial = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https URL
    partial.replace(destination)
    return destination


def load_uncharacterized(path: Path) -> set[int]:
    """Read the indices of molecules that failed the consistency check.

    The file is a human-readable report with a prose header; the data rows are
    the ones whose first token is an integer index.
    """
    indices: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens and tokens[0].isdigit():
            indices.add(int(tokens[0]))
    return indices


def load_qm9(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    *,
    exclude_uncharacterized: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download (once), parse, and return QM9 as a DataFrame.

    Parsing 133,885 small files takes a couple of minutes, so the result is
    cached as Parquet. Parquet rather than CSV because CSV cannot hold the
    variable-length coordinate arrays and silently loses dtypes.

    Args:
        raw_dir: Where the downloaded archive lives.
        processed_dir: Where the parsed cache is written.
        exclude_uncharacterized: Drop the 3,054 molecules the authors flagged.
        use_cache: Reuse an existing Parquet cache if present.

    Returns:
        One row per molecule. Energetic properties are in eV, not Hartree.
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    cache = processed_dir / (
        "qm9.parquet" if exclude_uncharacterized else "qm9_all.parquet"
    )

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    archive = _download(QM9_XYZ_URL, raw_dir / "dsgdb9nsd.xyz.tar.bz2")

    records = []
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            if not member.name.endswith(".xyz"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            records.append(parse_xyz(handle.read().decode("utf-8")))

    frame = pd.DataFrame.from_records(records).sort_values("index", ignore_index=True)

    if exclude_uncharacterized:
        flagged_path = _download(
            QM9_UNCHARACTERIZED_URL, raw_dir / "uncharacterized.txt"
        )
        flagged = load_uncharacterized(flagged_path)
        frame = frame[~frame["index"].isin(flagged)].reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test by random permutation.

    Three sets, not two, because the moment you choose anything by looking at
    a set -- learning rate, network width, when to stop -- that set stops
    being an unbiased estimate of unseen performance. Train fits parameters,
    validation guides our choices, test is touched once at the very end.

    ``seed`` makes the split reproducible: the same seed always yields the
    same partition, so a result from Phase 3 stays comparable to Phase 4.

    Note the honest caveat: a *random* split is the QM9 convention, but it is
    optimistic. Structurally near-identical molecules land on both sides, so
    the test score measures interpolation within a chemical space rather than
    generalisation to genuinely new chemistry. A scaffold split would be
    stricter.

    Returns:
        ``(train, validation, test)``, disjoint, together covering ``frame``.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must each lie in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac must be < 1 to leave a test set, "
            f"got {train_frac} + {val_frac} = {train_frac + val_frac}"
        )

    permutation = np.random.default_rng(seed).permutation(len(frame))
    n_train = int(len(frame) * train_frac)
    n_val = int(len(frame) * val_frac)

    shuffled = frame.iloc[permutation].reset_index(drop=True)
    return (
        shuffled.iloc[:n_train].reset_index(drop=True),
        shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True),
        shuffled.iloc[n_train + n_val :].reset_index(drop=True),
    )
