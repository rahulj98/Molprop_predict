"""Load QM9 and split it -- independently of the first package.

This module reads the same published archive and keeps only what a graph model
needs: the molecule index, its atoms, their positions, the target property, and
the SMILES string a scaffold split is computed from. It deliberately drops the
Mulliken charges, the rotational constants and the twelve properties this work
does not predict, because a loader that carries everything is a loader nobody
reads.

**Two splits, for two different questions.**

:func:`random_split` reproduces the partition the first package used, so the
graph network can be compared against a measured baseline on exactly the same
molecules. It is not *imported* from there -- it is reimplemented from the same
stated rule, and ``tests/gnn/test_equivalence.py`` asserts the two agree
molecule for molecule. That is the seam where independence and comparability
meet, and a test is the right place for it because drift there would otherwise
be silent.

:func:`scaffold_split` answers the harder question. A random split puts
structurally near-identical molecules on both sides, so it measures
interpolation within a chemical space. Grouping molecules by their Murcko
scaffold -- the ring system left after stripping substituents -- and keeping
whole groups together means the test set contains skeletons the model has never
seen. Every model scored this way does worse. That is the point: it is the
number that describes use on new chemistry.
"""

from __future__ import annotations

import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --- Locations --------------------------------------------------------------

#: Repository root, from this file rather than the working directory, so a
#: notebook and a test resolve ``data/`` to the same place.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Provenance -------------------------------------------------------------

#: Ramakrishnan et al., *Scientific Data* **1**, 140022 (2014).
#: https://doi.org/10.6084/m9.figshare.978904
QM9_XYZ_URL = "https://ndownloader.figshare.com/files/3195389"
QM9_UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"

#: CODATA 2018. QM9 publishes the orbital energies in Hartree; the literature
#: quotes errors in eV, so the target is converted on load and never again.
HARTREE_TO_EV = 27.211386245988

#: The five elements QM9 contains.
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

#: Position of each property in the header line, per the paper's Table 2. Only
#: the ones this project reads are named; the rest are skipped by index.
#: ``lumo`` is entry 7 counting from the first property (A, B, C, mu, alpha,
#: homo, lumo).
_LUMO_HEADER_OFFSET = 8  # header[0] = "gdb", header[1] = index, then A onwards

#: Cached parse, separate from the first package's cache so neither can be
#: mistaken for the other's output.
CACHE_NAME = "qm9_gnn.parquet"


def _parse_float(token: str) -> float:
    """Parse a QM9 float, including Mathematica's ``*^`` exponent notation.

    The published files contain values such as ``1.234*^-6``, which Python's
    ``float`` rejects. Silently dropping those molecules would bias the dataset
    in a way nothing downstream could detect.
    """
    return float(token.replace("*^", "e"))


def parse_xyz(text: str) -> dict:
    """Parse one QM9 ``.xyz`` file, keeping only what a graph model needs.

    Returns:
        ``index``, ``n_atoms``, ``atomic_numbers``, ``coordinates`` (flattened
        row-major), ``lumo`` in eV, and ``smiles_relaxed``.
    """
    lines = text.splitlines()
    n_atoms = int(lines[0])
    header = lines[1].split()

    atomic_numbers: list[int] = []
    coordinates: list[float] = []
    for line in lines[2 : 2 + n_atoms]:
        element, x, y, z, _charge = line.split()
        atomic_numbers.append(ATOMIC_NUMBERS[element])
        coordinates.extend((_parse_float(x), _parse_float(y), _parse_float(z)))

    # Two tab-separated SMILES on this line: the original GDB-17 string and the
    # one re-perceived from the relaxed geometry. The relaxed one describes the
    # structure the coordinates actually belong to, so scaffolds come from it.
    smiles = lines[2 + n_atoms + 1].split()

    return {
        "index": int(header[1]),
        "n_atoms": n_atoms,
        "atomic_numbers": atomic_numbers,
        "coordinates": coordinates,
        "lumo": _parse_float(header[_LUMO_HEADER_OFFSET]) * HARTREE_TO_EV,
        "smiles_relaxed": smiles[-1],
    }


def _download(url: str, destination: Path) -> Path:
    """Fetch ``url`` to ``destination`` unless it is already there.

    The download lands on a temporary name and is renamed into place only once
    it completes. Writing straight to ``destination`` means an interrupted
    download leaves a truncated file that every later run treats as complete --
    and a half-written 82 MB tarball fails much further downstream than the
    place that actually broke.
    """
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https URL
    partial.replace(destination)
    return destination


def load_uncharacterized(path: Path) -> set[int]:
    """Read the indices the dataset authors flagged as inconsistent.

    These are molecules whose relaxed geometry re-perceives to a different
    SMILES than the one they started from, so the structure and the properties
    describe different things. Keeping them would put unlearnable rows in the
    training set and unanswerable ones in the test set.
    """
    flagged: set[int] = set()
    # Encoding stated explicitly: `read_text()` defaults to the platform's
    # locale encoding, which is cp1252 on Windows and UTF-8 elsewhere. Letting
    # it default makes the parse machine-dependent for no reason.
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens and tokens[0].isdigit():
            flagged.add(int(tokens[0]))
    return flagged


def load_qm9(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download (once), parse, and return QM9 as a dataframe.

    Rows are sorted by molecule index and the flagged molecules are excluded,
    leaving 130,831 of the published 133,885. Both of those matter for the
    split: a permutation is only reproducible if it is applied to the same rows
    in the same order.

    Returns:
        One row per molecule, with ``lumo`` already in eV.
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    cache = processed_dir / CACHE_NAME

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

    flagged = load_uncharacterized(
        _download(QM9_UNCHARACTERIZED_URL, raw_dir / "uncharacterized.txt")
    )
    frame = frame[~frame["index"].isin(flagged)].reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def random_split(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by seeded random permutation, 80/10/10 by default.

    Reimplemented from the same rule the first package states -- permute the
    row order with ``default_rng(seed)``, then take the first 80% and the next
    10% -- rather than imported from it. The equivalence test is what makes
    that safe.

    Returns:
        ``(train, validation, test)``, disjoint, together covering ``frame``.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must each lie in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 to leave a test set")

    permutation = np.random.default_rng(seed).permutation(len(frame))
    n_train = int(len(frame) * train_frac)
    n_val = int(len(frame) * val_frac)

    shuffled = frame.iloc[permutation].reset_index(drop=True)
    return (
        shuffled.iloc[:n_train].reset_index(drop=True),
        shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True),
        shuffled.iloc[n_train + n_val :].reset_index(drop=True),
    )


def murcko_scaffold(smiles: str, *, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold of a molecule, as SMILES.

    The scaffold is what remains after stripping side chains: the ring systems
    and the linkers between them. Two molecules with the same scaffold are
    variations on one skeleton, which is exactly the relationship a random
    split fails to separate.

    Molecules with no rings -- and QM9 has many, being small -- reduce to the
    empty string. They are all grouped together, which is the honest handling:
    they genuinely share "no ring system" as their skeleton.

    Args:
        smiles: A SMILES string. Invalid input returns the empty string rather
            than raising, so one unparseable molecule cannot stop a split of
            130,000.
    """
    # Imported here rather than at module scope so that importing this module
    # -- which the graph code and the tests do -- never requires RDKit. It is a
    # development dependency used to compute the scaffold assignment once.
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=include_chirality
    )


def scaffold_split(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    smiles_column: str = "smiles_relaxed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split so that no scaffold appears in more than one set.

    Groups are assigned largest first: the common skeletons fill the training
    set and the rare ones fall into validation and test. That ordering is
    deliberate and is what makes this a *hard* split -- the alternative,
    distributing large groups across the three sets, produces friendlier
    numbers that describe nothing anyone would do with the model.

    There is no seed, because there is no randomness. Sorting is by group size
    then by scaffold string, so the partition is a function of the data alone
    and is reproducible without recording anything.

    Returns:
        ``(train, validation, test)``. The sizes only approximate the requested
        fractions: whole groups move together, so a group cannot be cut to make
        a boundary land exactly.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for position, smiles in enumerate(frame[smiles_column]):
        scaffolds[murcko_scaffold(smiles)].append(position)

    groups = sorted(scaffolds.items(), key=lambda item: (-len(item[1]), item[0]))

    n_train = int(len(frame) * train_frac)
    n_val = int(len(frame) * val_frac)

    train_rows: list[int] = []
    val_rows: list[int] = []
    test_rows: list[int] = []
    for _scaffold, rows in groups:
        if len(train_rows) + len(rows) <= n_train:
            train_rows.extend(rows)
        elif len(val_rows) + len(rows) <= n_val:
            val_rows.extend(rows)
        else:
            test_rows.extend(rows)

    return (
        frame.iloc[sorted(train_rows)].reset_index(drop=True),
        frame.iloc[sorted(val_rows)].reset_index(drop=True),
        frame.iloc[sorted(test_rows)].reset_index(drop=True),
    )


def geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(atomic_numbers, coordinates)`` for one row, coordinates as (n, 3)."""
    numbers = np.asarray(row["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(row["coordinates"], dtype=np.float64).reshape(-1, 3)
    return numbers, positions
