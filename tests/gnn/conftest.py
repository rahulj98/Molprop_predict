"""Fixtures for the graph-network tests.

The molecule below is the verbatim contents of ``dsgdb9nsd_000001.xyz``
(methane) from the published archive. It is a second copy of the same fixture
the first package's tests use, and that is deliberate: these tests exist partly
to check that two independent parsers agree, which they cannot do if they share
the input by importing it from each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from molecular_gnn.data import parse_xyz

METHANE_XYZ = (
    "5\n"
    "gdb 1\t157.7118\t157.70997\t157.70699\t0.\t13.21\t-0.3877\t0.1171\t0.5048\t"
    "35.3641\t0.044749\t-40.47893\t-40.476062\t-40.475117\t-40.498597\t6.469\t\n"
    "C\t-0.0126981359\t 1.0858041578\t 0.0080009958\t-0.535689\n"
    "H\t 0.002150416\t-0.0060313176\t 0.0019761204\t 0.133921\n"
    "H\t 1.0117308433\t 1.4637511618\t 0.0002765748\t 0.133922\n"
    "H\t-0.540815069\t 1.4475266138\t-0.8766437152\t 0.133923\n"
    "H\t-0.5238136345\t 1.4379326443\t 0.9063972942\t 0.133923\n"
    "1341.307\t1341.3284\t1341.365\t1562.6731\t1562.7453\t3038.3205\t3151.6034\t"
    "3151.6788\t3151.7078\n"
    "C\tC\t\n"
    "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4\n"
)


@pytest.fixture
def methane_xyz() -> str:
    """The raw text of the methane file."""
    return METHANE_XYZ


@pytest.fixture
def methane() -> dict:
    """Methane parsed by this package's own parser."""
    return parse_xyz(METHANE_XYZ)


@pytest.fixture
def methane_geometry(methane) -> tuple[np.ndarray, np.ndarray]:
    """``(atomic_numbers, positions)`` for methane."""
    numbers = np.asarray(methane["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(methane["coordinates"], dtype=np.float64).reshape(-1, 3)
    return numbers, positions


def pytest_runtest_setup(item):
    """Skip dataset-backed tests when the dataset is not on disk.

    A fresh clone has no QM9 archive, and a test suite that goes red for that
    reason teaches people to ignore red. The skip message names the command
    that fixes it.
    """
    if "dataset" not in item.keywords:
        return

    from molecular_gnn.data import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR

    have_cache = (DEFAULT_PROCESSED_DIR / "qm9_gnn.parquet").exists()
    have_archive = (DEFAULT_RAW_DIR / "dsgdb9nsd.xyz.tar.bz2").exists()
    if not (have_cache or have_archive):
        import pytest as _pytest

        _pytest.skip(
            "QM9 is not on disk; run notebooks/01_explore_qm9.ipynb or call "
            "molecular_gnn.data.load_qm9() once to download it"
        )
