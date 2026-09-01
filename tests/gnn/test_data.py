"""Tests for this package's own QM9 loader and its two splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molecular_gnn.data import (
    HARTREE_TO_EV,
    _parse_float,
    murcko_scaffold,
    random_split,
    scaffold_split,
)

# --- Parsing ------------------------------------------------------------------


def test_parses_the_molecule_index_and_size(methane):
    assert methane["index"] == 1
    assert methane["n_atoms"] == 5


def test_parses_atoms_and_positions(methane):
    assert methane["atomic_numbers"] == [6, 1, 1, 1, 1]
    assert len(methane["coordinates"]) == 15  # flattened (5, 3)


def test_converts_the_target_from_hartree_to_ev(methane):
    """The published LUMO for methane is 0.1171 Hartree."""
    assert methane["lumo"] == pytest.approx(0.1171 * HARTREE_TO_EV)
    assert methane["lumo"] == pytest.approx(3.186, abs=1e-3)


def test_reads_the_lumo_column_and_not_a_neighbouring_one(methane):
    """Guards the header offset, which is the easiest thing here to get wrong.

    HOMO (-0.3877 Ha) sits immediately before LUMO in the header and the gap
    immediately after. An off-by-one would still produce a plausible energy.
    """
    assert methane["lumo"] > 0  # HOMO is negative; LUMO here is not
    assert methane["lumo"] != pytest.approx(0.5048 * HARTREE_TO_EV)  # the gap


def test_keeps_the_smiles_from_the_relaxed_geometry(methane):
    assert methane["smiles_relaxed"] == "C"


def test_parses_mathematica_exponent_notation():
    """QM9 contains values like 1.234*^-6, which float() rejects."""
    assert _parse_float("1.234*^-6") == pytest.approx(1.234e-6)
    assert _parse_float("-3.5") == pytest.approx(-3.5)


# --- Random split -------------------------------------------------------------


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A frame with the columns the splits touch, and nothing else."""
    return pd.DataFrame(
        {
            "index": np.arange(1, 1001),
            "lumo": np.random.default_rng(1).normal(size=1000),
            # A handful of scaffolds with deliberately uneven populations.
            "smiles_relaxed": (
                ["c1ccccc1C"] * 500
                + ["C1CCCCC1CC"] * 300
                + ["c1ccncc1CC"] * 150
                + ["CCCC"] * 50
            ),
        }
    )


def test_random_split_uses_every_molecule_exactly_once(synthetic_frame):
    train, validation, test = random_split(synthetic_frame)

    recovered = pd.concat([train, validation, test])["index"]
    assert len(recovered) == len(synthetic_frame)
    assert set(recovered) == set(synthetic_frame["index"])


def test_random_split_respects_the_requested_fractions(synthetic_frame):
    train, validation, test = random_split(synthetic_frame)

    assert len(train) == 800
    assert len(validation) == 100
    assert len(test) == 100


def test_random_split_is_reproducible(synthetic_frame):
    first = random_split(synthetic_frame, seed=7)[2]["index"].tolist()
    second = random_split(synthetic_frame, seed=7)[2]["index"].tolist()

    assert first == second


def test_random_split_changes_with_the_seed(synthetic_frame):
    zero = random_split(synthetic_frame, seed=0)[2]["index"].tolist()
    one = random_split(synthetic_frame, seed=1)[2]["index"].tolist()

    assert zero != one


def test_random_split_rejects_fractions_leaving_no_test_set(synthetic_frame):
    with pytest.raises(ValueError, match="must be < 1"):
        random_split(synthetic_frame, train_frac=0.9, val_frac=0.1)


# --- Scaffolds ----------------------------------------------------------------


def test_murcko_scaffold_strips_substituents():
    """Aspirin reduces to its benzene ring."""
    assert murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") == "c1ccccc1"


def test_acyclic_molecules_have_an_empty_scaffold():
    """QM9 contains many; they all share 'no ring system' as their skeleton."""
    assert murcko_scaffold("C") == ""
    assert murcko_scaffold("CCCO") == ""


def test_unparseable_smiles_does_not_raise():
    """One bad molecule must not stop a split of 130,000."""
    assert murcko_scaffold("not a molecule") == ""


def test_scaffold_split_keeps_each_scaffold_in_one_set_only(synthetic_frame):
    """The property the whole split exists for."""
    parts = scaffold_split(synthetic_frame)

    scaffold_sets = [
        {murcko_scaffold(s) for s in part["smiles_relaxed"]} for part in parts
    ]
    for left in range(len(scaffold_sets)):
        for right in range(left + 1, len(scaffold_sets)):
            assert not scaffold_sets[left] & scaffold_sets[right]


def test_scaffold_split_uses_every_molecule_exactly_once(synthetic_frame):
    train, validation, test = scaffold_split(synthetic_frame)

    recovered = pd.concat([train, validation, test])["index"]
    assert len(recovered) == len(synthetic_frame)
    assert set(recovered) == set(synthetic_frame["index"])


def test_scaffold_split_puts_the_largest_group_in_training(synthetic_frame):
    """Common skeletons train the model; rare ones test it."""
    train, _, _ = scaffold_split(synthetic_frame)

    assert len(train) >= 500


def test_scaffold_split_is_deterministic_without_a_seed(synthetic_frame):
    first = scaffold_split(synthetic_frame)[2]["index"].tolist()
    second = scaffold_split(synthetic_frame.copy())[2]["index"].tolist()

    assert first == second


def test_scaffold_split_is_harder_than_random(synthetic_frame):
    """The test sets differ -- which is the entire reason for the second split.

    A weak assertion on purpose: what matters is that scaffold grouping selects
    different molecules than a permutation would, not by how much.
    """
    random_test = set(random_split(synthetic_frame)[2]["index"])
    scaffold_test = set(scaffold_split(synthetic_frame)[2]["index"])

    assert random_test != scaffold_test
