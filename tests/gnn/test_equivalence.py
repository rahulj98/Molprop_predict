"""The tests that make two independent implementations safe to compare.

:mod:`molecular_gnn` deliberately does not import from
:mod:`molecular_property_predictor`. That independence is what makes a
comparison between the two models meaningful -- and it is also exactly what
could make the comparison meaningless, if the two packages quietly came to
disagree about which molecules exist or which of them are in the test set.

This module is the only place in the project that imports both. Tests are the
right home for that coupling: nothing here ships, and a divergence that would
otherwise be invisible becomes a failing test instead of a wrong number in a
results table.

The two heavier tests need the QM9 archive on disk and are marked ``dataset``,
so a fresh clone skips them rather than failing. Run them with ``pytest -m
dataset`` once the dataset has been downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

import molecular_gnn.data as gnn_data
import molecular_property_predictor.data as mlp_data


def test_both_parsers_agree_on_one_molecule(methane_xyz):
    """The cheap half of the check, and it needs no dataset."""
    theirs = mlp_data.parse_xyz(methane_xyz)
    ours = gnn_data.parse_xyz(methane_xyz)

    assert ours["index"] == theirs["index"]
    assert ours["n_atoms"] == theirs["n_atoms"]
    assert ours["atomic_numbers"] == theirs["atomic_numbers"]
    assert ours["coordinates"] == pytest.approx(theirs["coordinates"])
    assert ours["lumo"] == pytest.approx(theirs["lumo"])
    assert ours["smiles_relaxed"] == theirs["smiles_relaxed"]


def test_both_packages_use_the_same_unit_conversion():
    """A different constant would shift every error by a factor nobody notices."""
    assert gnn_data.HARTREE_TO_EV == mlp_data.HARTREE_TO_EV


def test_both_packages_agree_on_the_elements():
    assert gnn_data.ATOMIC_NUMBERS == mlp_data.ATOMIC_NUMBERS


@pytest.mark.dataset
def test_both_loaders_select_the_same_molecules():
    """Same exclusions, same order -- the precondition for the split to match."""
    theirs = mlp_data.load_qm9()
    ours = gnn_data.load_qm9()

    assert len(ours) == len(theirs)
    assert ours["index"].tolist() == theirs["index"].tolist()


@pytest.mark.dataset
def test_both_random_splits_select_identical_molecules():
    """The assertion the whole comparison rests on.

    If this ever fails, the graph network's score is not comparable to the
    0.244 eV the first model reported, and any table putting them side by side
    is wrong.
    """
    theirs = mlp_data.load_qm9()
    ours = gnn_data.load_qm9()

    their_parts = mlp_data.split_dataset(theirs, seed=0)
    our_parts = gnn_data.random_split(ours, seed=0)

    for their_part, our_part, name in zip(
        their_parts, our_parts, ("train", "validation", "test")
    ):
        assert len(our_part) == len(their_part), f"{name} differs in size"
        assert np.array_equal(
            our_part["index"].to_numpy(), their_part["index"].to_numpy()
        ), f"{name} contains different molecules"


@pytest.mark.dataset
def test_the_target_values_match_molecule_for_molecule():
    """Same property, same units, same rows."""
    theirs = mlp_data.load_qm9()
    ours = gnn_data.load_qm9()

    assert ours["lumo"].to_numpy() == pytest.approx(theirs["lumo"].to_numpy())
