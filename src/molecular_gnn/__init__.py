"""A graph neural network for QM9, built as a continuation of this repository.

**Why a second package rather than more modules in the first one.** The point
of this work is to find out whether a *learned* molecular representation beats
the fixed one in :mod:`molecular_property_predictor` -- and a comparison is only
worth anything if the two sides are genuinely independent. Sharing a featuriser,
a loader, or a split function would make it impossible to say afterwards whether
a difference came from the model or from a helper both sides happened to lean
on. So nothing here imports from that package.

**What is deliberately duplicated, and why that is not an accident.** This
package parses QM9 itself and computes its own split. That is code the other
package already has, and copying code is normally the wrong instinct. It is
justified here by the one thing that matters more than dryness: independence of
measurement. The duplication is made safe by tests -- ``tests/gnn/`` asserts
that both parsers produce identical records and that both split functions select
identical molecules, so the two implementations cannot drift apart silently
while still claiming to be comparable.

**What is shared:** the dataset on disk, the target property (LUMO, in eV), and
the evaluation discipline -- the test split stays sealed until a model is
frozen, exactly as before.
"""

__version__ = "0.1.0"
