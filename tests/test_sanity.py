"""Phase 0 sanity check.

Confirms the project skeleton, virtual environment, and pytest are wired up
before any real code exists. The import below is the part that actually
matters: with a src/ layout the package resolves only if the project has been
installed (``pip install -e ".[dev]"``), so a passing test here proves the
packaging is correct, not just that pytest runs.
"""

import molecular_property_predictor


def test_package_is_importable():
    assert molecular_property_predictor.__version__ == "0.1.0"
