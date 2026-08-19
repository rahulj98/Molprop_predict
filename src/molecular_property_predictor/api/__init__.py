"""The HTTP layer: the trained model as a service.

Everything up to Phase 5 runs in a notebook on one machine. This subpackage is
the boundary where the model stops being a Python object and becomes something
any process that speaks HTTP can call -- without knowing it is Python, PyTorch,
or a Coulomb matrix.

The split across three modules is deliberate, and it is the same split the rest
of the project already uses between "what the science is" and "how it is
delivered":

- :mod:`~molecular_property_predictor.api.schemas` -- the contract. What a
  request must look like for the service to accept it, and what comes back.
  Knows nothing about models.
- :mod:`~molecular_property_predictor.api.service` -- featurise and predict.
  Knows nothing about HTTP, so it is testable without a client.
- :mod:`~molecular_property_predictor.api.main` -- the FastAPI application that
  wires the two together and owns the endpoints.

Note what is absent: MLflow. Phase 5 logged its checkpoints as plain artifacts
precisely so that serving one needs nothing but
:func:`~molecular_property_predictor.model.load_artifact`. Tracking is a
development-time concern and does not belong in the container that answers
prediction requests.
"""
