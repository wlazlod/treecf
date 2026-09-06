"""A packaged credit-shaped demo: a committed model, a background sample, one declined row.

Every example in the documentation starts from ``credit_demo()``, so it runs
as written on a fresh install without a training library. The model is a
committed LightGBM JSON dump of a small synthetic credit-risk classifier over
five features — ``income``, ``utilization``, ``dpd_12m``, ``tenure_months``,
and a native categorical ``occupation`` — and the background rows follow a
fixed synthetic recipe, so a given seed always returns the same data.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import numpy.typing as npt

from treecf.ir.model import EnsembleIR, apply_categories
from treecf.ir.parsers import parse_model

FloatArray = npt.NDArray[np.float64]

OCCUPATIONS = ("student", "clerk", "manager", "retired")
"""Display names of the demo model's ``occupation`` codes, in code order."""


def credit_demo(seed: int = 7, n: int = 400) -> tuple[EnsembleIR, FloatArray, FloatArray]:
    """The demo model, a background sample, and one declined applicant.

    Parameters
    ----------
    seed
        Seed of the background sample; the model itself is fixed.
    n
        Number of background rows.

    Returns
    -------
    ``(model, X, x)``: the parsed model (an ``EnsembleIR`` with the
    occupation names installed, accepted by ``Explainer`` directly), an
    ``(n, 5)`` background matrix, and its second row, an applicant the
    model scores well above a 5% default probability.
    """
    with resources.as_file(resources.files(__name__).joinpath("credit_model.json")) as path:
        model = apply_categories(parse_model(str(path)), {"occupation": OCCUPATIONS})
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(loc=4200.0, scale=1600.0, size=n),  # income
            np.clip(rng.beta(2.0, 3.5, size=n), 0.0, 1.0),  # utilization
            np.floor(rng.exponential(scale=6.0, size=n)),  # dpd_12m
            np.floor(rng.uniform(3, 240, size=n)),  # tenure_months
            rng.integers(0, 4, size=n).astype(np.float64),  # occupation codes
        ]
    )
    return model, X, X[1].copy()


__all__ = ["OCCUPATIONS", "credit_demo"]
