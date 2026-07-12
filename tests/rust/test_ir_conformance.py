"""Rust IR must be BITWISE-equal to the Python batch evaluator (migration P2)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf.ir.evaluate import raw_score_batch
from treecf.ir.flatten import flatten_ir
from treecf.ir.model import EnsembleIR

from ..conftest import make_random_ir
from ..parity.harness import load_scenario, scenario_paths

pytestmark = pytest.mark.rust


def rust_ensemble(ir: EnsembleIR) -> object:
    from treecf.backends.genetic_rust import _core as _load_core
    _treecf_core = _load_core()

    flat = flatten_ir(ir)
    return _treecf_core.RustEnsemble(
        flat["feature"],
        flat["threshold"],
        flat["is_lt"],
        flat["missing_left"],
        flat["left"],
        flat["right"],
        flat["value"],
        flat["tree_roots"],
        flat["base_score"],
        flat["link"],
        flat["n_features"],
    )


def probe_grid(ir: EnsembleIR, rng: np.random.Generator, n_random: int = 2000) -> np.ndarray:
    p = ir.n_features
    X = rng.normal(scale=3.0, size=(n_random, p))
    X[rng.random(X.shape) < 0.15] = np.nan
    rows = [X]
    base = np.zeros(p)
    for tree in ir.trees:
        for node in tree.nodes:
            if node.feature is None:
                continue
            t = float(node.threshold)  # type: ignore[arg-type]
            for v in (t, np.nextafter(t, -np.inf), np.nextafter(t, np.inf)):
                row = base.copy()
                row[node.feature] = v
                rows.append(row.reshape(1, -1))
    return np.ascontiguousarray(np.vstack(rows))


@pytest.mark.parametrize("seed", range(8))
def test_random_irs_bitwise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ir = make_random_ir(
        rng,
        n_features=int(rng.integers(2, 6)),
        n_trees=int(rng.integers(1, 8)),
        depth=int(rng.integers(1, 5)),
    )
    X = probe_grid(ir, rng)
    np.testing.assert_array_equal(
        np.asarray(rust_ensemble(ir).raw_score_batch(X)),  # type: ignore[attr-defined]
        raw_score_batch(ir, X),
    )


@pytest.mark.parametrize("path", scenario_paths(), ids=[p.stem for p in scenario_paths()])
def test_fixture_ensembles_bitwise(path: object) -> None:
    scenario = load_scenario(path)  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    for ir in filter(None, (scenario.ir, scenario.if_ir)):
        X = probe_grid(ir, rng, n_random=1000)
        np.testing.assert_array_equal(
            np.asarray(rust_ensemble(ir).raw_score_batch(X)),  # type: ignore[attr-defined]
            raw_score_batch(ir, X),
        )
