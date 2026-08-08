"""Smoke-test a built treecf wheel: parse a JSON dump, explain on both backends.

Run inside a fresh venv containing only numpy and the wheel under test — the
dump dict goes through the json_dump parser path, so no model extras are needed.
"""

from __future__ import annotations

import sys

import numpy as np

import treecf


def _split(feat: int, thr: float, left: object, right: object) -> dict[str, object]:
    return {"split_feature": feat, "threshold": thr, "decision_type": "<=",
            "missing_type": "NaN", "default_left": True,
            "left_child": left, "right_child": right}


DUMP: dict[str, object] = {
    "num_tree_per_iteration": 1,
    "objective": "binary",
    "max_feature_idx": 1,
    "feature_names": ["income", "debt"],
    "tree_info": [
        {"tree_structure": _split(0, 3.0, {"leaf_value": 1.0}, {"leaf_value": -1.0})},
        {"tree_structure": _split(1, 1.0, {"leaf_value": -0.5}, {"leaf_value": 0.5})},
    ],
}


def main() -> int:
    assert treecf.__version__, "treecf.__version__ is empty"
    explainer = treecf.Explainer(DUMP, normalizers=np.ones(2))
    x = np.array([5.0, 0.0])  # raw score -1.5 -> p ~= 0.18
    target = treecf.Target.probability(op=">=", value=0.6)
    for backend in ("genetic", "python"):
        result = explainer.explain(x, target, backend=backend, seed=0)
        assert isinstance(result, treecf.Counterfactual), f"{backend}: {result!r}"
        assert result.score_prob is not None and result.score_prob >= 0.6
        assert result.x_cf[0] <= 3.0, f"{backend}: expected income change, got {result.x_cf}"
    print(f"smoke OK: treecf {treecf.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
