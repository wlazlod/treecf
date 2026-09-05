"""CatBoost category hashing, held to the library's own hashes over drawn strings.

treecf reproduces CatBoost's category hash (CityHash64 of the UTF-8 bytes,
low 32 bits) in pure Python so a JSON dump parses without the library. The
pure invariants run everywhere; the model-backed property (marked ``slow``,
needs catboost and pandas) draws category strings — empty, whitespace,
digits, long, astral-plane code points — trains a tiny one-hot model over
them, and checks that every hash the model stored is one treecf computes,
then that the parsed model scores like the native one.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from treecf.ir.parsers._catboost_cat import cat_feature_hash, city_hash_64, signed32

# imported at collection time: catboost raises the interpreter's recursion
# limit on import, which hypothesis would flag as a change made mid-example
try:
    import catboost
    import pandas as pd
except ImportError:  # the pure invariants below still run without the extras
    catboost = None
    pd = None

_TEXT = st.text(
    alphabet=st.characters(min_codepoint=1, blacklist_categories=("Cs",)),
    min_size=0,
    max_size=40,
)


class TestPureInvariants:
    @given(value=_TEXT)
    @settings(max_examples=200, deadline=None)
    def test_hash_is_a_stable_unsigned_32_bit_value(self, value: str) -> None:
        first = cat_feature_hash(value)
        assert 0 <= first < 2**32
        assert cat_feature_hash(value) == first
        assert signed32(first) & 0xFFFFFFFF == first
        assert -(2**31) <= signed32(first) < 2**31

    @pytest.mark.parametrize(
        "length", [0, 1, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 129]
    )
    def test_every_length_bucket_is_deterministic(self, length: int) -> None:
        data = bytes((i * 37 + 11) % 256 for i in range(length))
        assert city_hash_64(data) == city_hash_64(data)
        assert 0 <= city_hash_64(data) < 2**64

    def test_utf8_bytes_decide_the_hash(self) -> None:
        assert cat_feature_hash("é") == cat_feature_hash("é")
        assert cat_feature_hash("é") != cat_feature_hash("e")
        assert cat_feature_hash("𝔘") != cat_feature_hash("U")


def _frame(rng: np.random.Generator, values: list[str], n: int = 500):
    assert pd is not None
    codes = rng.integers(0, len(values), size=n)
    frame = pd.DataFrame(
        {"num": rng.normal(size=n), "occ": [values[c] for c in codes]}
    )
    prob = np.linspace(0.15, 0.85, len(values))
    y = (rng.random(n) < prob[codes]).astype(int)
    y = y ^ (frame["num"].to_numpy() > 0.5)
    return frame, y.astype(int), codes


@pytest.mark.slow
@settings(max_examples=12, deadline=None)
@example(values=["", " ", "12", "007"])
@example(values=["a", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "x" * 300])
@example(values=["nan", "None", "1.5", "\t"])
@given(values=st.lists(_TEXT, min_size=3, max_size=8, unique=True))
def test_model_hashes_and_predictions_agree(values: list[str]) -> None:
    if catboost is None or pd is None:
        pytest.skip("catboost and pandas are required")
    from treecf.ir.parsers.catboost import parse_catboost

    from .harness import assert_conformance

    rng = np.random.default_rng(len("".join(values)) % 1000)
    frame, y, codes = _frame(rng, values)
    if len(set(y.tolist())) < 2:
        return
    pool = catboost.Pool(frame, label=y, cat_features=["occ"])
    model = catboost.CatBoostClassifier(
        iterations=8, depth=3, one_hot_max_size=255, boosting_type="Plain",
        verbose=False, random_seed=1, allow_writing_files=False,
    )
    model.fit(pool)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/model.json"
        model.save_model(path, format="json")
        with open(path, encoding="utf-8") as fh:
            dump = json.load(fh)
    stored = {
        int(v)
        for f in dump["features_info"].get("categorical_features") or []
        for v in (f.get("values") or [])
    }
    ours = {signed32(cat_feature_hash(v)) for v in values}
    assert stored <= ours, f"hashes the model stored but treecf cannot produce: {stored - ours}"

    ir = parse_catboost(model, categories={"occ": values})
    X = np.column_stack([frame["num"].to_numpy(), codes.astype(float)])

    def predict(A: np.ndarray) -> np.ndarray:
        rows = pd.DataFrame({"num": A[:, 0], "occ": [values[int(c)] for c in A[:, 1]]})
        return model.predict(rows, prediction_type="Probability")[:, 1]

    assert_conformance(ir, X, predict, include_nan=False)
