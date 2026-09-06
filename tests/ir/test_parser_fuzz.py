"""Parser fuzzing: a dump either parses into a model that scores, or fails cleanly.

Two phases. The committed corpus under ``tests/fuzz_corpus/`` replays every
dump that once broke a parser, deterministically and in every test leg. The
random phase (marked ``slow``) draws seeds, builds valid dumps of random
shape in each format, corrupts most of them, and holds the parsers to one
contract: return an ``EnsembleIR`` whose raw score is finite on random
inputs, or raise ``UnsupportedModelError`` (``ParserError`` included) —
never a bare ``KeyError``, ``IndexError``, ``TypeError``, ``ValueError``,
``AttributeError``, or a numpy warning promoted to an error.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from treecf._errors import UnsupportedModelError
from treecf.ir.evaluate import raw_score_batch
from treecf.ir.parsers import parse_dump

from ..fuzz.dump_builders import BUILDERS, FORMATS, as_json, mutate

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[1] / "fuzz_corpus"
CORPUS = sorted(CORPUS_DIR.glob("*.json"))


def _check(dump: object, rng: np.random.Generator) -> None:
    """The contract: a scoring model, or a clean refusal."""
    try:
        ir = parse_dump(dump)  # type: ignore[arg-type]
    except UnsupportedModelError:
        return
    X = rng.normal(scale=3.0, size=(40, max(ir.n_features, 1)))[:, : ir.n_features]
    scores = raw_score_batch(ir, X)
    assert np.all(np.isfinite(scores)), "a parsed model must score finitely"


@pytest.mark.parametrize("path", CORPUS, ids=[p.stem for p in CORPUS])
def test_corpus_replays(path: pathlib.Path) -> None:
    entry = json.loads(path.read_text(encoding="utf-8"))
    _check(entry["dump"], np.random.default_rng(0))


def test_corpus_is_not_empty() -> None:
    assert len(CORPUS) >= 8


@pytest.mark.parametrize("fmt", FORMATS)
def test_unmutated_dumps_parse(fmt: str) -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ir = parse_dump(as_json(BUILDERS[fmt](rng)))
        X = rng.normal(size=(10, ir.n_features))
        assert np.all(np.isfinite(raw_score_batch(ir, X)))


@pytest.mark.slow
@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_random_dumps_parse_or_refuse_cleanly(seed: int) -> None:
    rng = np.random.default_rng(seed)
    fmt = FORMATS[int(rng.integers(len(FORMATS)))]
    dump: object = BUILDERS[fmt](rng)
    for _ in range(int(rng.integers(0, 4))):  # zero to three corruptions
        dump = mutate(dump, rng) if isinstance(dump, dict) else dump
    _check(as_json(dump), rng)
