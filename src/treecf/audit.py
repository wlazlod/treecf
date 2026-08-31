"""Audit certificates: reproducibility records with a fresh verification.

A certificate binds a returned result to a model fingerprint, a constraint
fingerprint, and the solve parameters, and re-verifies the returned plan at
issue time — it does not cryptographically prove that a search ran or that an
optimality claim is true; re-running with the recorded seed and budgets
against a fingerprint-matching model is how a validator checks that.

Certificates are plain ``dict[str, object]`` values (``"schema_version": 2``;
version 1 files, which carry no categorical fields, still verify)
that serialize with ``json.dumps(cert, allow_nan=False, sort_keys=True)``:
non-finite floats are encoded as the strings ``"NaN"``, ``"Infinity"``, and
``"-Infinity"`` wherever they can occur. See
[Certification — audit certificates](concepts/certification.md#audit-certificates).
"""

from __future__ import annotations

import hashlib
import math
import struct
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError, TreecfWarning
from treecf.constraints.objects import (
    AllowedCategories,
    AllowMissing,
    Constraint,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
)
from treecf.ir.evaluate import bitset_words, raw_score
from treecf.ir.model import EnsembleIR, SplitOp

if TYPE_CHECKING:
    from treecf.api import Counterfactual, Explainer, Infeasible
    from treecf.targets import Target

__all__ = [
    "build_certificate",
    "check_certificate",
    "constraints_fingerprint",
    "ir_fingerprint",
]

FloatArray = npt.NDArray[np.float64]

_NONE_U32 = 0xFFFFFFFF  # sentinel for an index field that does not apply
_NONE_F64 = b"\xff" * 8  # sentinel for a float field that does not apply


def _json_float(value: float) -> float | str:
    """A float as strict JSON allows it: finite as-is, non-finite as a string."""
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return float(value)


def _from_json_float(value: object) -> float:
    """Inverse of ``_json_float``."""
    if value == "NaN":
        return math.nan
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TreecfError(f"not a certificate float: {value!r}")
    return float(value)


def _encode_array(values: FloatArray) -> list[float | str]:
    return [_json_float(v) for v in values.tolist()]


def _decode_array(values: object) -> FloatArray:
    if not isinstance(values, list):
        raise TreecfError("certificate array field is not a list")
    return np.array([_from_json_float(v) for v in values], dtype=np.float64)


def ir_fingerprint(ir: EnsembleIR) -> str:
    """SHA-256 fingerprint of an ensemble over a canonical byte encoding.

    The encoding is positional bytes — the link name, the base score, the
    tree count, then every node of every tree in index order with fixed-width
    little-endian fields and fixed sentinel bytes where a field does not
    apply to the node kind — so the fingerprint is stable across Python
    versions, platforms, and dict ordering, and changes when any structural
    or numeric detail of the ensemble changes (a one-ulp leaf perturbation
    included).

    Parameters
    ----------
    ir
        The parsed ensemble to fingerprint (``Explainer.ir``).

    Returns
    -------
    A 64-character SHA-256 hex digest.
    """
    hasher = hashlib.sha256()
    hasher.update(ir.link.name.encode("utf-8") + b"\x00")
    hasher.update(struct.pack("<d", ir.base_score))
    hasher.update(struct.pack("<I", len(ir.trees)))
    for tree in ir.trees:
        hasher.update(struct.pack("<I", len(tree.nodes)))
        for node in tree.nodes:
            if node.feature is None:  # leaf
                assert node.value is not None
                hasher.update(b"\x00" + struct.pack("<I", _NONE_U32) + _NONE_F64)
                hasher.update(b"\x00\x02")  # op / missing_left sentinels
                hasher.update(struct.pack("<II", _NONE_U32, _NONE_U32))
                hasher.update(struct.pack("<d", node.value))
            elif node.categories is not None:  # set-membership split
                assert node.missing_left is not None
                assert node.left is not None and node.right is not None
                hasher.update(b"\x02" + struct.pack("<I", node.feature))
                hasher.update(struct.pack("<B", 1 if node.missing_left else 0))
                hasher.update(struct.pack("<II", node.left, node.right))
                words = bitset_words(node.categories)
                hasher.update(struct.pack("<I", len(words)))
                for word in words:
                    hasher.update(struct.pack("<Q", word))
            else:
                assert node.threshold is not None and node.op is not None
                assert node.missing_left is not None
                assert node.left is not None and node.right is not None
                hasher.update(b"\x01" + struct.pack("<I", node.feature))
                hasher.update(struct.pack("<d", node.threshold))
                hasher.update(
                    bytes((1 if node.op is SplitOp.LT else 2, 1 if node.missing_left else 0))
                )
                hasher.update(struct.pack("<II", node.left, node.right))
                hasher.update(_NONE_F64)
    if ir.categorical:  # absent on numeric-only models, keeping their digests unchanged
        hasher.update(b"CAT" + struct.pack("<I", len(ir.categorical)))
        for j in sorted(ir.categorical):
            hasher.update(struct.pack("<II", j, ir.categorical[j].cardinality))
    return hasher.hexdigest()


def _constraint_record(c: Constraint, index: dict[str, int]) -> bytes:
    """One constraint as canonical bytes: type tag, resolved feature indices,
    parameters as little-endian f64. Never ``repr`` — bytes stay stable."""
    if isinstance(c, Freeze):
        return b"Freeze\x00" + struct.pack("<I", index[c.feature])
    if isinstance(c, Monotone):
        return b"Monotone\x00" + struct.pack("<I", index[c.feature]) + c.direction.encode("utf-8")
    if isinstance(c, Range):
        return b"Range\x00" + struct.pack("<Idd", index[c.feature], c.lo, c.hi)
    if isinstance(c, Linear):
        terms = sorted((index[name], float(coef)) for name, coef in c.coefficients.items())
        body = b"".join(struct.pack("<Id", j, coef) for j, coef in terms)
        return (
            b"Linear\x00" + body + c.op.encode("utf-8") + b"\x00"
            + struct.pack("<d", c.rhs) + c.missing_policy.encode("utf-8")
        )
    if isinstance(c, Equals):
        return b"Equals\x00" + struct.pack("<Id", index[c.feature], c.value)
    if isinstance(c, Implies):
        return (
            b"Implies\x00"
            + struct.pack("<Id", index[c.condition.feature], c.condition.value)
            + struct.pack("<Id", index[c.consequence.feature], c.consequence.value)
        )
    if isinstance(c, OneHot):
        members = sorted(index[name] for name in c.features)
        return b"OneHot\x00" + b"".join(struct.pack("<I", j) for j in members)
    if isinstance(c, AllowedCategories):
        codes = sorted(entry for entry in c.allowed if isinstance(entry, int))
        names = sorted(entry for entry in c.allowed if isinstance(entry, str))
        body = b"".join(struct.pack("<q", code) for code in codes)
        body += b"".join(name.encode("utf-8") + b"\x00" for name in names)
        return b"AllowedCategories\x00" + struct.pack("<I", index[c.feature]) + body
    assert isinstance(c, AllowMissing)
    delta_from = c.delta_miss if c.delta_from_miss is None else c.delta_from_miss
    return b"AllowMissing\x00" + struct.pack("<Idd", index[c.feature], c.delta_miss, delta_from)


def _constraints_encoding(explainer: Explainer) -> tuple[bytes, str | None]:
    """Canonical bytes for the effective objective and constraint set, plus a
    non-reproducibility reason when a component has no canonical encoding."""
    index = {name: j for j, name in enumerate(explainer.ir.feature_names)}
    records = sorted(_constraint_record(c, index) for c in explainer.compiled.constraints)
    parts = [struct.pack("<I", len(records)), *records]
    parts.append(np.asarray(explainer.sigma, dtype="<f8").tobytes())
    parts.append(np.asarray(explainer.weights, dtype="<f8").tobytes())
    reason: str | None = None
    for name in sorted(explainer.value_policy):
        policy = explainer.value_policy[name]
        parts.append(struct.pack("<I", index[name]))
        if isinstance(policy, str):
            parts.append(policy.encode("utf-8") + b"\x00")
        elif callable(policy):
            # a callable has no canonical encoding — never hash its repr
            parts.append(b"unhashable_custom\x00")
            reason = (
                f"value_policy[{name!r}] is a callable policy with no canonical "
                "encoding, so the constraints fingerprint cannot pin it"
            )
        else:
            parts.append(b"Grid\x00" + struct.pack("<dd", policy.step, policy.anchor))
    return b"".join(parts), reason


def constraints_fingerprint(explainer: Explainer) -> str:
    """SHA-256 fingerprint of an explainer's effective objective and constraints.

    Covers the compiled constraint set (type tags, resolved feature indices,
    parameters), the distance normalizers ``sigma``, the per-feature
    ``weights``, and every value-policy entry, all as canonical little-endian
    bytes. A callable value policy has no canonical encoding: it is hashed as
    a fixed ``unhashable_custom`` tag, and any certificate built from the
    explainer records ``"reproducible": false`` with a reason.

    Parameters
    ----------
    explainer
        The explainer whose constraint set to fingerprint.

    Returns
    -------
    A 64-character SHA-256 hex digest.
    """
    encoding, _ = _constraints_encoding(explainer)
    return hashlib.sha256(encoding).hexdigest()


def _backend_of(stats: dict[str, object]) -> str:
    """Recover which backend family produced a result from its own stats;
    ``"unknown"`` when the stats identify neither family."""
    if "completed" in stats:  # the exact backend always reports this key
        return "exact"
    if not stats or "generations" in stats:
        return "genetic/python"
    return "unknown"


def _json_stats(stats: dict[str, object]) -> dict[str, object]:
    """A JSON-safe copy of solver stats (numpy scalars unwrapped, floats
    encoded per the non-finite rule, anything exotic stringified)."""
    out: dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool | int | str):
            out[key] = value
        elif isinstance(value, float):
            out[key] = _json_float(value)
        else:
            out[key] = repr(value)
    return out


def _region_points(
    x_cf: FloatArray,
    intervals: dict[str, tuple[float, float]],
    feature_names: tuple[str, ...],
) -> list[tuple[str, FloatArray]]:
    """The finite point set a certificate verifies for a region (all corners
    would be exponential): each widened feature's own two endpoints holding
    every other coordinate at ``x_cf``, plus the all-lo and all-hi corners."""
    index = {name: j for j, name in enumerate(feature_names)}
    points: list[tuple[str, FloatArray]] = []
    all_lo = x_cf.copy()
    all_hi = x_cf.copy()
    for name, (lo, hi) in intervals.items():
        j = index[name]
        for side, endpoint in (("lo", lo), ("hi", hi)):
            point = x_cf.copy()
            point[j] = endpoint
            points.append((f"{name}={side}", point))
        all_lo[j] = lo
        all_hi[j] = hi
    points.append(("all-lo", all_lo))
    points.append(("all-hi", all_hi))
    return points


def _verify_plan(
    explainer: Explainer,
    x: FloatArray,
    x_cf: FloatArray,
    interval: tuple[float, float],
    region_intervals: dict[str, tuple[float, float]] | None,
    region_categories: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Fresh verification of a plan: recomputed score, target membership, the
    compiled constraint check, plausibility when configured, and the region's
    sampled points. Returns the block and the names of any failed checks."""
    score = raw_score(explainer.ir, x_cf)
    in_target = bool(interval[0] <= score <= interval[1])
    constraints_ok = bool(explainer.compiled.check_matrix(x_cf[None, :], x)[0])
    verification: dict[str, object] = {
        "score_raw": _json_float(score),
        "in_target_interval": in_target,
        "constraints_ok": constraints_ok,
    }
    failed = [
        name
        for name, ok in (("in_target_interval", in_target), ("constraints_ok", constraints_ok))
        if not ok
    ]
    if explainer.plausibility is not None:
        anomaly = explainer.plausibility.anomaly_score(x_cf)
        plausibility_ok = bool(anomaly <= explainer.plausibility.max_anomaly_score + 1e-12)
        verification["plausibility_ok"] = plausibility_ok
        if not plausibility_ok:
            failed.append("plausibility_ok")
    if region_intervals is not None or region_categories is not None:
        checked: list[dict[str, object]] = []
        points = _region_points(
            x_cf, region_intervals or {}, explainer.ir.feature_names
        )
        points.extend(
            _region_category_points(
                x_cf, region_categories or {}, explainer.ir.feature_names
            )
        )
        for label, point in points:
            ok = explainer._verify(x, point, interval) is None
            checked.append({"point": label, "ok": ok})
            if not ok:
                failed.append(f"region point {label}")
        verification["region_points"] = checked
    return verification, failed


def _region_category_points(
    x_cf: FloatArray,
    categories: dict[str, tuple[int, ...]],
    feature_names: tuple[str, ...],
) -> list[tuple[str, FloatArray]]:
    """One sampled point per certified category code, holding every other
    coordinate at ``x_cf``."""
    index = {name: j for j, name in enumerate(feature_names)}
    points: list[tuple[str, FloatArray]] = []
    for name, codes in categories.items():
        j = index[name]
        for code in codes:
            point = x_cf.copy()
            point[j] = float(code)
            points.append((f"{name}={code}", point))
    return points


def _verify_infeasible(
    explainer: Explainer, x: FloatArray, interval: tuple[float, float]
) -> dict[str, object]:
    """For an ``Infeasible`` there is no plan to verify — record only whether
    the factual itself already sits outside the target interval."""
    score = raw_score(explainer.ir, x)
    return {
        "factual_score_raw": _json_float(score),
        "factual_in_target_interval": bool(interval[0] <= score <= interval[1]),
    }


def _duck_fingerprint(calibrator: object) -> str | None:
    """The calibrator's duck-typed ``fingerprint()``, or ``None``.

    Absent or raising members degrade to ``None`` — provenance is optional
    by design; treecf never requires a calibration library at runtime.
    """
    fn = getattr(calibrator, "fingerprint", None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:
        return None
    return str(value) if value is not None else None


def _target_block(
    target: Target, band: str | None, explainer: Explainer
) -> tuple[dict[str, object], tuple[float, float]]:
    """The certificate's target block and the resolved raw interval."""
    if target.bands_spec is not None:
        if band is None:
            raise TreecfError(
                "certificate for a Target.bands result requires band= naming which "
                "band the result belongs to"
            )
        by_name = {name: (lo, hi) for name, lo, hi in target.bands_spec}
        if band not in by_name:
            raise TreecfError(f"band {band!r} is not part of this Target.bands ladder")
        lo, hi = by_name[band]
        interval = target.band_intervals(explainer.ir.link)[band]
    else:
        if band is not None:
            raise TreecfError("band= is only valid for a Target.bands ladder")
        lo, hi = target.lo, target.hi
        interval = target.raw_interval(explainer.ir.link)
    block: dict[str, object] = {
        "space": target.space,
        "lo": _json_float(lo),
        "hi": _json_float(hi),
        "raw_interval": [_json_float(interval[0]), _json_float(interval[1])],
    }
    if band is not None:
        block["band"] = band
    if target.space == "calibrated":
        block["calibrator"] = {
            "embedded": False,
            "fingerprint": _duck_fingerprint(target.calibrator),
            "type": type(target.calibrator).__name__,
            "buffer_logit": _json_float(target.buffer_logit),
        }
    return block, interval


def _factual_block(explainer: Explainer, x: FloatArray, target: Target) -> dict[str, object]:
    block: dict[str, object] = {"x": _encode_array(x)}
    if target.space == "calibrated":
        from treecf.api import _calibrated_readout

        readout = _calibrated_readout(target, raw_score(explainer.ir, x))
        block["score_calibrated"] = _json_float(readout) if readout is not None else None
    return block


def build_certificate(
    explainer: Explainer,
    x: FloatArray,
    result: Counterfactual | Infeasible,
    target: Target,
    *,
    band: str | None = None,
    seed: int | None = None,
    node_budget: int | None = None,
    gap: float | None = None,
    time_budget_s: float | None = None,
    warm_start: bool | None = None,
) -> dict[str, object]:
    """Body of ``Explainer.certificate``; see its docstring."""
    from treecf import __version__
    from treecf.api import Counterfactual

    x = np.asarray(x, dtype=np.float64)
    target_block, interval = _target_block(target, band, explainer)
    _, reproducible_reason = _constraints_encoding(explainer)

    model: dict[str, object] = {
        "ir_fingerprint": ir_fingerprint(explainer.ir),
        "feature_names": list(explainer.ir.feature_names),
        "link": explainer.ir.link.name,
    }
    if explainer.plausibility is not None:
        model["plausibility"] = {
            "ir_fingerprint": ir_fingerprint(explainer.plausibility.if_ir),
            "min_total_path": _json_float(explainer.plausibility.min_total_path),
        }

    declared: dict[str, object] = {}
    if seed is not None:
        declared["seed"] = seed
    if node_budget is not None:
        declared["node_budget"] = node_budget
    if gap is not None:
        declared["gap"] = _json_float(gap)
    if time_budget_s is not None:
        declared["time_budget_s"] = _json_float(time_budget_s)
    if warm_start is not None:
        declared["warm_start"] = warm_start
    solve: dict[str, object] = {
        "backend": _backend_of(result.solver_stats),
        "proof": result.proof,
        "solver_stats": _json_stats(result.solver_stats),
    }
    if declared:
        solve["declared"] = declared

    cert: dict[str, object] = {
        "schema_version": 2,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "treecf_version": __version__,
        "reproducible": reproducible_reason is None,
        "model": model,
        "constraints": {
            "fingerprint": constraints_fingerprint(explainer),
            "listing": [repr(c) for c in explainer.compiled.constraints],
        },
        "target": target_block,
        "solve": solve,
        "factual": _factual_block(explainer, x, target),
    }
    if reproducible_reason is not None:
        cert["reproducible_reason"] = reproducible_reason

    if isinstance(result, Counterfactual):
        plan: dict[str, object] = {
            "x_cf": _encode_array(result.x_cf),
            "changes": {
                name: [_json_float(src), _json_float(dst)]
                for name, (src, dst) in result.changes.items()
            },
            "distance": _json_float(result.distance),
            "snapped": dict(result.snapped),
        }
        region_intervals = None
        region_categories = None
        if result.region is not None:
            region_intervals = dict(result.region.feature_intervals)
            plan["region_feature_intervals"] = {
                name: [_json_float(lo), _json_float(hi)]
                for name, (lo, hi) in region_intervals.items()
            }
            if result.region.feature_categories:
                region_categories = dict(result.region.feature_categories)
                plan["region_feature_categories"] = {
                    name: list(codes) for name, codes in region_categories.items()
                }
        cert["plan"] = plan
        verification, failed = _verify_plan(
            explainer, x, result.x_cf, interval, region_intervals, region_categories
        )
        if failed:
            warnings.warn(
                "certificate verification failed: " + ", ".join(failed)
                + "; the certificate is issued with the failing checks recorded",
                TreecfWarning,
                stacklevel=3,  # build_certificate <- Explainer.certificate <- user code
            )
    else:
        cert["infeasible"] = {"reason": result.reason, "proof": result.proof}
        verification = _verify_infeasible(explainer, x, interval)
    cert["verification"] = verification
    return cert


def check_certificate(
    explainer: Explainer, cert: dict[str, object], *, calibrator: object | None = None
) -> dict[str, object]:
    """Body of ``Explainer.check_certificate``; see its docstring."""
    mismatches: list[str] = []

    model = cert.get("model")
    model_match = isinstance(model, dict) and model.get("ir_fingerprint") == ir_fingerprint(
        explainer.ir
    )
    if not model_match:
        mismatches.append("model fingerprint does not match this explainer's ensemble")
    stored_plaus = model.get("plausibility") if isinstance(model, dict) else None
    if explainer.plausibility is None:
        if stored_plaus is not None:
            model_match = False
            mismatches.append(
                "certificate declares a plausibility ensemble; this explainer has none"
            )
    elif not (
        isinstance(stored_plaus, dict)
        and stored_plaus.get("ir_fingerprint") == ir_fingerprint(explainer.plausibility.if_ir)
    ):
        model_match = False
        mismatches.append("plausibility ensemble fingerprint does not match this explainer's")

    constraints = cert.get("constraints")
    constraints_match = (
        isinstance(constraints, dict)
        and constraints.get("fingerprint") == constraints_fingerprint(explainer)
    )
    if not constraints_match:
        mismatches.append("constraints fingerprint does not match this explainer's constraint set")

    verification_ok = True
    schema_version = cert.get("schema_version", 1)
    if schema_version not in (1, 2):
        return {
            "model_match": model_match,
            "constraints_match": constraints_match,
            "verification_ok": False,
            "mismatches": [
                *mismatches,
                f"unknown certificate schema_version {schema_version!r} "
                "(this release verifies versions 1 and 2)",
            ],
        }
    try:
        target = cert.get("target")
        if not isinstance(target, dict):
            raise TreecfError("certificate has no target block")
        raw = target.get("raw_interval")
        if not isinstance(raw, list) or len(raw) != 2:
            raise TreecfError("certificate target block has no raw_interval")
        interval = (_from_json_float(raw[0]), _from_json_float(raw[1]))
        factual = cert.get("factual")
        if not isinstance(factual, dict):
            raise TreecfError("certificate has no factual block")
        x = _decode_array(factual.get("x"))
        plan = cert.get("plan")
        if isinstance(plan, dict):
            x_cf = _decode_array(plan.get("x_cf"))
            intervals: dict[str, tuple[float, float]] | None = None
            stored_intervals = plan.get("region_feature_intervals")
            if isinstance(stored_intervals, dict):
                intervals = {
                    str(name): (_from_json_float(pair[0]), _from_json_float(pair[1]))
                    for name, pair in stored_intervals.items()
                }
            categories: dict[str, tuple[int, ...]] | None = None
            stored_categories = plan.get("region_feature_categories")
            if isinstance(stored_categories, dict):  # version 1 files carry none
                categories = {
                    str(name): tuple(int(c) for c in codes)
                    for name, codes in stored_categories.items()
                }
            _, failed = _verify_plan(explainer, x, x_cf, interval, intervals, categories)
            if failed:
                verification_ok = False
                mismatches.extend(f"verification failed: {name}" for name in failed)
        else:
            fresh = _verify_infeasible(explainer, x, interval)
            stored = cert.get("verification")
            stored_flag = stored.get("factual_in_target_interval") if isinstance(
                stored, dict
            ) else None
            if stored_flag != fresh["factual_in_target_interval"]:
                verification_ok = False
                mismatches.append(
                    "the recomputed factual-in-target check no longer matches the certificate"
                )
    except Exception as exc:  # a validator's tool reports; it never raises on bad input
        verification_ok = False
        mismatches.append(f"verification could not be re-run: {exc}")

    report: dict[str, object] = {
        "model_match": model_match,
        "constraints_match": constraints_match,
        "verification_ok": verification_ok,
        "mismatches": mismatches,
    }
    if calibrator is not None:
        report["calibrator_match"] = _check_calibrator(cert, calibrator, mismatches)
    return report


def _check_calibrator(
    cert: dict[str, object], calibrator: object, mismatches: list[str]
) -> bool:
    """The two extra checks ``check_certificate(calibrator=...)`` adds.

    (a) the duck-typed fingerprint against the stored one — unavailable on
    either side is noted in ``mismatches`` without failing the match, since
    absence is not evidence of a different calibrator; (b) the certificate's
    calibrated ``lo``/``hi`` re-inverted through the calibrator against the
    stored ``raw_interval`` (rtol 1e-9, infinities by identity) — the check
    that actually proves this calibrator produces this plan geometry.
    """
    ok = True
    target = cert.get("target")
    if not isinstance(target, dict) or target.get("space") != "calibrated":
        mismatches.append("calibrator= given but the certificate's target is not calibrated")
        return False
    stored_block = target.get("calibrator")
    stored_fp = stored_block.get("fingerprint") if isinstance(stored_block, dict) else None
    fresh_fp = _duck_fingerprint(calibrator)
    if stored_fp is None or fresh_fp is None:
        mismatches.append(
            "calibrator fingerprint unavailable on "
            + ("both sides" if stored_fp is None and fresh_fp is None else "one side")
            + " — identity not confirmed by fingerprint"
        )
    elif stored_fp != fresh_fp:
        ok = False
        mismatches.append("calibrator fingerprint does not match the certificate's")
    stored_buffer = (
        _from_json_float(stored_block.get("buffer_logit", 0.0))
        if isinstance(stored_block, dict)
        else 0.0
    )
    raw = target.get("raw_interval")
    try:
        lo = _from_json_float(target.get("lo"))
        hi = _from_json_float(target.get("hi"))
        fresh_lo, fresh_hi = calibrator.interval_inverse(  # type: ignore[attr-defined]
            lo, hi, space="logit", buffer_logit=stored_buffer
        )
        stored_lo = _from_json_float(raw[0])  # type: ignore[index]
        stored_hi = _from_json_float(raw[1])  # type: ignore[index]
        for fresh, stored in ((fresh_lo, stored_lo), (fresh_hi, stored_hi)):
            if math.isinf(fresh) or math.isinf(stored):
                same = fresh == stored
            else:
                same = math.isclose(fresh, stored, rel_tol=1e-9, abs_tol=1e-12)
            if not same:
                ok = False
                mismatches.append(
                    "re-inverted interval does not match the stored raw_interval: "
                    f"fresh ({fresh_lo}, {fresh_hi}) vs stored ({stored_lo}, {stored_hi})"
                )
                break
    except Exception as exc:
        ok = False
        mismatches.append(f"re-inversion through the supplied calibrator failed: {exc}")
    return ok
