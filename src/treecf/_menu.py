"""Recourse menus over lever sets, and lever-diverse plan sets.

A *recourse menu* answers "which combinations of levers reach the target,
and at what cost?" for one factual. Every candidate lever set is solved as a
coalition — every other feature frozen — so an entry is exactly the plan the
search finds when only those levers may move, with the same proof the exact
backend attaches to any solve. Two plans count as different ways to reach
the target when they change different sets of features; that is the only
notion of diversity here.

Feasibility over lever sets is monotone: freeing more features never removes
a plan. The enumeration leans on that in two places — a set whose plan
changed only a strict subset is filed under the subset (and recorded as
implied), and the default mode skips every superset of a set already known
to be feasible. A certified-infeasible set that contains a feasible set would
contradict monotonicity, so it is reported as an error rather than returned.
"""

from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError, TreecfWarning

if TYPE_CHECKING:
    from treecf.api import Counterfactual, Explainer, Infeasible
    from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]

MENU_SCHEMA_VERSION = 1
_CERTIFIED_PLAN_PROOFS = frozenset({"optimal", "optimal_within_gap"})
_MODES = ("minimal", "all")
_CRITERIA = ("levers", "coalitions")
_MENU_OPTIONS: dict[str, Any] = {
    "backend": "exact",
    "search": "classic",
    "seed": None,
    "time_budget_s": None,
    "total_budget_s": None,
    "warm_start": None,
    "node_budget": None,
    "gap": None,
}


def _menu_options(menu_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The menu's keyword options with defaults filled in; unknown names raise."""
    unknown = sorted(set(menu_kwargs) - set(_MENU_OPTIONS))
    if unknown:
        raise TypeError(f"unknown option(s) for the menu: {', '.join(unknown)}")
    return {**_MENU_OPTIONS, **menu_kwargs}


def _key(names: Iterable[str]) -> str:
    return "+".join(sorted(names))


def _members(key: str) -> frozenset[str]:
    return frozenset(key.split("+"))


def _is_certified(entry: Counterfactual | Infeasible) -> bool:
    from treecf.api import Counterfactual

    if isinstance(entry, Counterfactual):
        return entry.proof in _CERTIFIED_PLAN_PROOFS
    return entry.proof == "certified"


@dataclass(frozen=True)
class RecourseMenu(Mapping[str, "Counterfactual | Infeasible"]):
    """Every lever set solved for one factual, keyed by the features its plan changed.

    A mapping ``{key: Counterfactual | Infeasible}`` whose keys are the sorted
    feature names joined by ``"+"`` (``"debt+income"``), so any function that
    takes the ``explain_coalitions`` mapping shape — ``plot_recourse_map``
    among them — takes a menu unchanged. Iteration order is the display
    order: the minimal feasible sets by ascending cost, then the remaining
    feasible sets by cost, then the infeasible sets by size.

    Attributes
    ----------
    entries
        The mapping itself, in display order.
    minimal
        The frontier: feasible keys no other feasible key is a subset of.
        No listed set contains another.
    implied
        ``{lever set: key}`` for every solved lever set whose plan changed
        only a strict subset of it; the set is feasible by monotonicity and
        its plan lives under the subset's key.
    certified_infeasible
        Lever sets a completed exact search proved cannot reach the target
        on their own: "no acceptance is reachable by changing only these
        levers".
    unresolved
        Lever sets the total time budget did not reach; they have no entry.
    complete
        ``True`` iff nothing is unresolved and every entry is certified
        (``optimal``, ``optimal_within_gap``, or a certified
        ``Infeasible``): the menu then settles every lever set up to
        ``max_levers``. Never ``True`` for the genetic backend.
    mode
        ``"minimal"`` or ``"all"``.
    max_levers
        The largest set size enumerated.
    levers
        The candidate lever names, sorted: the features the search would
        branch on for this factual (not frozen, more than one candidate
        value, influential).
    """

    entries: dict[str, Counterfactual | Infeasible]
    minimal: tuple[str, ...]
    implied: dict[str, str]
    certified_infeasible: tuple[str, ...]
    unresolved: tuple[str, ...]
    complete: bool
    mode: str
    max_levers: int
    levers: tuple[str, ...]

    def __getitem__(self, key: str) -> Counterfactual | Infeasible:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> dict[str, str]:
        """One sentence per entry, in display order.

        A feasible entry reads ``"change a, b (cost 1.23, optimal)"``; a
        certified-infeasible one ``"no acceptance is reachable by changing
        only a, b"``; an uncertified one names the levers and says the search
        found no plan without a certificate.

        Returns
        -------
        ``{key: sentence}``.
        """
        from treecf.api import Counterfactual

        out: dict[str, str] = {}
        for key, entry in self.entries.items():
            levers = ", ".join(sorted(_members(key)))
            if isinstance(entry, Counterfactual):
                out[key] = f"change {levers} (cost {entry.distance:.3g}, {entry.proof})"
            elif entry.proof == "certified":
                out[key] = f"no acceptance is reachable by changing only {levers}"
            else:
                out[key] = f"no plan found changing only {levers} (not certified)"
        return out

    def to_frame(self) -> list[dict[str, object]]:
        """One row per entry, in display order.

        Returns
        -------
        ``[{"key", "size", "feasible", "distance", "proof", "changed"}, ...]``;
        ``distance`` is ``None`` and ``changed`` empty for an infeasible entry.
        """
        from treecf.api import Counterfactual

        rows: list[dict[str, object]] = []
        for key, entry in self.entries.items():
            feasible = isinstance(entry, Counterfactual)
            rows.append(
                {
                    "key": key,
                    "size": len(_members(key)),
                    "feasible": feasible,
                    "distance": entry.distance if isinstance(entry, Counterfactual) else None,
                    "proof": entry.proof,
                    "changed": sorted(entry.changes) if isinstance(entry, Counterfactual) else [],
                }
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        """The menu as strict JSON (non-finite floats encoded as the
        certificate encodes them; readers tolerate unknown keys).

        Returns
        -------
        A plain dict with ``menu_schema_version`` 1.
        """
        from treecf.api import Counterfactual
        from treecf.audit import _json_float

        entries: dict[str, object] = {}
        for key, entry in self.entries.items():
            if isinstance(entry, Counterfactual):
                entries[key] = {
                    "feasible": True,
                    "proof": entry.proof,
                    "distance": _json_float(entry.distance),
                    "changed": sorted(entry.changes),
                    "changes": {
                        name: [_json_float(before), _json_float(after)]
                        for name, (before, after) in sorted(entry.changes.items())
                    },
                    "x_cf": [_json_float(float(v)) for v in entry.x_cf],
                }
            else:
                entries[key] = {"feasible": False, "proof": entry.proof, "reason": entry.reason}
        return {
            "menu_schema_version": MENU_SCHEMA_VERSION,
            "mode": self.mode,
            "max_levers": self.max_levers,
            "levers": list(self.levers),
            "complete": self.complete,
            "minimal": list(self.minimal),
            "implied": dict(self.implied),
            "certified_infeasible": list(self.certified_infeasible),
            "unresolved": list(self.unresolved),
            "entries": entries,
        }


@dataclass(frozen=True)
class DiverseSet(Mapping[str, "Counterfactual"]):
    """Up to ``k`` plans that reach the target through different lever sets.

    A mapping ``{key: Counterfactual}`` in ascending cost order. Under the
    ``"levers"`` criterion the keys are lever-set keys of the underlying
    menu; under ``"coalitions"`` they are coalition names, joined by ``"+"``
    for a union.

    Attributes
    ----------
    plans
        The mapping itself, cheapest first.
    criterion
        ``"levers"`` or ``"coalitions"``.
    complete
        ``True`` iff fewer than ``k`` plans came back *and* the search that
        produced them was exhausted with every entry certified — no further
        distinct plan exists under the criterion. ``False`` whenever ``k``
        plans were returned, since more may exist.
    jaccard
        Pairwise Jaccard distance between the plans' changed sets, in
        mapping order: zero on the diagonal, one for disjoint sets.
    menu
        The ``RecourseMenu`` the plans were read from (``"levers"`` only;
        ``None`` for the coalition criterion).
    """

    plans: dict[str, Counterfactual]
    criterion: str
    complete: bool
    jaccard: FloatArray
    menu: RecourseMenu | None = None

    def __getitem__(self, key: str) -> Counterfactual:
        return self.plans[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.plans)

    def __len__(self) -> int:
        return len(self.plans)

    def to_frame(self) -> list[dict[str, object]]:
        """One row per plan, cheapest first.

        Returns
        -------
        ``[{"key", "distance", "proof", "changed"}, ...]``.
        """
        return [
            {
                "key": key,
                "distance": plan.distance,
                "proof": plan.proof,
                "changed": sorted(plan.changes),
            }
            for key, plan in self.plans.items()
        ]


def _jaccard(changed: Sequence[frozenset[str]]) -> FloatArray:
    n = len(changed)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            union = changed[i] | changed[j]
            distance = 1.0 - len(changed[i] & changed[j]) / len(union) if union else 0.0
            out[i, j] = out[j, i] = distance
    return out


def _candidate_levers(explainer: Explainer, x: FloatArray, target: Target) -> tuple[str, ...]:
    profile = explainer.search_profile(x, target)
    features = profile["features"]
    assert isinstance(features, dict)
    return tuple(
        sorted(
            name
            for name, info in features.items()
            if not info["frozen"] and info["domain"] > 1 and info["influential"]
        )
    )


def _warn_factual_once(explainer: Explainer, x: FloatArray) -> None:
    violations = explainer.compiled.factual_violations(x)
    if violations:
        warnings.warn(
            f"factual violates {len(violations)} constraint(s): "
            + "; ".join(violations)
            + ". The returned plans will include changes made solely to satisfy them.",
            TreecfWarning,
            stacklevel=4,  # _warn_factual_once <- build_* <- Explainer method <- user code
        )


def _warm_rust_cache(explainer: Explainer) -> None:
    """Marshal the ensembles once on the parent so every clone reuses them."""
    from treecf.backends.exact_rust import _rust_available
    from treecf.backends.genetic_rust import build_rust_ensemble

    if not _rust_available():
        return
    cache = explainer._rust_cache
    if "ensemble" not in cache:
        cache["ensemble"] = build_rust_ensemble(explainer.ir)
    plausibility = explainer._plausibility_bound()
    if plausibility is not None and "if_ensemble" not in cache:
        cache["if_ensemble"] = build_rust_ensemble(plausibility[0])


def _solve_levers(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    levers: Iterable[str],
    *,
    backend: str,
    time_budget_s: float,
    seed: int | None,
    warm_start: bool | None,
    node_budget: int | None,
    gap: float | None,
    search: str | None,
    degraded: list[Any],
) -> Counterfactual | Infeasible:
    free = set(levers)
    clone = explainer._with_extra_freezes([f for f in explainer.ir.feature_names if f not in free])
    return clone._explain_one(
        x, target, backend, time_budget_s, 0.0, seed, warn_factual=False,
        warm_start=warm_start, node_budget=node_budget, gap=gap, search=search,
        degraded=degraded,
    )


def build_menu(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    *,
    max_levers: int,
    mode: str,
    backend: str,
    search: str | None,
    seed: int | None,
    time_budget_s: float | None,
    total_budget_s: float | None,
    warm_start: bool | None,
    node_budget: int | None,
    gap: float | None,
) -> RecourseMenu:
    """The enumeration behind ``Explainer.recourse_menu``."""
    from treecf.api import (
        _DEFAULT_TIME_BUDGET_S,
        Counterfactual,
        _degraded_summary,
        _resolve_exact_kwargs,
    )
    from treecf.ir.evaluate import raw_score

    if target.bands_spec is not None:
        raise TreecfError("recourse_menu takes a single-interval target, not bands")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if max_levers < 1:
        raise ValueError(f"max_levers must be at least 1, got {max_levers!r}")
    if total_budget_s is not None and total_budget_s < 0.0:
        raise ValueError(f"total_budget_s must be non-negative, got {total_budget_s!r}")
    _resolve_exact_kwargs(backend, warm_start, node_budget, gap, search)
    x = np.asarray(x, dtype=np.float64)
    lo, hi = target.raw_interval(explainer.ir.link)
    if lo <= raw_score(explainer.ir, x) <= hi:
        raise TreecfError(
            "the factual already satisfies the target; there is no lever set to enumerate"
        )
    _warn_factual_once(explainer, x)
    per_solve = _DEFAULT_TIME_BUDGET_S if time_budget_s is None else time_budget_s
    levers = _candidate_levers(explainer, x, target)
    _warm_rust_cache(explainer)

    start = time.monotonic()
    degraded: list[Any] = []
    entries: dict[str, Counterfactual | Infeasible] = {}
    implied: dict[str, str] = {}
    certified_infeasible: list[str] = []
    unresolved: list[str] = []
    feasible_sets: list[frozenset[str]] = []
    n_sets = 0

    def check_consistency(set_key: str, members: frozenset[str]) -> None:
        for feasible in feasible_sets:
            if feasible <= members:
                raise TreecfError(
                    f"contradictory verdicts: lever set {set_key!r} is certified infeasible "
                    f"but its subset {_key(feasible)!r} has a feasible plan; this is a solver "
                    "inconsistency, please report it"
                )

    for size in range(1, max_levers + 1):
        for combo in itertools.combinations(levers, size):
            members = frozenset(combo)
            set_key = _key(combo)
            if mode == "minimal" and any(f <= members for f in feasible_sets):
                continue
            n_sets += 1
            if total_budget_s is not None and time.monotonic() - start >= total_budget_s:
                unresolved.append(set_key)
                continue
            result = _solve_levers(
                explainer, x, target, combo, backend=backend, time_budget_s=per_solve,
                seed=seed, warm_start=warm_start, node_budget=node_budget, gap=gap,
                search=search, degraded=degraded,
            )
            if isinstance(result, Counterfactual):
                changed = frozenset(result.changes)
                key = _key(changed)
                if changed != members:
                    implied[set_key] = key
                for infeasible_key in certified_infeasible:
                    if changed <= _members(infeasible_key):
                        check_consistency(infeasible_key, _members(infeasible_key))
                current = entries.get(key)
                if (
                    current is None
                    or not isinstance(current, Counterfactual)
                    or result.distance < current.distance
                ):
                    entries[key] = result
                if changed not in feasible_sets:
                    feasible_sets.append(changed)
                    check_all = [k for k in certified_infeasible if changed <= _members(k)]
                    for k in check_all:
                        check_consistency(k, _members(k))
            else:
                if result.proof == "certified":
                    check_consistency(set_key, members)
                    certified_infeasible.append(set_key)
                if not isinstance(entries.get(set_key), Counterfactual):
                    entries[set_key] = result

    feasible = {k: e for k, e in entries.items() if isinstance(e, Counterfactual)}
    minimal = [
        k
        for k in feasible
        if not any(other != k and _members(other) < _members(k) for other in feasible)
    ]
    minimal.sort(key=lambda k: (feasible[k].distance, k))
    listed = set(minimal)
    rest = sorted((k for k in feasible if k not in listed), key=lambda k: (feasible[k].distance, k))
    infeasible = sorted(
        (k for k in entries if k not in feasible), key=lambda k: (len(_members(k)), k)
    )
    ordered = {k: entries[k] for k in [*minimal, *rest, *infeasible]}
    complete = not unresolved and all(_is_certified(e) for e in ordered.values())

    uncertified = sum(1 for e in ordered.values() if not _is_certified(e))
    parts: list[str] = []
    if unresolved:
        parts.append(f"{len(unresolved)}/{n_sets} lever sets unresolved within total_budget_s")
    if backend == "exact" and uncertified:
        parts.append(f"{uncertified} resolved lever set(s) carry no certificate")
        summary = _degraded_summary(degraded, len(degraded), n_sets, "solves")
        if summary is not None:
            parts.append(summary)
    if parts:
        warnings.warn(
            "recourse menu: " + "; ".join(parts) + "; complete=False.",
            TreecfWarning,
            stacklevel=3,  # build_menu <- Explainer.recourse_menu <- user code
        )
    return RecourseMenu(
        entries=ordered,
        minimal=tuple(minimal),
        implied=implied,
        certified_infeasible=tuple(certified_infeasible),
        unresolved=tuple(unresolved),
        complete=complete,
        mode=mode,
        max_levers=max_levers,
        levers=levers,
    )


def build_diverse(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    *,
    k: int,
    diversity: str,
    coalitions: Mapping[str, Sequence[str]] | None,
    max_levers: int,
    menu_kwargs: dict[str, Any],
) -> DiverseSet:
    """The selection behind ``Explainer.explain_diverse``."""
    from treecf.api import Counterfactual

    if diversity not in _CRITERIA:
        raise ValueError(f"diversity must be one of {_CRITERIA}, got {diversity!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")
    options = _menu_options(menu_kwargs)
    if diversity == "levers":
        if coalitions is not None:
            raise ValueError("coalitions= applies only to diversity='coalitions'")
        menu = build_menu(explainer, x, target, max_levers=max_levers, mode="minimal", **options)
        feasible = [(key, e) for key, e in menu.items() if isinstance(e, Counterfactual)]
        plans = dict(feasible[:k])
        complete = len(feasible) < k and menu.complete
        changed = [frozenset(plan.changes) for plan in plans.values()]
        return DiverseSet(
            plans=plans, criterion="levers", complete=complete, jaccard=_jaccard(changed),
            menu=menu,
        )
    if coalitions is None:
        raise TreecfError("diversity='coalitions' needs coalitions= in the explain_coalitions form")
    plans_by_key, exhausted, all_certified = _coalition_ladder(
        explainer, x, target, coalitions, k=k, options=options
    )
    ordered = sorted(plans_by_key.items(), key=lambda item: (item[1].distance, item[0]))
    plans = dict(ordered[:k])
    complete = len(ordered) < k and exhausted and all_certified
    changed = [frozenset(plan.changes) for plan in plans.values()]
    return DiverseSet(
        plans=plans, criterion="coalitions", complete=complete, jaccard=_jaccard(changed),
        menu=None,
    )


def _coalition_ladder(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    coalitions: Mapping[str, Sequence[str]],
    *,
    k: int,
    options: dict[str, Any],
) -> tuple[dict[str, Counterfactual], bool, bool]:
    """Solve each coalition, then unions of two, three, ... until ``k``
    distinct feasible plans exist or the unions run out."""
    from treecf.api import (
        _DEFAULT_TIME_BUDGET_S,
        Counterfactual,
        _degraded_summary,
        _resolve_exact_kwargs,
        _validate_coalitions,
    )

    backend = options["backend"]
    search = options["search"]
    seed = options["seed"]
    time_budget_s = options["time_budget_s"]
    warm_start = options["warm_start"]
    node_budget = options["node_budget"]
    gap = options["gap"]
    _resolve_exact_kwargs(backend, warm_start, node_budget, gap, search)
    if target.bands_spec is not None:
        raise TreecfError("explain_diverse takes a single-interval target, not bands")
    x = np.asarray(x, dtype=np.float64)
    _warn_factual_once(explainer, x)
    normalized = _validate_coalitions(coalitions, explainer.ir.feature_names, False)
    names = list(normalized)
    per_solve = _DEFAULT_TIME_BUDGET_S if time_budget_s is None else time_budget_s
    _warm_rust_cache(explainer)

    degraded: list[Any] = []
    by_changed: dict[frozenset[str], tuple[str, Counterfactual]] = {}
    all_certified = True
    exhausted = True
    for level in range(1, len(names) + 1):
        for combo in itertools.combinations(names, level):
            levers: set[str] = set()
            for name in combo:
                levers.update(normalized[name])
            result = _solve_levers(
                explainer, x, target, levers, backend=backend, time_budget_s=per_solve,
                seed=seed, warm_start=warm_start, node_budget=node_budget, gap=gap,
                search=search, degraded=degraded,
            )
            all_certified = all_certified and _is_certified(result)
            if not isinstance(result, Counterfactual):
                continue
            changed = frozenset(result.changes)
            key = "+".join(combo)
            held = by_changed.get(changed)
            if held is None or result.distance < held[1].distance:
                by_changed[changed] = (key, result)
        if len(by_changed) >= k and level < len(names):
            exhausted = False
            break
    summary = _degraded_summary(degraded, len(degraded), max(len(degraded), 1), "solves")
    if summary is not None:
        warnings.warn(summary, TreecfWarning, stacklevel=4)
    return {key: plan for key, plan in by_changed.values()}, exhausted, all_certified


__all__ = ["DiverseSet", "RecourseMenu", "build_diverse", "build_menu"]
