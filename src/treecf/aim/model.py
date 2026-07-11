"""Abstract intermediate model (spec §5): the backend-agnostic problem description.

Everything is integer at value scale K and coefficient scale Q. The structure
restricts itself to the MILP-safe subset (§8.4): one-hot cell booleans, integer
value variables linked to cells by half-reified implications, leaf booleans with
exactly-one per tree, and pure-boolean path conditions.

Scaling algebra (referenced by build.py and the backends):
- value var v_j        ~ round(K * x'_j)
- distance var d_j     >= |v_j - x_scaled_j|            (integer, scale K)
- dist_coef_j          = round(Q * w_j / sigma_j)
- lambda_scaled        = round(Q * K * lambda)
- objective            = sum_j dist_coef_j * d_j + lambda_scaled * sum_j z_j
                       ~ Q * K * J(x')
- leaf/base values     ~ round(K * value); score bounds pre-widened by (T+1)/2 units (§5.4)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellDomain:
    """One allowed cell of a feature: v_j must lie in [v_lo, v_hi] (integer, scale K)."""

    cell_index: int
    v_lo: int
    v_hi: int


@dataclass(frozen=True)
class FeatureBlock:
    """Decision variables of one mutable feature."""

    index: int
    name: str
    cells: tuple[CellDomain, ...]
    v_lo: int  # global bounds: min/max over cells
    v_hi: int
    x_scaled: int  # factual value at scale K (clamped into its cell)
    x_cell: int | None  # position in `cells` holding the factual value, if allowed
    dist_coef: int
    binary: bool = False  # v restricted to {0, K} via a boolean (Equals/Implies/OneHot)
    allow_missing: bool = False  # NaN is a feasible state (m boolean exists, §4.2)
    factual_missing: bool = False  # the factual value is NaN ("unchanged" means m = 1)
    delta_to_scaled: int = 0  # d when flipping value -> NaN (scale K)
    delta_from_scaled: int = 0  # d when flipping NaN -> value (scale K)


@dataclass(frozen=True)
class ScaledLinear:
    """sum(coef * v[block_pos]) op rhs, all integer at combined scale (spec §7.4).

    ``enforce_not_missing`` lists block positions whose m-booleans gate the
    constraint (missing_policy "satisfied": enforced only when all are present).
    """

    terms: tuple[tuple[int, int], ...]  # (block position, integer coefficient)
    op: str  # "<=" | ">=" | "=="
    rhs: int
    enforce_not_missing: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScaledImplication:
    """cond => cons over binary blocks: (block position, required boolean value)."""

    cond_pos: int
    cond_is_one: bool
    cons_pos: int
    cons_is_one: bool


@dataclass(frozen=True)
class ScaledOneHot:
    """sum of binary blocks at `positions` equals `required` (fixed members folded)."""

    positions: tuple[int, ...]
    required: int


@dataclass(frozen=True)
class LeafSpec:
    """A reachable leaf: value plus per-feature admissible states.

    Each condition is (block position, admissible cell positions, missing_ok):
    the leaf is reachable iff the feature is missing (when missing_ok) or its
    cell is one of the listed positions.
    """

    leaf_id: int
    value_scaled: int
    conditions: tuple[tuple[int, tuple[int, ...], bool], ...]


@dataclass(frozen=True)
class TreeBlock:
    leaves: tuple[LeafSpec, ...]


@dataclass(frozen=True)
class AimProblem:
    features: tuple[FeatureBlock, ...]  # mutable (non-NaN) features only
    trees: tuple[TreeBlock, ...]
    base_scaled: int
    score_lo: int
    score_hi: int
    lambda_scaled: int
    scale_k: int
    scale_q: int
    linears: tuple[ScaledLinear, ...] = ()
    implications: tuple[ScaledImplication, ...] = ()
    onehots: tuple[ScaledOneHot, ...] = ()
    must_have_value: tuple[int, ...] = ()  # block positions where m is forced to 0
    plaus_trees: tuple[TreeBlock, ...] = ()  # isolation forest (§9); not part of the score
    plaus_lo: int = 0  # sum of plausibility leaf values must be >= this (scale K)


@dataclass(frozen=True)
class BuildInfeasible:
    """The problem is unsatisfiable at build time."""

    reason: str
    resolution_related: bool = False  # True => retrying with a larger K may help
