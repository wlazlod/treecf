"""IR + constraints + objective -> AimProblem (spec §5).

Fixed-point rounding is conservative so that any solver value maps back to a
float strictly inside its cell (given breakpoints spaced > 2/K, checked here):
closed bounds round inward (ceil for lo, floor for hi), strict bounds step one
integer unit inside.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell, cell_index, feature_cells
from treecf.aim.model import (
    AimProblem,
    BuildInfeasible,
    CellDomain,
    FeatureBlock,
    LeafSpec,
    ScaledImplication,
    ScaledLinear,
    ScaledOneHot,
    TreeBlock,
)
from treecf.constraints.compile import CompiledConstraints
from treecf.ir.model import EnsembleIR, Node, SplitOp, Tree

FloatArray = npt.NDArray[np.float64]


def build_problem(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    scale_k: int = 10**6,
    scale_q: int = 10**6,
    plausibility: tuple[EnsembleIR, float] | None = None,
) -> AimProblem | BuildInfeasible:
    if plausibility is not None:
        per_feature_cells = feature_cells(ir, plausibility[0])
    else:
        per_feature_cells = feature_cells(ir)
    check = _check_breakpoint_resolution(per_feature_cells, scale_k)
    if check is not None:
        return check

    lo_b, hi_b, frozen = compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)  # Monotone on a NaN factual: no bound
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)

    features: list[FeatureBlock] = []
    fixed_values: dict[int, float] = {}
    block_of: dict[int, int] = {}
    for j in range(ir.n_features):
        if lo_b[j] > hi_b[j]:
            return BuildInfeasible(
                reason=f"contradictory constraints on feature {ir.feature_names[j]!r}"
            )
        allow = j in compiled.allow_missing and not frozen[j]
        if frozen[j] or (math.isnan(x[j]) and not allow):
            # Fixed features carry no variables; their routing is resolved statically.
            fixed_values[j] = float(x[j])
            continue
        deltas = compiled.allow_missing.get(j, (0.0, 0.0))
        block = _feature_block(
            j, ir.feature_names[j], per_feature_cells[j], x[j], lo_b[j], hi_b[j],
            weights[j], sigma[j], scale_k, scale_q,
            binary=j in compiled.binary_features,
            allow_missing=allow,
            delta_to_scaled=round(scale_k * deltas[0]),
            delta_from_scaled=round(scale_k * deltas[1]),
        )
        if block is None:
            return BuildInfeasible(
                reason=f"constraints leave no feasible value for feature {ir.feature_names[j]!r}"
            )
        block_of[j] = len(features)
        features.append(block)

    cells_by_block = [per_feature_cells[block.index] for block in features]
    trees: list[TreeBlock] = []
    for tree in ir.trees:
        leaves = _tree_leaves(tree, features, cells_by_block, block_of, fixed_values, scale_k)
        if not leaves:
            return BuildInfeasible(
                reason="a tree has no reachable leaf under the constraints"
            )
        trees.append(TreeBlock(leaves=tuple(leaves)))

    # Target bounds are widened OUTWARD by the fixed-point error bound (T+1)/2 units:
    # boundary-exact optima are common (tree scores are piecewise constant), so inward
    # narrowing would systematically reject them at any K. False accepts inside the
    # widening margin are caught by float-space verification and a K*10 retry (§8.1).
    n_trees = len(ir.trees)
    widen = (n_trees + 1) / 2.0
    lo_t, hi_t = interval
    natural_lo, natural_hi = _score_range(trees, round(scale_k * ir.base_score))
    score_lo = natural_lo if lo_t == -math.inf else math.floor(scale_k * lo_t - widen)
    score_hi = natural_hi if hi_t == math.inf else math.ceil(scale_k * hi_t + widen)
    if score_lo > score_hi:
        return BuildInfeasible(reason="empty target interval")

    plaus_trees: tuple[TreeBlock, ...] = ()
    plaus_lo = 0
    if plausibility is not None:
        if_ir, min_total_path = plausibility
        blocks = []
        for tree in if_ir.trees:
            leaves = _tree_leaves(tree, features, cells_by_block, block_of, fixed_values, scale_k)
            if not leaves:
                return BuildInfeasible(reason="isolation forest tree has no reachable leaf")
            blocks.append(TreeBlock(leaves=tuple(leaves)))
        plaus_trees = tuple(blocks)
        # widened outward like the target (§5.4); float verification closes the gap
        plaus_lo = math.floor(scale_k * min_total_path - (len(if_ir.trees) + 1) / 2.0)

    pos_of = {block.index: pos for pos, block in enumerate(features)}
    encoded = _encode_relational(compiled, fixed_values, pos_of, features, scale_k, scale_q)
    if isinstance(encoded, BuildInfeasible):
        return encoded
    linears, implications, onehots, must_have_value = encoded

    return AimProblem(
        features=tuple(features),
        trees=tuple(trees),
        base_scaled=round(scale_k * ir.base_score),
        score_lo=score_lo,
        score_hi=score_hi,
        lambda_scaled=round(scale_q * scale_k * lam),
        scale_k=scale_k,
        scale_q=scale_q,
        linears=linears,
        implications=implications,
        onehots=onehots,
        must_have_value=must_have_value,
        plaus_trees=plaus_trees,
        plaus_lo=plaus_lo,
    )


def swap_target(problem: AimProblem, interval: tuple[float, float]) -> AimProblem:
    """Rebind only the score bounds — the ladder amortization (§6: 1 build, N solves)."""
    import dataclasses

    n_trees = len(problem.trees)
    widen = (n_trees + 1) / 2.0
    natural_lo, natural_hi = _score_range(list(problem.trees), problem.base_scaled)
    lo_t, hi_t = interval
    score_lo = natural_lo if lo_t == -math.inf else math.floor(problem.scale_k * lo_t - widen)
    score_hi = natural_hi if hi_t == math.inf else math.ceil(problem.scale_k * hi_t + widen)
    return dataclasses.replace(problem, score_lo=score_lo, score_hi=score_hi)


def _check_breakpoint_resolution(
    per_feature_cells: tuple[tuple[Cell, ...], ...], scale_k: int
) -> BuildInfeasible | None:
    for cells in per_feature_cells:
        breakpoints = sorted({c.lo for c in cells if math.isfinite(c.lo)})
        for a, b in itertools.pairwise(breakpoints):
            if b - a <= 2.0 / scale_k:
                return BuildInfeasible(
                    reason=f"breakpoints {a} and {b} closer than 2/K; raise K",
                    resolution_related=True,
                )
    return None


def _feature_block(
    j: int,
    name: str,
    cells: tuple[Cell, ...],
    x_j: float,
    lo_bound: float,
    hi_bound: float,
    weight: float,
    sigma_j: float,
    scale_k: int,
    scale_q: int,
    binary: bool = False,
    allow_missing: bool = False,
    delta_to_scaled: int = 0,
    delta_from_scaled: int = 0,
) -> FeatureBlock | None:
    factual_missing = math.isnan(x_j)
    anchor = 0.0 if factual_missing else x_j
    finite = [c.lo for c in cells if math.isfinite(c.lo)]
    if math.isfinite(lo_bound):
        finite.append(lo_bound)
    if math.isfinite(hi_bound):
        finite.append(hi_bound)
    span_lo = min([anchor, *finite]) - 1.0
    span_hi = max([anchor, *finite]) + 1.0

    domains: list[CellDomain] = []
    for c_idx, cell in enumerate(cells):
        lo = max(cell.lo if math.isfinite(cell.lo) else span_lo, lo_bound)
        hi = min(cell.hi if math.isfinite(cell.hi) else span_hi, hi_bound)
        if lo > hi:
            continue
        lo_int = _lower_int(lo, strict=cell.lo_open and lo == cell.lo, scale_k=scale_k)
        hi_int = _upper_int(hi, strict=cell.hi_open and hi == cell.hi, scale_k=scale_k)
        if lo_int > hi_int:
            continue
        domains.append(CellDomain(cell_index=c_idx, v_lo=lo_int, v_hi=hi_int))
    if not domains:
        return None

    x_scaled = round(scale_k * anchor)
    x_cell: int | None = None
    if not factual_missing and lo_bound <= x_j <= hi_bound:
        # "unchanged" is only an option if x itself is feasible
        factual_cell = cell_index(cells, x_j)
        for pos, domain in enumerate(domains):
            if domain.cell_index == factual_cell:
                x_cell = pos
                # keep the factual anchor representable inside its own cell at scale K
                x_scaled = min(max(x_scaled, domain.v_lo), domain.v_hi)
                break

    return FeatureBlock(
        index=j,
        name=name,
        cells=tuple(domains),
        v_lo=min(d.v_lo for d in domains),
        v_hi=max(d.v_hi for d in domains),
        x_scaled=x_scaled,
        x_cell=x_cell,
        dist_coef=round(scale_q * weight / sigma_j),
        binary=binary,
        allow_missing=allow_missing,
        factual_missing=factual_missing,
        delta_to_scaled=delta_to_scaled,
        delta_from_scaled=delta_from_scaled,
    )


def _encode_relational(
    compiled: CompiledConstraints,
    fixed_values: dict[int, float],
    pos_of: dict[int, int],
    features: list[FeatureBlock],
    scale_k: int,
    scale_q: int,
) -> tuple[
    tuple[ScaledLinear, ...],
    tuple[ScaledImplication, ...],
    tuple[ScaledOneHot, ...],
    tuple[int, ...],
] | BuildInfeasible:
    """Resolve Linear/Implies/OneHot to block positions, folding fixed features."""
    linears: list[ScaledLinear] = []
    must_have_value: set[int] = set()
    for lin in compiled.linears:
        integral = all(float(c).is_integer() for c in lin.coefs)
        qc = 1 if integral else scale_q
        rhs_float = qc * scale_k * lin.rhs
        terms: list[tuple[int, int]] = []
        gating: list[int] = []
        dropped = False
        for j, coef in zip(lin.indices, lin.coefs, strict=True):
            if j in fixed_values:
                value = fixed_values[j]
                if math.isnan(value):
                    if lin.missing_policy == "satisfied":
                        dropped = True  # vacuously true (§4.2)
                        break
                    return BuildInfeasible(
                        reason=f"Linear over missing frozen feature index {j} "
                        f"with missing_policy={lin.missing_policy!r}"
                    )
                rhs_float -= qc * coef * scale_k * value
            else:
                pos = pos_of[j]
                terms.append((pos, round(qc * coef)))
                if features[pos].allow_missing:
                    if lin.missing_policy == "satisfied":
                        gating.append(pos)
                    else:  # "violated" / "forbid_missing": NaN is not an option here
                        must_have_value.add(pos)
        if dropped:
            continue
        rhs = round(rhs_float)
        if not terms:  # fully static: check numerically
            if not _op_holds(0, lin.op, rhs):
                return BuildInfeasible(reason="Linear constraint over fixed features fails")
            continue
        linears.append(
            ScaledLinear(
                terms=tuple(terms), op=lin.op, rhs=rhs, enforce_not_missing=tuple(gating)
            )
        )

    implications: list[ScaledImplication] = []
    for imp in compiled.implications:
        cond_fixed = imp.cond_index in fixed_values
        cons_fixed = imp.cons_index in fixed_values
        if cond_fixed:
            if abs(fixed_values[imp.cond_index] - imp.cond_value) > 1e-9:
                continue  # condition statically false: implication is vacuous
            if cons_fixed:
                if abs(fixed_values[imp.cons_index] - imp.cons_value) > 1e-9:
                    return BuildInfeasible(reason="Implies over fixed features fails")
                continue
            # condition true: force the consequence
            linears.append(
                ScaledLinear(
                    terms=((pos_of[imp.cons_index], 1),),
                    op="==",
                    rhs=round(scale_k * imp.cons_value),
                )
            )
            continue
        if cons_fixed:
            if abs(fixed_values[imp.cons_index] - imp.cons_value) <= 1e-9:
                continue  # consequence statically true
            # consequence false: condition must not hold
            linears.append(
                ScaledLinear(
                    terms=((pos_of[imp.cond_index], 1),),
                    op="==",
                    rhs=round(scale_k * (1.0 - imp.cond_value)),
                )
            )
            continue
        implications.append(
            ScaledImplication(
                cond_pos=pos_of[imp.cond_index],
                cond_is_one=imp.cond_value == 1.0,
                cons_pos=pos_of[imp.cons_index],
                cons_is_one=imp.cons_value == 1.0,
            )
        )

    onehots: list[ScaledOneHot] = []
    for group in compiled.onehot_groups:
        required = 1
        positions: list[int] = []
        for j in group:
            if j in fixed_values:
                value = fixed_values[j]
                if math.isnan(value) or value not in (0.0, 1.0):
                    return BuildInfeasible(
                        reason="OneHot group contains a fixed non-binary value"
                    )
                required -= int(value)
            else:
                positions.append(pos_of[j])
        if required < 0 or required > 1:
            return BuildInfeasible(reason="OneHot group is over-determined by fixed features")
        if not positions:
            if required != 0:
                return BuildInfeasible(reason="OneHot group of fixed features does not sum to 1")
            continue
        onehots.append(ScaledOneHot(positions=tuple(positions), required=required))

    return tuple(linears), tuple(implications), tuple(onehots), tuple(sorted(must_have_value))


def _op_holds(lhs: float, op: str, rhs: float) -> bool:
    if op == "<=":
        return lhs <= rhs
    if op == ">=":
        return lhs >= rhs
    return lhs == rhs


def _lower_int(lo: float, strict: bool, scale_k: int) -> int:
    if strict:
        return math.floor(scale_k * lo) + 1
    return math.ceil(scale_k * lo)


def _upper_int(hi: float, strict: bool, scale_k: int) -> int:
    if strict:
        return math.ceil(scale_k * hi) - 1
    return math.floor(scale_k * hi)


def _tree_leaves(
    tree: Tree,
    features: list[FeatureBlock],
    cells_by_block: list[tuple[Cell, ...]],
    block_of: dict[int, int],
    fixed_values: dict[int, float],
    scale_k: int,
) -> list[LeafSpec]:
    """Enumerate reachable leaves with per-feature admissible (cells, missing) states."""
    leaves: list[LeafSpec] = []
    State = tuple[frozenset[int], bool]  # (admissible cell positions, missing admissible)

    def full_state(block_idx: int) -> State:
        block = features[block_idx]
        return frozenset(range(len(block.cells))), block.allow_missing

    def walk(node: Node, admissible: dict[int, State], alive: bool) -> None:
        if not alive:
            return
        if node.feature is None:
            assert node.value is not None
            conditions = []
            for block_idx, (positions, missing_ok) in sorted(admissible.items()):
                if (positions, missing_ok) == full_state(block_idx):
                    continue  # unconstrained
                conditions.append((block_idx, tuple(sorted(positions)), missing_ok))
            leaves.append(
                LeafSpec(
                    leaf_id=node.node_id,
                    value_scaled=round(scale_k * node.value),
                    conditions=tuple(conditions),
                )
            )
            return
        assert node.left is not None and node.right is not None
        j = node.feature
        left_node, right_node = tree.nodes[node.left], tree.nodes[node.right]

        if j in fixed_values:
            # Fixed feature (frozen or factual NaN): routing is static.
            value = fixed_values[j]
            if math.isnan(value):
                goes_left = bool(node.missing_left)
            else:
                assert node.threshold is not None
                goes_left = (
                    value < node.threshold
                    if node.op is SplitOp.LT
                    else value <= node.threshold
                )
            walk(left_node if goes_left else right_node, admissible, alive)
            return

        block_idx = block_of[j]
        block = features[block_idx]
        positions, missing_ok = admissible.get(block_idx, full_state(block_idx))
        left_set, right_set = set(), set()
        for pos in positions:
            cell = cells_by_block[block_idx][block.cells[pos].cell_index]
            if _cell_on_left(cell, node):
                left_set.add(pos)
            else:
                right_set.add(pos)
        missing_goes_left = bool(node.missing_left) if block.allow_missing else False
        for child, side_set, side_missing in (
            (left_node, left_set, missing_ok and missing_goes_left),
            (right_node, right_set, missing_ok and not missing_goes_left),
        ):
            child_admissible = dict(admissible)
            child_admissible[block_idx] = (frozenset(side_set), side_missing)
            walk(child, child_admissible, bool(side_set) or side_missing)

    walk(tree.nodes[0], {}, alive=True)
    return leaves


def _cell_on_left(cell: Cell, node: Node) -> bool:
    """Whether the cell lies entirely on the split's left side (cells are atomic, §5.1)."""
    assert node.threshold is not None and node.op is not None
    if node.op is SplitOp.LT:  # left side is x < threshold
        return cell.hi < node.threshold or (cell.hi == node.threshold and cell.hi_open)
    return cell.hi <= node.threshold  # left side is x <= threshold


def _score_range(trees: list[TreeBlock], base_scaled: int) -> tuple[int, int]:
    lo = base_scaled + sum(min(leaf.value_scaled for leaf in t.leaves) for t in trees)
    hi = base_scaled + sum(max(leaf.value_scaled for leaf in t.leaves) for t in trees)
    return lo, hi
