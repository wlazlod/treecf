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
) -> AimProblem | BuildInfeasible:
    per_feature_cells = feature_cells(ir)
    check = _check_breakpoint_resolution(per_feature_cells, scale_k)
    if check is not None:
        return check

    lo_b, hi_b, frozen = compiled.instance_bounds(x)

    features: list[FeatureBlock] = []
    fixed_values: dict[int, float] = {}
    block_of: dict[int, int] = {}
    for j in range(ir.n_features):
        if lo_b[j] > hi_b[j]:
            return BuildInfeasible(
                reason=f"contradictory constraints on feature {ir.feature_names[j]!r}"
            )
        if math.isnan(x[j]) or frozen[j]:
            # Fixed features carry no variables; their routing is resolved statically.
            fixed_values[j] = float(x[j])
            continue
        block = _feature_block(
            j, ir.feature_names[j], per_feature_cells[j], x[j], lo_b[j], hi_b[j],
            weights[j], sigma[j], scale_k, scale_q,
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

    return AimProblem(
        features=tuple(features),
        trees=tuple(trees),
        base_scaled=round(scale_k * ir.base_score),
        score_lo=score_lo,
        score_hi=score_hi,
        lambda_scaled=round(scale_q * scale_k * lam),
        scale_k=scale_k,
        scale_q=scale_q,
    )


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
) -> FeatureBlock | None:
    finite = [c.lo for c in cells if math.isfinite(c.lo)]
    if math.isfinite(lo_bound):
        finite.append(lo_bound)
    if math.isfinite(hi_bound):
        finite.append(hi_bound)
    span_lo = min([x_j, *finite]) - 1.0
    span_hi = max([x_j, *finite]) + 1.0

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

    x_scaled = round(scale_k * x_j)
    x_cell: int | None = None
    if lo_bound <= x_j <= hi_bound:  # "unchanged" is only an option if x itself is feasible
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
    )


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
    """Enumerate reachable leaves; conditions are per-feature admissible cell positions."""
    leaves: list[LeafSpec] = []

    def walk(node: Node, admissible: dict[int, set[int]], alive: bool) -> None:
        if node.feature is None:
            assert node.value is not None
            if alive and all(positions for positions in admissible.values()):
                conditions = tuple(
                    (block_idx, tuple(sorted(positions)))
                    for block_idx, positions in sorted(admissible.items())
                    if len(positions) < len(features[block_idx].cells)
                )
                leaves.append(
                    LeafSpec(
                        leaf_id=node.node_id,
                        value_scaled=round(scale_k * node.value),
                        conditions=conditions,
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
            walk(left_node, admissible, alive and goes_left)
            walk(right_node, admissible, alive and not goes_left)
            return

        block_idx = block_of[j]
        block = features[block_idx]
        current = admissible.get(block_idx, set(range(len(block.cells))))
        left_set, right_set = set(), set()
        for pos in current:
            cell = cells_by_block[block_idx][block.cells[pos].cell_index]
            if _cell_on_left(cell, node):
                left_set.add(pos)
            else:
                right_set.add(pos)
        for child, side_set in ((left_node, left_set), (right_node, right_set)):
            child_admissible = dict(admissible)
            child_admissible[block_idx] = side_set
            walk(child, child_admissible, alive and bool(side_set))

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
