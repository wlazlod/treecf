"""CP-SAT adapter (spec §8.1). ortools is imported lazily inside the solve call."""

from __future__ import annotations

from typing import Any

from treecf._errors import MissingExtraError
from treecf.aim.model import AimProblem
from treecf.backends.base import BackendSolution


class CpsatBackend:
    def solve(
        self,
        problem: AimProblem,
        time_budget_s: float = 10.0,
        num_workers: int = 0,
    ) -> BackendSolution:
        cp_model = _import_cp_model()
        model = cp_model.CpModel()

        cell_bools: list[list[Any]] = []
        v_vars: list[Any] = []
        d_vars: list[Any] = []
        z_vars: list[Any] = []
        for block in problem.features:
            bools = [
                model.NewBoolVar(f"cell_{block.index}_{d.cell_index}") for d in block.cells
            ]
            model.AddExactlyOne(bools)
            v = model.NewIntVar(block.v_lo, block.v_hi, f"v_{block.index}")
            for domain, b in zip(block.cells, bools, strict=True):
                model.Add(v >= domain.v_lo).OnlyEnforceIf(b)
                model.Add(v <= domain.v_hi).OnlyEnforceIf(b)
            d_hi = max(block.v_hi - block.x_scaled, block.x_scaled - block.v_lo, 0)
            d = model.NewIntVar(0, d_hi, f"d_{block.index}")
            model.Add(d >= v - block.x_scaled)
            model.Add(d >= block.x_scaled - v)
            z = model.NewBoolVar(f"z_{block.index}")
            if block.x_cell is None:
                # factual value violates the constraints: the feature must change
                model.Add(z == 1)
            else:
                model.Add(v == block.x_scaled).OnlyEnforceIf(z.Not())
            cell_bools.append(bools)
            v_vars.append(v)
            d_vars.append(d)
            z_vars.append(z)

        score_terms = [problem.base_scaled]
        for t_idx, tree in enumerate(problem.trees):
            leaf_bools = [
                model.NewBoolVar(f"leaf_{t_idx}_{leaf.leaf_id}") for leaf in tree.leaves
            ]
            model.AddExactlyOne(leaf_bools)
            for leaf, lb in zip(tree.leaves, leaf_bools, strict=True):
                for block_idx, positions in leaf.conditions:
                    model.Add(
                        sum(cell_bools[block_idx][pos] for pos in positions) >= 1
                    ).OnlyEnforceIf(lb)
            score_terms.extend(
                leaf.value_scaled * lb
                for leaf, lb in zip(tree.leaves, leaf_bools, strict=True)
            )

        score = sum(score_terms)
        model.Add(score >= problem.score_lo)
        model.Add(score <= problem.score_hi)

        model.Minimize(
            sum(
                block.dist_coef * d for block, d in zip(problem.features, d_vars, strict=True)
            )
            + problem.lambda_scaled * sum(z_vars)
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget_s
        if num_workers:
            solver.parameters.num_search_workers = num_workers
        status = solver.Solve(model)

        stats: dict[str, object] = {
            "status": solver.StatusName(status),
            "wall_time_s": solver.WallTime(),
            "branches": solver.NumBranches(),
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return BackendSolution(
                status="infeasible", values_scaled=None, objective=None, gap=None, stats=stats
            )

        descale = float(problem.scale_k) * float(problem.scale_q)
        objective = solver.ObjectiveValue() / descale
        gap = (solver.ObjectiveValue() - solver.BestObjectiveBound()) / descale
        values_scaled = {
            block.index: solver.Value(v)
            for block, v in zip(problem.features, v_vars, strict=True)
        }
        return BackendSolution(
            status="optimal" if status == cp_model.OPTIMAL else "feasible",
            values_scaled=values_scaled,
            objective=objective,
            gap=gap,
            stats=stats,
        )


def _import_cp_model() -> Any:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise MissingExtraError(
            "backend='cpsat' requires ortools: pip install treecf[cpsat]"
        ) from exc
    return cp_model
