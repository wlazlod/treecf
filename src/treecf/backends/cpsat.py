"""CP-SAT adapter (spec §8.1). ortools is imported lazily inside the solve call."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from treecf._errors import MissingExtraError
from treecf.aim.model import AimProblem
from treecf.backends.base import BackendSolution, DiversityCut


class CpsatBackend:
    def solve(
        self,
        problem: AimProblem,
        time_budget_s: float = 10.0,
        num_workers: int = 0,
        cuts: Sequence[DiversityCut] = (),
    ) -> BackendSolution:
        cp_model = _import_cp_model()
        model = cp_model.CpModel()

        cell_bools: list[list[Any]] = []
        v_vars: list[Any] = []
        d_vars: list[Any] = []
        z_vars: list[Any] = []
        b_vars: dict[int, Any] = {}  # block position -> boolean for binary features
        m_vars: dict[int, Any] = {}  # block position -> "is missing" boolean (§4.2)
        for pos, block in enumerate(problem.features):
            bools = [
                model.NewBoolVar(f"cell_{block.index}_{d.cell_index}") for d in block.cells
            ]
            m = None
            if block.allow_missing:
                m = model.NewBoolVar(f"m_{block.index}")
                m_vars[pos] = m
                model.AddExactlyOne([*bools, m])
            else:
                model.AddExactlyOne(bools)
            v = model.NewIntVar(block.v_lo, block.v_hi, f"v_{block.index}")
            for domain, b in zip(block.cells, bools, strict=True):
                model.Add(v >= domain.v_lo).OnlyEnforceIf(b)
                model.Add(v <= domain.v_hi).OnlyEnforceIf(b)
            d_hi = max(
                block.v_hi - block.x_scaled,
                block.x_scaled - block.v_lo,
                block.delta_to_scaled,
                block.delta_from_scaled,
                0,
            )
            d = model.NewIntVar(0, d_hi, f"d_{block.index}")
            z = model.NewBoolVar(f"z_{block.index}")

            # z must be an exact change indicator in BOTH directions, otherwise
            # diversity cuts (§8.3) could be dodged by marking unchanged features.
            # (The z=1 => v != x disjunction has the standard big-M linear
            # equivalent for the MILP subset, §8.4.)
            if block.factual_missing:
                assert m is not None
                # unchanged = stay NaN; taking a value costs delta_from and counts as change
                model.Add(d == 0).OnlyEnforceIf(m)
                model.Add(d == block.delta_from_scaled).OnlyEnforceIf(m.Not())
                model.Add(m == 1).OnlyEnforceIf(z.Not())
                model.AddImplication(m.Not(), z)
                model.AddImplication(z, m.Not())
            else:
                if m is not None:
                    model.Add(d >= v - block.x_scaled).OnlyEnforceIf(m.Not())
                    model.Add(d >= block.x_scaled - v).OnlyEnforceIf(m.Not())
                    model.Add(d == block.delta_to_scaled).OnlyEnforceIf(m)
                    model.AddImplication(m, z)
                    model.AddImplication(z.Not(), m.Not())
                else:
                    model.Add(d >= v - block.x_scaled)
                    model.Add(d >= block.x_scaled - v)
                if block.x_cell is None:
                    # factual value violates the constraints: the feature must change
                    model.Add(z == 1)
                else:
                    model.Add(v == block.x_scaled).OnlyEnforceIf(z.Not())
                    if m is not None:
                        model.Add(v != block.x_scaled).OnlyEnforceIf([z, m.Not()])
                    else:
                        model.Add(v != block.x_scaled).OnlyEnforceIf(z)

            if block.binary:
                b = model.NewBoolVar(f"b_{block.index}")
                model.Add(v == problem.scale_k * b)
                b_vars[pos] = b
            cell_bools.append(bools)
            v_vars.append(v)
            d_vars.append(d)
            z_vars.append(z)

        for pos in problem.must_have_value:
            model.Add(m_vars[pos] == 0)

        for lin in problem.linears:
            expr = sum(coef * v_vars[pos] for pos, coef in lin.terms)
            if lin.op == "<=":
                ct = model.Add(expr <= lin.rhs)
            elif lin.op == ">=":
                ct = model.Add(expr >= lin.rhs)
            else:
                ct = model.Add(expr == lin.rhs)
            if lin.enforce_not_missing:
                # missing_policy "satisfied": constraint applies only when all present
                ct.OnlyEnforceIf([m_vars[pos].Not() for pos in lin.enforce_not_missing])

        for imp in problem.implications:
            cond = b_vars[imp.cond_pos] if imp.cond_is_one else b_vars[imp.cond_pos].Not()
            cons = b_vars[imp.cons_pos] if imp.cons_is_one else b_vars[imp.cons_pos].Not()
            model.AddImplication(cond, cons)

        for onehot in problem.onehots:
            model.Add(sum(b_vars[pos] for pos in onehot.positions) == onehot.required)

        score_terms = [problem.base_scaled]
        for t_idx, tree in enumerate(problem.trees):
            leaf_bools = [
                model.NewBoolVar(f"leaf_{t_idx}_{leaf.leaf_id}") for leaf in tree.leaves
            ]
            model.AddExactlyOne(leaf_bools)
            for leaf, lb in zip(tree.leaves, leaf_bools, strict=True):
                for block_idx, positions, missing_ok in leaf.conditions:
                    terms = [cell_bools[block_idx][pos] for pos in positions]
                    if missing_ok:
                        terms.append(m_vars[block_idx])
                    model.Add(sum(terms) >= 1).OnlyEnforceIf(lb)
            score_terms.extend(
                leaf.value_scaled * lb
                for leaf, lb in zip(tree.leaves, leaf_bools, strict=True)
            )

        score = sum(score_terms)
        model.Add(score >= problem.score_lo)
        model.Add(score <= problem.score_hi)

        if problem.plaus_trees:
            plaus_terms: list[Any] = []
            for t_idx, tree in enumerate(problem.plaus_trees):
                leaf_bools = [
                    model.NewBoolVar(f"plaus_{t_idx}_{leaf.leaf_id}") for leaf in tree.leaves
                ]
                model.AddExactlyOne(leaf_bools)
                for leaf, lb in zip(tree.leaves, leaf_bools, strict=True):
                    for block_idx, positions, missing_ok in leaf.conditions:
                        terms = [cell_bools[block_idx][pos] for pos in positions]
                        if missing_ok:
                            terms.append(m_vars[block_idx])
                        model.Add(sum(terms) >= 1).OnlyEnforceIf(lb)
                plaus_terms.extend(
                    leaf.value_scaled * lb
                    for leaf, lb in zip(tree.leaves, leaf_bools, strict=True)
                )
            model.Add(sum(plaus_terms) >= problem.plaus_lo)

        for cut in cuts:
            if cut.mode == "distinct_changes":
                # forbid the exact change-set: some changed z drops or some unchanged z rises
                model.Add(
                    sum(z_vars[pos].Not() for pos in cut.changed)
                    + sum(z_vars[pos] for pos in cut.unchanged)
                    >= 1
                )
            else:  # distinct_solution: forbid the exact cell/missing assignment
                chosen_bools = [
                    m_vars[pos] if cell_pos == -1 else cell_bools[pos][cell_pos]
                    for pos, cell_pos in cut.chosen
                ]
                model.Add(sum(chosen_bools) <= len(chosen_bools) - 1)

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
        missing = {
            problem.features[pos].index: bool(solver.Value(m)) for pos, m in m_vars.items()
        }
        changed_positions = tuple(
            pos for pos, z in enumerate(z_vars) if solver.Value(z)
        )
        chosen_cells = []
        for pos, bools in enumerate(cell_bools):
            if pos in m_vars and solver.Value(m_vars[pos]):
                chosen_cells.append((pos, -1))
                continue
            for cell_pos, b in enumerate(bools):
                if solver.Value(b):
                    chosen_cells.append((pos, cell_pos))
                    break
        return BackendSolution(
            status="optimal" if status == cp_model.OPTIMAL else "feasible",
            values_scaled=values_scaled,
            objective=objective,
            gap=gap,
            stats=stats,
            missing=missing,
            changed_positions=changed_positions,
            chosen_cells=tuple(chosen_cells),
        )


def _import_cp_model() -> Any:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise MissingExtraError(
            "backend='cpsat' requires ortools: pip install treecf[cpsat]"
        ) from exc
    return cp_model
