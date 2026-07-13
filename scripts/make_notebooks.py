"""Generate and execute the three docs notebooks (committed executed, D15)."""

import sys

import nbformat as nbf
from nbclient import NotebookClient

OUT = "docs/notebooks"

DATA_CELL = '''
import numpy as np

rng = np.random.default_rng(42)
n = 6000
names = [
    "income_monthly", "utilization", "n_active_loans", "n_loans_total",
    "max_dpd_30d", "max_dpd_12m", "months_since_last_delinq", "age",
]
income = rng.lognormal(8.3, 0.5, n).round(-1)
utilization = rng.beta(2, 3, n).round(3)
n_total = rng.poisson(4, n).astype(float) + 1
n_active = np.minimum(np.floor(n_total * rng.beta(3, 2, n)), n_total)
dpd_12m = np.floor(rng.exponential(6, n)) * (rng.random(n) < 0.4)
dpd_30d = np.floor(dpd_12m * rng.beta(2, 4, n))
months_delinq = rng.exponential(14, n).round(0)
months_delinq[dpd_12m == 0] = np.nan          # no delinquency -> no record
age = rng.integers(21, 75, n).astype(float)

X = np.column_stack([income, utilization, n_active, n_total,
                     dpd_30d, dpd_12m, months_delinq, age])
risk = (
    -0.9 * np.log(income / 4000)
    + 2.2 * utilization
    + 0.35 * dpd_30d + 0.15 * dpd_12m
    - 0.02 * np.nan_to_num(months_delinq, nan=36.0)
    - 0.015 * age
)
y = (risk + rng.logistic(scale=0.8, size=n) > np.median(risk)).astype(int)
X.shape, y.mean().round(3)
'''

TRAIN_CELL = '''
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.2,
                          random_state=0)
model.fit(X, y)
model.get_booster().feature_names = list(names)   # domain names for explanations
proba = model.predict_proba(X)[:, 1]
cutoff = 0.30                     # the credit policy's PD cutoff
applicant = X[int(np.argmax(proba))]      # a clearly declined applicant
float(model.predict_proba(applicant.reshape(1, -1))[0, 1])
'''


def quickstart() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Quickstart\n\n"
            "Train an XGBoost credit model on synthetic data, then ask treecf the core "
            "question: *what is the minimal, feasible change that gets this applicant "
            "under the approval cutoff?* The search runs on treecf's bundled Rust engine "
            "and every answer is float-verified against the model before it is returned."
        ),
        nbf.v4.new_code_cell(DATA_CELL.strip()),
        nbf.v4.new_code_cell(TRAIN_CELL.strip()),
        nbf.v4.new_markdown_cell(
            "## Build the explainer\n\nThe background sample fits robust per-feature "
            "distance normalizers (MAD chain). Two domain constraints: age is immutable "
            "here, and the 30-day DPD can never exceed the 12-month DPD."
        ),
        nbf.v4.new_code_cell(
            "from treecf import Explainer, Freeze, Target, constraint\n\n"
            "exp = Explainer(\n"
            "    model,\n"
            "    background=X,\n"
            "    constraints=[\n"
            "        Freeze(\"age\"),\n"
            "        constraint(\"max_dpd_30d <= max_dpd_12m\"),\n"
            "        constraint(\"n_active_loans <= n_loans_total\"),\n"
            "    ],\n"
            "    weights={\"income_monthly\": 2.0},   # income is hard to change\n"
            ")\n"
            "res = exp.explain(applicant, target=Target.probability(range=(0.0, cutoff)),\n"
            "                  seed=0)\n"
            "res.proof, res.n_changed, round(res.score_prob, 4)"
        ),
        nbf.v4.new_code_cell("res.changes"),
        nbf.v4.new_markdown_cell("## Visualize the recommendation"),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_changes\n\nplot_changes(res);"
        ),
        nbf.v4.new_markdown_cell(
            "**Waterfall** (SHAP-style): each bar is the exact probability delta from "
            "one change, applied largest-first; the red line is the policy cutoff. "
            "**Effort** shows where the applicant's work goes instead."
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_effort, plot_waterfall\n\n"
            "plot_waterfall(exp, res, target=Target.probability(range=(0.0, cutoff)));"
        ),
        nbf.v4.new_code_cell("plot_effort(exp, res);"),
        nbf.v4.new_markdown_cell(
            "## Compare alternative plans\n\n"
            "One plan is rarely the whole story. `diversity=\"lever-blocking\"` re-solves "
            "with each plan's biggest lever frozen, producing structurally distinct "
            "alternatives. Compare them: every plan's changes on shared axes "
            "(standardized to Δ/σ, one color per plan), and what each plan costs "
            "against what it buys."
        ),
        nbf.v4.new_code_cell(
            "batch = exp.explain_batch(\n"
            "    applicant.reshape(1, -1),\n"
            "    target=Target.probability(range=(0.0, cutoff)),\n"
            "    n_per_example=3,\n"
            "    diversity=\"lever-blocking\",\n"
            "    seed=0,\n"
            ")\n"
            "plans = batch.for_id(0)\n"
            "[(round(p.distance, 2), p.blocked_lever, sorted(p.changes)) for p in plans]"
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_alternatives, plot_tradeoff\n\n"
            "plot_alternatives(plans, explainer=exp);   # deltas in sigma units"
        ),
        nbf.v4.new_code_cell(
            "plot_tradeoff(plans, target=Target.probability(range=(0.0, cutoff)));"
        ),
    ]
    return nb


def tutorial() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Credit-risk tutorial\n\nThe full workflow: mine candidate constraints "
            "from data, review and accept them, price a missing-value flip, build a "
            "rating ladder, and generate diverse alternatives."
        ),
        nbf.v4.new_code_cell(DATA_CELL.strip()),
        nbf.v4.new_code_cell(TRAIN_CELL.strip()),
        nbf.v4.new_markdown_cell(
            "## 1. Mine candidate constraints\n\nInvariants are mined from the sample "
            "and returned **for review** — nothing is auto-applied. Note the DPD "
            "hierarchy and loan-count order arriving as ready-to-paste code."
        ),
        nbf.v4.new_code_cell(
            "import treecf\n\n"
            "mined = treecf.suggest_constraints(X, feature_names=names)\n"
            "for s in mined[:8]:\n"
            "    print(s.as_code())"
        ),
        nbf.v4.new_code_cell(
            "accepted = [s.constraint for s in mined\n"
            "            if s.kind == \"order\" and s.support == 1.0]\n"
            "accepted"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Explain with a priced missing-value flip\n\n"
            "`months_since_last_delinq` is NaN when there is no delinquency record — "
            "and *reaching* that state is a legitimate recommendation with an explicit "
            "price (`delta_miss`)."
        ),
        nbf.v4.new_code_cell(
            "from treecf import AllowMissing, Explainer, Freeze, Target\n\n"
            "exp = Explainer(\n"
            "    model,\n"
            "    background=X,\n"
            "    constraints=accepted + [\n"
            "        Freeze(\"age\"),\n"
            "        AllowMissing(\"months_since_last_delinq\", delta_miss=2.0),\n"
            "    ],\n"
            "    value_policy={\"max_dpd_30d\": \"integer\", \"max_dpd_12m\": \"integer\",\n"
            "                  \"n_active_loans\": \"integer\", \"n_loans_total\": \"integer\"},\n"
            ")\n"
            "res = exp.explain(applicant, target=Target.probability(range=(0.0, cutoff)),\n"
            "                  seed=0)\n"
            "res.changes"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. The rating ladder\n\nOne search per band: the increasing cost of "
            "each better grade."
        ),
        nbf.v4.new_code_cell(
            "ladder = exp.explain(applicant, seed=0, target=Target.bands({\n"
            "    \"approve\": (0.00, 0.30),\n"
            "    \"prime\":   (0.00, 0.15),\n"
            "    \"super\":   (0.00, 0.05),\n"
            "}))\n"
            "{k: (round(v.distance, 3) if hasattr(v, 'distance') else v.reason)\n"
            " for k, v in ladder.items()}"
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_ladder\n\nplot_ladder(ladder);"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Alternative plans, and which levers are essential\n\n"
            "Block each of the primary plan's biggest levers in turn and re-solve. "
            "Levers with a workaround yield an alternative plan (at a higher cost); "
            "levers with none are *essential* — approval is unreachable without them. "
            "Both answers are useful to an applicant."
        ),
        nbf.v4.new_code_cell(
            "from treecf import Infeasible\n\n"
            "base = accepted + [Freeze(\"age\"),\n"
            "                   AllowMissing(\"months_since_last_delinq\", delta_miss=2.0)]\n"
            "policy = {\"max_dpd_30d\": \"integer\", \"max_dpd_12m\": \"integer\",\n"
            "          \"n_active_loans\": \"integer\", \"n_loans_total\": \"integer\"}\n"
            "idx = {f: i for i, f in enumerate(names)}\n"
            "levers = sorted(res.changes,\n"
            "                key=lambda f: abs(res.changes[f][1] - res.changes[f][0])\n"
            "                / exp.sigma[idx[f]], reverse=True)[:3]\n"
            "alternatives, essential = [res], []\n"
            "for lever in levers:\n"
            "    alt = Explainer(model, background=X, value_policy=policy,\n"
            "                    constraints=base + [Freeze(lever)])\n"
            "    cand = alt.explain(applicant, seed=0, time_budget_s=30,\n"
            "                       target=Target.probability(range=(0.0, cutoff)))\n"
            "    if isinstance(cand, Infeasible):\n"
            "        essential.append(lever)      # no plan exists without this lever\n"
            "    else:\n"
            "        alternatives.append(cand)\n"
            "print(\"essential levers:\", essential)\n"
            "[(round(a.distance, 2), sorted(a.changes)) for a in alternatives]"
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_counterfactuals\n\nplot_counterfactuals(alternatives);"
        ),
        nbf.v4.new_markdown_cell(
            "**Compare the plans directly**: every alternative's changes on shared feature "
            "axes (one color per plan, gray dots mark the factual values), and the "
            "cost-vs-outcome trade-off — what each plan asks for and what it buys."
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_alternatives, plot_tradeoff\n\n"
            "plot_alternatives(alternatives, explainer=exp);   # deltas in sigma units"
        ),
        nbf.v4.new_code_cell(
            "plot_tradeoff(alternatives, target=Target.probability(range=(0.0, cutoff)));"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Mass-producing counterfactuals for a day's declines\n\n"
            "Score a day's applications, take the declines, and produce (up to) two "
            "recourse plans per applicant in one call — then store the batch and look "
            "plans up later without recomputing."
        ),
        nbf.v4.new_code_cell(
            "import time\n\n"
            "declined = np.flatnonzero(proba > cutoff)[:200]     # today's declines\n"
            "app_ids = [f\"APP-{i:05d}\" for i in declined]\n\n"
            "start = time.perf_counter()\n"
            "batch = exp.explain_batch(\n"
            "    X[declined],\n"
            "    target=Target.probability(range=(0.0, cutoff)),\n"
            "    n_per_example=2,           # counterfactuals per applicant\n"
            "    diversity=\"seeds\",\n"
            "    ids=app_ids,\n"
            "    seed=0,\n"
            ")\n"
            "wall = time.perf_counter() - start\n"
            "feasible = sum(r.feasible for r in batch)\n"
            "print(f\"{len(batch)} plans for {len(declined)} applicants \"\n"
            "      f\"in {wall:.1f}s ({1000 * wall / len(declined):.0f} ms/applicant)\")\n"
            "print(f\"feasible plans: {feasible}\")"
        ),
        nbf.v4.new_code_cell(
            "import pathlib, tempfile\n\n"
            "store_path = pathlib.Path(tempfile.mkdtemp()) / \"counterfactuals_today.json\"\n"
            "batch.save(store_path)                        # store once...\n\n"
            "from treecf import BatchResult\n"
            "stored = BatchResult.load(store_path)\n"
            "stored.for_id(app_ids[0])                     # ...look up any time"
        ),
        nbf.v4.new_code_cell(
            "stored.to_frame().head(6)      # or analyze the whole day as a DataFrame"
        ),
        nbf.v4.new_markdown_cell(
            "### Visualizing the batch\n\n"
            "Four batch-level views: which levers the plans use (and in which direction), "
            "the effort each change costs per plan, cost/sparsity/feasibility at a glance, "
            "and how far each lever actually moves across applicants."
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz_batch import plot_batch_levers, plot_batch_matrix\n\n"
            "plot_batch_levers(batch);"
        ),
        nbf.v4.new_code_cell("plot_batch_matrix(batch, explainer=exp, max_row_labels=15);"),
        nbf.v4.new_code_cell(
            "from treecf.viz_batch import plot_batch_deltas, plot_batch_summary\n\n"
            "plot_batch_summary(batch);"
        ),
        nbf.v4.new_code_cell(
            "plot_batch_deltas(batch, explainer=exp);   # deltas in sigma units"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Coalitions: grouped recourse\n\n"
            "A single plan can mix unrelated levers — income, credit usage, and debt "
            "history in one instruction. **Coalitions** split recourse by what the "
            "applicant controls together: one counterfactual per named feature group, "
            "each allowed to change only its own group. An *infeasible* group is a "
            "finding in itself: that front alone cannot reach the target. The "
            "`\"(all levers)\"` baseline shows what the grouping costs versus the "
            "unrestricted optimum. Opt-in — plain `explain` is unchanged."
        ),
        nbf.v4.new_code_cell(
            "groups = {\n"
            "    \"debt history\": [\"max_dpd_30d\", \"max_dpd_12m\","
            " \"months_since_last_delinq\"],\n"
            "    \"credit usage\": [\"utilization\", \"n_active_loans\","
            " \"n_loans_total\"],\n"
            "    \"income\":       [\"income_monthly\"],\n"
            "}\n"
            "grouped = exp.explain_coalitions(\n"
            "    applicant, target=Target.probability(range=(0.0, cutoff)),\n"
            "    coalitions=groups, include_full=True, seed=0,\n"
            ")\n"
            "{name: (round(out.distance, 2) if hasattr(out, 'distance') else 'infeasible')\n"
            " for name, out in grouped.items()}"
        ),
        nbf.v4.new_code_cell(
            "plot_alternatives(grouped, explainer=exp);   # coalition names label the plans"
        ),
        nbf.v4.new_code_cell(
            "plot_tradeoff(grouped, target=Target.probability(range=(0.0, cutoff)));"
        ),
        nbf.v4.new_markdown_cell(
            "The same mode scales to the whole batch: one record per coalition per "
            "applicant, with the group name in the `coalition` column."
        ),
        nbf.v4.new_code_cell(
            "grouped_batch = exp.explain_batch(\n"
            "    X[declined][:20], target=Target.probability(range=(0.0, cutoff)),\n"
            "    diversity=\"coalitions\", coalitions=groups, include_full=True,\n"
            "    ids=app_ids[:20], seed=0,\n"
            ")\n"
            "grouped_batch.to_frame()[\n"
            "    [\"id\", \"k\", \"coalition\", \"feasible\", \"distance\", \"n_changed\"]\n"
            "].head(8)"
        ),
    ]
    return nb


def no_solver() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Restricted environments: JSON dumps + the Rust genetic engine\n\n"
            "Model-validation and audit hosts often cannot install the training "
            "framework or a solver. treecf parses **JSON dumps** directly, and its "
            "genetic backend runs on a **compiled Rust core bundled in the wheel** — "
            "no ortools, no xgboost, no Python dependencies beyond numpy."
        ),
        nbf.v4.new_code_cell(DATA_CELL.strip()),
        nbf.v4.new_code_cell(TRAIN_CELL.strip()),
        nbf.v4.new_markdown_cell(
            "## Ship the dump, not the framework\n\n(Here we round-trip through a file "
            "in place of the audit host's copy.)"
        ),
        nbf.v4.new_code_cell(
            "import tempfile, pathlib\n\n"
            "dump_path = pathlib.Path(tempfile.mkdtemp()) / \"model.json\"\n"
            "model.get_booster().save_model(str(dump_path))\n\n"
            "from treecf import Explainer, Target, constraint\n\n"
            "exp = Explainer(\n"
            "    str(dump_path),                      # no xgboost import needed\n"
            "    background=X,\n"
            "    constraints=[constraint(\"max_dpd_30d <= max_dpd_12m\")],\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## Solve without ortools\n\nThe genetic engine is feasibility-first and "
            "seed-deterministic. It returns `proof=\"heuristic\"` — it never claims "
            "optimality, and the result is still float-verified against the model "
            "before being returned."
        ),
        nbf.v4.new_code_cell(
            "res = exp.explain(applicant,\n"
            "                  target=Target.probability(range=(0.0, cutoff)),\n"
            "                  backend=\"genetic\", seed=0)\n"
            "res.proof, res.solver_stats[\"backend\"], res.changes"
        ),
        nbf.v4.new_code_cell(
            "float(model.predict_proba(np.nan_to_num(res.x_cf, nan=np.nan)"
            ".reshape(1, -1))[0, 1])  # the native model agrees"
        ),
        nbf.v4.new_markdown_cell(
            "## How fast is the Rust engine?\n\n"
            "`backend=\"python\"` runs the original numpy implementation of the same "
            "algorithm, kept as a reference engine — identical result quality (the two "
            "are held to statistical parity), just slower. On production-sized models "
            "(300 trees, 50 features) the gap is 44–58× — see the Benchmarks page. "
            "On this notebook's small model:"
        ),
        nbf.v4.new_code_cell(
            "import time\n\n"
            "def timed(backend):\n"
            "    start = time.perf_counter()\n"
            "    for seed in range(5):\n"
            "        exp.explain(applicant,\n"
            "                    target=Target.probability(range=(0.0, cutoff)),\n"
            "                    backend=backend, seed=seed)\n"
            "    return (time.perf_counter() - start) / 5\n"
            "\n"
            "rust_s, python_s = timed(\"genetic\"), timed(\"python\")\n"
            "print(f\"rust   {rust_s * 1000:7.1f} ms/solve\")\n"
            "print(f\"python {python_s * 1000:7.1f} ms/solve\")\n"
            "print(f\"speedup {python_s / rust_s:5.1f}x\")"
        ),
    ]
    return nb


def main() -> None:
    for name, build in (
        ("01-quickstart", quickstart),
        ("02-credit-risk-tutorial", tutorial),
        ("03-no-solver-environments", no_solver),
    ):
        nb = build()
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute()
        path = f"{OUT}/{name}.ipynb"
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("executed and wrote", path)


if __name__ == "__main__":
    sys.exit(main())
