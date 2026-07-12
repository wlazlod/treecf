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
            "under the approval cutoff?* The CP-SAT backend answers with an optimality "
            "proof; for solver-free environments there is a bundled Rust genetic engine "
            "(see the third notebook)."
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
            "res = exp.explain(applicant, target=Target.probability(range=(0.0, cutoff)))\n"
            "res.proof, res.n_changed, round(res.score_prob, 4)"
        ),
        nbf.v4.new_code_cell("res.changes"),
        nbf.v4.new_markdown_cell("## Visualize the recommendation"),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_changes\n\nplot_changes(res);"
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
            "res = exp.explain(applicant, target=Target.probability(range=(0.0, cutoff)))\n"
            "res.changes"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. The rating ladder\n\nOne compilation, one solve per band: the "
            "increasing cost of each better grade."
        ),
        nbf.v4.new_code_cell(
            "ladder = exp.explain(applicant, target=Target.bands({\n"
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
            "## 4. Diverse alternatives\n\nNo-good cuts produce structurally different "
            "recommendations, in non-decreasing cost order."
        ),
        nbf.v4.new_code_cell(
            "diverse = exp.explain(applicant,\n"
            "                      target=Target.probability(range=(0.0, cutoff)),\n"
            "                      n_counterfactuals=3)\n"
            "[sorted(r.changes) for r in diverse]"
        ),
        nbf.v4.new_code_cell(
            "from treecf.viz import plot_counterfactuals\n\nplot_counterfactuals(diverse);"
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
