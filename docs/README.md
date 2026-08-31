# Writing docs

Notes for anyone editing the Markdown under `docs/`. This is not part of the
built site (mkdocs never lists it in `nav`); it is here for the same reason
a package keeps a `CONTRIBUTING.md`: instructions for the people editing
the source, not for the reader of the output.

## The snippet convention

Every fenced ` ```python ` block on a docs page is executed by
`tests/test_docs_snippets.py` (marked `slow`), page by page, in one
namespace pre-seeded with a fixed vocabulary of solved credit-shaped data.
That turns the fragment style (showing a call without repeating a full
model build every time) from a rot risk into a tested contract. Two rules
keep it that way:

1. **Every block imports what it names.** A block runs after the ones
   above it on the same page (one namespace per page), so it may reuse a
   name an earlier block on the same page defined, but never a name
   assumed from nowhere.
2. **Elided data uses the fixed vocabulary, named in a first-line
   comment.** When a block needs a model, a factual, or a solved result
   without re-deriving them, it draws only from this set, and says so:

   ```python
   # docs: no-run, illustrates the convention, not a runnable snippet
   # exp, x, target: the docs explainer, one rejected applicant, the target
   ...
   ```

   The vocabulary, and what the test harness seeds it with, is exactly:

   | Name | What it is |
   |---|---|
   | `exp` | An `Explainer` over `tests/fixtures/docs_model.json` (a committed LightGBM dump: `income`, `utilization`, `dpd_12m`, `tenure_months`, and a native categorical `occupation`), background `X_bg`, `categories={"occupation": OCCUPATIONS}` |
   | `X_bg` | The background matrix (400 rows, fixed recipe in the harness) |
   | `x` | One rejected factual row (`X_bg[1]`) |
   | `target` | `Target.probability(range=(0.0, 0.05))` |
   | `res` | `exp.explain(x, target=target, seed=0)` — a feasible `Counterfactual` |
   | `batch` | `exp.explain_batch(X_bg[:20], target=target)` |
   | `cal` | A stub calibrator satisfying the duck protocol (no probcal import) |
   | `OCCUPATIONS` | `("student", "clerk", "manager", "retired")` |
   | `np` | numpy |

   A page that needs a name outside this set either defines it locally in
   the block (e.g. a constraint list built inline) or, if the vocabulary
   should grow, extends the table above in this one place; it never
   invents a page-local convention.

The harness skips a block that is not meant to run: REPL transcripts
(first non-blank line starting with `>>>`), `--8<--` includes, and genuine
pseudo-code that cannot be made to run without a heavier fixture than the
vocabulary supports. It recognizes the first two cases on its own; for
pseudo-code, mark it explicitly and say why:

```python
# docs: no-run — model.json stands in for your own trained model file
```

Use `# docs: no-run` sparingly: it opts a block out of the tested
contract, so prefer making the block actually run against the vocabulary
whenever that is feasible.

## Pages that need an optional extra

CI runs the harness in the full dev environment, but a page whose blocks
genuinely need a training library or probcal declares it once, anywhere in
the file:

`<!-- docs: requires <package> -->`, with the package's import name in
place of `<package>`.

The harness then `importorskip`s each named package and skips the whole
page where it is missing. Declare it only when the *page* is about that
integration (`guide/probcal.md`, a library-specific parser section); when
one block on an otherwise core page reaches for an extra, mark that block
`# docs: no-run` instead, so the rest of the page stays under test.

The marker is matched against the whole page, code fences included, so
this note deliberately writes it with a `<package>` placeholder: a literal
example marker anywhere on a page, even inside a fenced block showing the
syntax, would make the harness skip that page for real.

## Regenerating figures

`docs/scripts/generate_figures.py` writes every PNG under
`docs/concepts/img/` and `docs/guide/img/` from the docs model. It does
not run at build time; that is deliberate.

The PNGs are **not** byte-reproducible across matplotlib versions or
platforms: re-running the generator on a different machine rewrites files
whose content is unchanged. Commit a regenerated figure only when the
picture actually changed; `git checkout` the rest rather than adding a
diff nobody can review.
