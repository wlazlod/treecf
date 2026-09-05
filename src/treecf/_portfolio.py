"""The portfolio report: one campaign's recourse picture as an audit artifact.

``portfolio_report`` reads a ``BatchResult`` and produces a strict-JSON dict
— population counts and the proof mix, the recourse burden per segment, the
levers each segment's cheapest plans lean on, and the missing-value
transitions those plans ask for — optionally with disparity ratios against a
reference segment. The dict is the fingerprintable artifact; ``path`` writes
it as JSON, or renders it as a single self-contained HTML document (figures
embedded as PNG data, no external references) or as Markdown with the
figures written beside the file. Rendering needs the ``viz`` extra; the
dict itself never touches matplotlib.
"""

from __future__ import annotations

import base64
import html as html_lib
import io
import json
import math
import os
import string
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from treecf.api import Explainer
    from treecf.batch import BatchRecord, BatchResult

# the sentence every disparity ratio travels with
DISPARITY_FRAMING = (
    "a ratio under one declared cost model and constraint set; which comparison "
    "matters is a modeling choice this report does not make."
)

_FORMATS = ("html", "markdown", "json")


def portfolio_report(
    batch: BatchResult,
    groups: Sequence[object] | None = None,
    *,
    explainer: Explainer | None = None,
    path: str | os.PathLike[str] | None = None,
    format: str = "html",
    title: str | None = None,
    disparity: bool = False,
    reference_group: object | None = None,
    top_levers: int = 5,
    min_group_size: int = 10,
) -> dict[str, object]:
    """Summarize a batch of counterfactuals as an auditable report.

    The returned dict is strict-JSON-serializable (``json.dumps(report,
    allow_nan=False)``; non-finite floats are the strings ``"NaN"``,
    ``"Infinity"``, ``"-Infinity"``, as in certificates) and carries
    ``"portfolio_schema_version": 1``. Burdens and ratios compare costs under
    one declared cost model and constraint set; a difference between segments
    is a finding to investigate, not a fairness verdict.

    Parameters
    ----------
    batch
        The batch to report on.
    groups
        One segment label per input row of the batch, in the batch's row
        order; ``None`` reports a single ``"all"`` segment.
    explainer
        The explainer the batch came from. Adds the model and constraint
        fingerprints, scales numeric lever moves by the explainer's
        normalizers, and names categorical target categories.
    path
        Where to write the report; nothing is written when omitted.
    format
        ``"html"`` (one self-contained file), ``"markdown"`` (figures written
        beside the file, in ``<stem>_figures/``), or ``"json"``. HTML and
        Markdown need the ``viz`` extra.
    title
        The document title; defaults to a generic one.
    disparity
        Add per-segment ratios of median burden and of no-recourse share
        against ``reference_group``, each framed by the same fixed sentence.
    reference_group
        The segment the ratios compare against; required exactly when
        ``disparity`` is set.
    top_levers
        How many dominant levers to list per segment.
    min_group_size
        Segments smaller than this are flagged ``small``.

    Returns
    -------
    The report as a plain ``dict``.

    Raises
    ------
    TreecfError
        If ``groups`` does not have one label per batch row.
    ValueError
        If ``format`` is unknown, ``format="markdown"`` comes without
        ``path``, ``reference_group`` is given without ``disparity`` (or
        missing or unknown with it).
    MissingExtraError
        If an HTML or Markdown render is requested without matplotlib.
    """
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
    if format == "markdown" and path is None:
        raise ValueError('format="markdown" requires path (figures are written next to it)')
    if disparity and reference_group is None:
        raise ValueError("reference_group is required when disparity=True")
    if not disparity and reference_group is not None:
        raise ValueError("reference_group is only valid with disparity=True")

    report = _build(batch, groups, explainer, disparity, reference_group, top_levers,
                    min_group_size)
    if path is None:
        return report
    if format == "json":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, allow_nan=False, indent=1, sort_keys=True)
        return report
    text = _render(report, batch, groups, format, Path(path), title, min_group_size)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return report


# ------------------------------------------------------------ the content ---


def _build(
    batch: BatchResult,
    groups: Sequence[object] | None,
    explainer: Explainer | None,
    disparity: bool,
    reference_group: object | None,
    top_levers: int,
    min_group_size: int,
) -> dict[str, object]:
    from treecf import __version__
    from treecf.audit import _json_float, constraints_fingerprint, ir_fingerprint
    from treecf.viz_batch import _rows_with_burdens, recourse_burden_table

    rows = _rows_with_burdens(batch)
    labels = list(groups) if groups is not None else ["all"] * len(rows)
    table = recourse_burden_table(batch, labels, min_group_size=min_group_size)
    labels = [_label(v) for v in labels]

    fingerprints: dict[str, object] = {}
    if explainer is not None:
        fingerprints["ir"] = ir_fingerprint(explainer.ir)
        fingerprints["constraints"] = constraints_fingerprint(explainer)
    calibrator = next(
        (r.calibrator_fingerprint for r in batch.records if r.calibrator_fingerprint), None
    )
    if calibrator is not None:
        fingerprints["calibrator"] = calibrator

    with_recourse = sum(1 for _, burden, _ in rows if burden is not None)
    certified_no = sum(1 for _, burden, certified in rows if burden is None and certified)
    proof_mix: dict[str, int] = {}
    for record in batch.records:
        proof_mix[record.proof] = proof_mix.get(record.proof, 0) + 1

    cheapest = _cheapest_plans(batch)
    row_group = {row_id: label for (row_id, _, _), label in zip(rows, labels, strict=True)}
    kinds, names_of, sigma_of = _feature_metadata(batch, explainer)
    levers: dict[str, list[dict[str, object]]] = {}
    for entry in table:
        label = _label(entry["group"])
        plans = [
            plan for row_id, plan in cheapest.items() if row_group[row_id] == label
        ]
        levers[str(label)] = _dominant_levers(plans, kinds, names_of, sigma_of, top_levers)

    report: dict[str, object] = {
        "portfolio_schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "treecf_version": __version__,
        "fingerprints": fingerprints,
        "population": {
            "rows": len(rows),
            "records": len(batch.records),
            "with_recourse": with_recourse,
            "certified_no_recourse": certified_no,
            "unproven_no_recourse": len(rows) - with_recourse - certified_no,
        },
        "proof_mix": proof_mix,
        "recourse_burden_table": [
            {
                key: (_json_float(v) if isinstance(v, float) else _label(v))
                for key, v in entry.items()
            }
            for entry in table
        ],
        "dominant_levers": levers,
        "missing_transitions": _missing_transitions(cheapest.values()),
    }
    if disparity:
        report["disparity"] = _disparity(table, reference_group)
    return report


def _label(value: object) -> object:
    """A group label as strict JSON can carry it."""
    from treecf.audit import _json_float

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _json_float(value)
    return str(value)


def _cheapest_plans(batch: BatchResult) -> dict[object, BatchRecord]:
    """Per row id, the feasible record with the smallest distance (first on ties)."""
    best: dict[object, BatchRecord] = {}
    for record in batch.records:
        if not record.feasible or record.distance is None:
            continue
        current = best.get(record.id)
        if current is None or (
            current.distance is not None and record.distance < current.distance
        ):
            best[record.id] = record
    return best


def _feature_metadata(
    batch: BatchResult, explainer: Explainer | None
) -> tuple[dict[str, str], dict[str, tuple[str, ...] | None], dict[str, float]]:
    """Per feature name: its kind, its category names, and its normalizer —
    all from the explainer when there is one, numeric-by-default otherwise."""
    kinds: dict[str, str] = dict.fromkeys(batch.feature_names, "numeric")
    names_of: dict[str, tuple[str, ...] | None] = dict.fromkeys(batch.feature_names, None)
    sigma_of: dict[str, float] = {}
    if explainer is None:
        return kinds, names_of, sigma_of
    for j, name in enumerate(explainer.ir.feature_names):
        info = explainer.ir.categorical.get(j)
        if info is not None:
            kinds[name] = "categorical"
            names_of[name] = info.categories
        sigma_of[name] = float(explainer.sigma[j])
    return kinds, names_of, sigma_of


def _dominant_levers(
    plans: Sequence[BatchRecord],
    kinds: Mapping[str, str],
    names_of: Mapping[str, tuple[str, ...] | None],
    sigma_of: Mapping[str, float],
    top_levers: int,
) -> list[dict[str, object]]:
    from treecf.audit import _json_float

    counts: dict[str, int] = {}
    deltas: dict[str, list[float]] = {}
    targets: dict[str, dict[str, int]] = {}
    for plan in plans:
        for feature, (src, dst) in plan.changes.items():
            counts[feature] = counts.get(feature, 0) + 1
            if kinds.get(feature) == "categorical":
                names = names_of.get(feature)
                code = int(dst) if not math.isnan(dst) else None
                label = (
                    names[code]
                    if code is not None and names is not None and 0 <= code < len(names)
                    else ("missing" if code is None else str(code))
                )
                targets.setdefault(feature, {})[label] = targets.get(feature, {}).get(label, 0) + 1
            elif not math.isnan(src) and not math.isnan(dst):
                deltas.setdefault(feature, []).append(abs(dst - src))
    ranked = sorted(counts, key=lambda f: (-counts[f], f))[:top_levers]
    out: list[dict[str, object]] = []
    for feature in ranked:
        entry: dict[str, object] = {
            "feature": feature,
            "count": counts[feature],
            "share": counts[feature] / len(plans) if plans else math.nan,
            "kind": kinds.get(feature, "numeric"),
        }
        if entry["kind"] == "categorical":
            by_count = sorted(targets.get(feature, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            entry["top_targets"] = [[label, n] for label, n in by_count[:top_levers]]
        else:
            moves = deltas.get(feature, [])
            median = float(np.median(moves)) if moves else math.nan
            entry["median_abs_delta"] = _json_float(median)
            if feature in sigma_of:
                entry["median_abs_delta_sigma"] = _json_float(
                    median / sigma_of[feature] if sigma_of[feature] else math.nan
                )
        out.append(entry)
    return out


def _missing_transitions(plans: Sequence[BatchRecord] | Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for plan in plans:
        for feature, (src, dst) in plan.changes.items():
            provide = math.isnan(src) and not math.isnan(dst)
            drop = not math.isnan(src) and math.isnan(dst)
            if not (provide or drop):
                continue
            entry = out.setdefault(feature, {"provide": 0, "drop": 0})
            entry["provide" if provide else "drop"] += 1
    return dict(sorted(out.items()))


def _disparity(
    table: Sequence[Mapping[str, object]], reference_group: object
) -> dict[str, object]:
    from treecf.audit import _json_float

    reference = next((e for e in table if _label(e["group"]) == _label(reference_group)), None)
    if reference is None:
        raise ValueError(f"reference_group {reference_group!r} is not one of the groups")

    def ratio(value: float, base: float) -> float | str | None:
        if math.isnan(value) or math.isnan(base) or base == 0.0:
            return None
        return _json_float(value / base)

    ref_median = float(reference["median_burden"])  # type: ignore[arg-type]
    ref_no = 1.0 - float(reference["recourse_share"])  # type: ignore[arg-type]
    groups_out: list[dict[str, object]] = []
    for entry in table:
        groups_out.append(
            {
                "group": _label(entry["group"]),
                "median_burden_ratio": ratio(float(entry["median_burden"]), ref_median),  # type: ignore[arg-type]
                "no_recourse_share_ratio": ratio(
                    1.0 - float(entry["recourse_share"]), ref_no  # type: ignore[arg-type]
                ),
                "framing": DISPARITY_FRAMING,
            }
        )
    return {
        "reference_group": _label(reference_group),
        "framing": DISPARITY_FRAMING,
        "groups": groups_out,
    }


# ------------------------------------------------------------ the renderer ---

_HTML_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:62rem;margin:2rem auto;"
    "padding:0 1rem;color:#222;line-height:1.45}"
    "table{border-collapse:collapse;margin:.8rem 0}"
    "th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:left;font-size:.92em}"
    "img{max-width:100%;display:block;margin:.6rem 0}"
    "h1,h2,h3{font-weight:600;margin-top:1.6rem}"
    ".meta{color:#666;font-size:.9em}"
    ".note{background:#f5f5f5;border-left:3px solid #999;padding:.5rem .8rem;margin:.8rem 0}"
)

_HTML_TEMPLATE = string.Template(
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<title>$title</title>\n<style>$style</style>\n</head>\n<body>\n"
    "<h1>$title</h1>\n<p class=\"meta\">$timestamp (treecf $version)</p>\n"
    "$fingerprints\n$sections\n</body>\n</html>\n"
)

_MD_TEMPLATE = string.Template(
    "# $title\n\n$timestamp (treecf $version)\n\n$fingerprints\n$sections"
)


class _FigureSink:
    """Turns a matplotlib drawing into what the document needs: a data URI
    for HTML, a file beside the document for Markdown."""

    def __init__(self, fmt: str, path: Path) -> None:
        self.fmt = fmt
        self.fig_dir_name = f"{path.stem}_figures"
        self.fig_dir = path.parent / self.fig_dir_name

    def figure(self, draw: Callable[[], Any], name: str) -> str:
        from treecf.viz import _import_pyplot

        plt = _import_pyplot()
        artist = draw()
        if isinstance(artist, list | tuple | np.ndarray):
            artist = np.asarray(artist).ravel()[0]  # an array of axes: one figure
        fig = artist if hasattr(artist, "savefig") else artist.figure
        if self.fmt == "html":
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f'<img src="data:image/png;base64,{encoded}" alt="{_escape_html(name)}">'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.fig_dir / f"{name}.png", format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return f"![{name}]({self.fig_dir_name}/{name}.png)"


def _escape_html(text: object) -> str:
    return html_lib.escape(str(text), quote=True)


def _escape_md_cell(text: object) -> str:
    return str(text).replace("|", "\\|")


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "—" if math.isnan(value) else f"{value:.3g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _table(fmt: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if fmt == "html":
        head = "".join(f"<th>{_escape_html(h)}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_escape_html(_fmt(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    lines = [
        "| " + " | ".join(_escape_md_cell(h) for h in headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md_cell(_fmt(c)) for c in row) + " |")
    return "\n".join(lines)


def _kv(fmt: str, pairs: Sequence[tuple[str, object]]) -> str:
    if fmt == "html":
        items = "".join(
            f"<li><b>{_escape_html(k)}:</b> {_escape_html(_fmt(v))}</li>" for k, v in pairs
        )
        return f"<ul>{items}</ul>"
    return "\n".join(f"- **{k}:** {_fmt(v)}" for k, v in pairs)


def _note(fmt: str, text: str) -> str:
    if fmt == "html":
        return f'<p class="note">{_escape_html(text)}</p>'
    return f"> {text}"


def _subheading(fmt: str, text: str) -> str:
    return f"<h3>{_escape_html(text)}</h3>" if fmt == "html" else f"### {text}"


def _section(fmt: str, title: str, body: str) -> str:
    if not body.strip():
        return ""
    if fmt == "html":
        return f"<section>\n<h2>{_escape_html(title)}</h2>\n{body}\n</section>\n"
    return f"## {title}\n\n{body}\n\n"


def _render(
    report: Mapping[str, Any],
    batch: BatchResult,
    groups: Sequence[object] | None,
    fmt: str,
    path: Path,
    title: str | None,
    min_group_size: int,
) -> str:
    from treecf.viz import _import_pyplot
    from treecf.viz_batch import plot_recourse_burden

    plt = _import_pyplot()
    sink = _FigureSink(fmt, path)
    page_title = title if title is not None else "Portfolio recourse report"
    labels = list(groups) if groups is not None else ["all"] * report["population"]["rows"]

    # -- header: fingerprints, population, proof mix --------------------------
    fingerprints = report["fingerprints"]
    fingerprint_block = (
        _kv(fmt, [(k, v) for k, v in fingerprints.items()]) if fingerprints else ""
    )
    population = report["population"]
    proof_mix = report["proof_mix"]

    def draw_proof_mix() -> Any:
        _, ax = plt.subplots(figsize=(5, 2.6))
        names = list(proof_mix)
        ax.bar(names, [proof_mix[n] for n in names], color="C0")
        ax.set_ylabel("records")
        ax.set_title("proof mix")
        return ax

    header_body = "\n".join(
        [
            _kv(fmt, [(k.replace("_", " "), v) for k, v in population.items()]),
            _table(fmt, ["proof", "records"], [[k, v] for k, v in proof_mix.items()]),
            sink.figure(draw_proof_mix, "proof_mix"),
        ]
    )

    # -- burden --------------------------------------------------------------
    burden_rows = report["recourse_burden_table"]
    burden_headers = [
        "group", "n", "recourse share", "certified no", "unproven no", "median burden",
        "mean burden", "p90 burden", "small",
    ]
    burden_table = _table(
        fmt,
        burden_headers,
        [
            [
                r["group"], r["n"], r["recourse_share"], r["certified_no_share"],
                r["unproven_no_share"], r["median_burden"], r["mean_burden"],
                r["p90_burden"], r["small"],
            ]
            for r in burden_rows
        ],
    )
    burden_body = "\n".join(
        [
            burden_table,
            sink.figure(
                lambda: plot_recourse_burden(batch, labels, min_group_size=min_group_size),
                "recourse_burden",
            ),
            _note(
                fmt,
                "Burden compares costs under one declared cost model and constraint set; "
                "a difference between segments is a finding to investigate, not a verdict.",
            ),
        ]
    )

    # -- levers --------------------------------------------------------------
    levers = report["dominant_levers"]
    lever_parts: list[str] = []
    for group, entries in levers.items():
        rows = []
        for e in entries:
            detail: object
            if e.get("kind") == "categorical":
                detail = ", ".join(f"{label} ×{n}" for label, n in e.get("top_targets", []))
            else:
                detail = e.get("median_abs_delta_sigma", e.get("median_abs_delta"))
            rows.append([e["feature"], e["count"], e["share"], e["kind"], detail])
        lever_parts.append(_subheading(fmt, f"segment {group}"))
        lever_parts.append(
            _table(fmt, ["lever", "plans", "share", "kind", "move (median |Δ|/σ) or targets"], rows)
        )

    def draw_levers() -> Any:
        n_groups = max(len(levers), 1)
        fig, axes = plt.subplots(n_groups, 1, figsize=(6, 1.2 + 1.6 * n_groups), squeeze=False)
        for ax, (group, entries) in zip(axes[:, 0], levers.items(), strict=False):
            names = [e["feature"] for e in entries][::-1]
            counts = [e["count"] for e in entries][::-1]
            ax.barh(names, counts, color="C0")
            ax.set_title(f"dominant levers — {group}", fontsize=9)
            ax.set_xlabel("plans using the lever")
        fig.tight_layout()
        return fig

    lever_body = "\n".join([*lever_parts, sink.figure(draw_levers, "dominant_levers")])

    # -- missing transitions --------------------------------------------------
    missing = report["missing_transitions"]
    missing_rows = [[f, v["provide"], v["drop"]] for f, v in missing.items()]
    missing_body = _table(fmt, ["feature", "provide", "drop"], missing_rows) if missing else ""

    # -- disparity -----------------------------------------------------------
    disparity_body = ""
    if "disparity" in report:
        block = report["disparity"]
        disparity_body = "\n".join(
            [
                _kv(fmt, [("reference group", block["reference_group"])]),
                _table(
                    fmt,
                    ["group", "median burden ratio", "no-recourse share ratio"],
                    [
                        [
                            g["group"],
                            "—" if g["median_burden_ratio"] is None else g["median_burden_ratio"],
                            "—"
                            if g["no_recourse_share_ratio"] is None
                            else g["no_recourse_share_ratio"],
                        ]
                        for g in block["groups"]
                    ],
                ),
                _note(fmt, block["framing"]),
            ]
        )

    sections = "".join(
        [
            _section(fmt, "Population and proofs", header_body),
            _section(fmt, "Recourse burden by segment", burden_body),
            _section(fmt, "Dominant levers", lever_body),
            _section(fmt, "Missing-value transitions", missing_body),
            _section(fmt, "Disparity ratios", disparity_body),
        ]
    )
    template = _HTML_TEMPLATE if fmt == "html" else _MD_TEMPLATE
    return template.substitute(
        title=_escape_html(page_title) if fmt == "html" else page_title,
        style=_HTML_STYLE,
        timestamp=report["created_utc"],
        version=report["treecf_version"],
        fingerprints=fingerprint_block,
        sections=sections,
    )
