# Cells and category blocks

Every treecf backend searches the same finite grid, derived once from the
parsed model. This page names its two halves — numeric *cells* and
categorical *blocks* — because every stronger claim on this site (optimality
proofs, certified infeasibility, certified regions) quantifies over exactly
this grid.

## Numeric features: routing-atomic cells

A tree ensemble is piecewise constant. For one numeric feature, collect
every threshold any tree splits on: those thresholds cut the real line into
**routing-atomic cells** — intervals inside which every tree routes
identically. Moving within a cell changes nothing about the model's output;
only crossing a threshold does. Within a chosen cell there is exactly one
optimal value, the point nearest the factual, so the search over
$\mathbb{R}^p$ collapses to a finite search over cells with no loss.

Cells respect each split's exact comparison as stored — strict and
non-strict thresholds are kept distinct, never normalized into each other —
and a value landing on an open bound is placed one float32 ulp inside it,
because gradient-boosting libraries compare in float32
([the full rule](../how-it-works.md#the-search-space-cells-not-real-numbers)).

## Categorical features: category blocks

A categorical feature has no thresholds and no order, so its grid is built
from set-membership instead: codes that fall on the same side of every split
mentioning the feature are **routing-equivalent**, and the equivalence
classes — *category blocks* — play exactly the role cells play for numeric
features. One representative per block covers every behavior the model can
express; a claim proved over blocks is a claim over all codes.
[Categorical features](categorical.md) develops the consequences for cost,
constraints, and certified regions.

## Why the grid is the search space, and the claim space

The exact backend's `proof="optimal"` means: no cheaper feasible assignment
exists *in this grid* — and because the grid provably contains an optimal
representative of every model behavior, that is optimality over the reals,
not an approximation of it. The same holds for `Infeasible.proof="certified"`
and for [certified regions](certification.md), whose bounds are grown
cell-by-cell and block-by-block. When constraints refine the grid (a `Range`
truncating cells, an `AllowedCategories` shrinking a block's members), every
backend sees the same refined grid, compiled once
([constraints](constraints.md)).
