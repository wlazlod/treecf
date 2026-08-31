# API reference

Rendered from the numpy-style docstrings, split across four pages (a single
page with every module triggers a third-party rendering pathology):

- [Explainer and results](api/explainer.md): `Explainer`, the result types,
  batch production, regions, and the audit surface.
- [Constraints](api/constraints.md): the constraint objects, the
  `constraint()` mini-language, mining, and plausibility.
- [Targets](api/targets.md): `Target` and its constructors.
- [Visualization](api/viz.md): every plot function in `treecf.viz` and
  `treecf.viz_batch` (extra: `treecf[viz]`).

The public surface is exported flat from `treecf`; its exact extent, and the
promises attached to serialized artifacts, are stated in
[API stability](api-stability.md).
