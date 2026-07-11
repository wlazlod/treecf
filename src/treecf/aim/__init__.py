"""Abstract intermediate model: cells, variables, linear constraints (spec §5)."""

from treecf.aim.cells import Cell, build_cells, cell_index, feature_cells

__all__ = ["Cell", "build_cells", "cell_index", "feature_cells"]
