"""Flat-array (CSR) serialization of compiled constraints — cross-language contract.

Order matters where the Python semantics are order-dependent (repair applies
linears, implications, and one-hot groups in declaration order); the CSR
encodings preserve it. Bounds constraints (freeze/range/equals/monotone) are
order-insensitive (max/min intersection).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.objects import Equals, Freeze, Monotone, Range

_OP_CODE = {"<=": 0, ">=": 1, "==": 2}
_POLICY_CODE = {"satisfied": 0, "forbid_missing": 1, "violated": 1}


def flatten_constraints(compiled: CompiledConstraints) -> dict[str, Any]:
    index = {name: j for j, name in enumerate(compiled.feature_names)}

    freeze: list[int] = []
    range_idx: list[int] = []
    range_lo: list[float] = []
    range_hi: list[float] = []
    equals_idx: list[int] = []
    equals_val: list[float] = []
    mono_idx: list[int] = []
    mono_dir: list[int] = []
    for c in compiled.constraints:
        if isinstance(c, Freeze):
            freeze.append(index[c.feature])
        elif isinstance(c, Range):
            range_idx.append(index[c.feature])
            range_lo.append(c.lo)
            range_hi.append(c.hi)
        elif isinstance(c, Equals):
            equals_idx.append(index[c.feature])
            equals_val.append(c.value)
        elif isinstance(c, Monotone):
            mono_idx.append(index[c.feature])
            mono_dir.append(1 if c.direction == "increase" else -1)

    lin_offsets = [0]
    lin_indices: list[int] = []
    lin_coefs: list[float] = []
    lin_op: list[int] = []
    lin_rhs: list[float] = []
    lin_policy: list[int] = []
    for lin in compiled.linears:
        lin_indices.extend(lin.indices)
        lin_coefs.extend(lin.coefs)
        lin_offsets.append(len(lin_indices))
        lin_op.append(_OP_CODE[lin.op])
        lin_rhs.append(lin.rhs)
        lin_policy.append(_POLICY_CODE[lin.missing_policy])

    oh_offsets = [0]
    oh_indices: list[int] = []
    for group in compiled.onehot_groups:
        oh_indices.extend(group)
        oh_offsets.append(len(oh_indices))

    am_idx = sorted(compiled.allow_missing)
    return {
        "n_features": len(compiled.feature_names),
        "freeze": np.asarray(freeze, dtype=np.uint32),
        "range_idx": np.asarray(range_idx, dtype=np.uint32),
        "range_lo": np.asarray(range_lo, dtype=np.float64),
        "range_hi": np.asarray(range_hi, dtype=np.float64),
        "equals_idx": np.asarray(equals_idx, dtype=np.uint32),
        "equals_val": np.asarray(equals_val, dtype=np.float64),
        "mono_idx": np.asarray(mono_idx, dtype=np.uint32),
        "mono_dir": np.asarray(mono_dir, dtype=np.int8),
        "lin_offsets": np.asarray(lin_offsets, dtype=np.uint32),
        "lin_indices": np.asarray(lin_indices, dtype=np.uint32),
        "lin_coefs": np.asarray(lin_coefs, dtype=np.float64),
        "lin_op": np.asarray(lin_op, dtype=np.uint8),
        "lin_rhs": np.asarray(lin_rhs, dtype=np.float64),
        "lin_policy": np.asarray(lin_policy, dtype=np.uint8),
        "imp_cond_idx": np.asarray(
            [i.cond_index for i in compiled.implications], dtype=np.uint32
        ),
        "imp_cond_val": np.asarray(
            [i.cond_value for i in compiled.implications], dtype=np.float64
        ),
        "imp_cons_idx": np.asarray(
            [i.cons_index for i in compiled.implications], dtype=np.uint32
        ),
        "imp_cons_val": np.asarray(
            [i.cons_value for i in compiled.implications], dtype=np.float64
        ),
        "oh_offsets": np.asarray(oh_offsets, dtype=np.uint32),
        "oh_indices": np.asarray(oh_indices, dtype=np.uint32),
        "am_idx": np.asarray(am_idx, dtype=np.uint32),
        "am_to": np.asarray(
            [compiled.allow_missing[j][0] for j in am_idx], dtype=np.float64
        ),
        "am_from": np.asarray(
            [compiled.allow_missing[j][1] for j in am_idx], dtype=np.float64
        ),
    }
