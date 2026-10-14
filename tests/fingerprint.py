"""Canonical fingerprint of a CP-SAT model - used to prove that a refactor did not change the equations.

The canonical form does not depend on the order in which variables and constraints were created:
variables are referred to by name (with their domain), linear terms are merged and sorted, and the
constraints are compared as a sorted multiset. Two models with the same fingerprint contain exactly
the same variables, constraints and objective.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model


def to_pb(model: cp_model.CpModel) -> cp_model_pb2.CpModelProto:
    """Model as a standard protobuf message (newer OR-Tools return a pybind wrapper from Proto())."""
    proto = model.Proto()
    if isinstance(proto, cp_model_pb2.CpModelProto):
        return proto
    fd, path = tempfile.mkstemp(suffix=".pb")
    os.close(fd)
    try:
        model.export_to_file(path)
        return cp_model_pb2.CpModelProto.FromString(Path(path).read_bytes())
    finally:
        os.unlink(path)


def _var_labels(proto: cp_model_pb2.CpModelProto) -> list[str]:
    return [f"{v.name}{list(v.domain)}" for v in proto.variables]


def _literal(labels: list[str], ref: int) -> str:
    return labels[ref] if ref >= 0 else f"not({labels[-ref - 1]})"


def _linear(labels: list[str], msg: dict) -> dict:
    terms: dict[str, int] = {}
    for var, coeff in zip(msg.get("vars", []), msg.get("coeffs", [])):
        key = _literal(labels, var)
        terms[key] = terms.get(key, 0) + int(coeff)
    out = {"terms": sorted((k, c) for k, c in terms.items() if c)}
    for key, value in msg.items():
        if key not in ("vars", "coeffs"):
            out[key] = value
    return out


def _canonical(labels: list[str], node):
    """Replace variable indices by labels and make order-free parts of the message sorted."""
    if isinstance(node, dict):
        if "vars" in node and "coeffs" in node:
            return {k: _canonical(labels, v) for k, v in _linear(labels, node).items()}
        out = {}
        for key, value in node.items():
            if key in ("literals", "enforcementLiteral"):
                out[key] = sorted(_literal(labels, int(v)) for v in value)
            elif key == "vars":
                out[key] = sorted(_literal(labels, int(v)) for v in value)
            elif key == "exprs":
                out[key] = sorted(json.dumps(_canonical(labels, e), sort_keys=True) for e in value)
            else:
                out[key] = _canonical(labels, value)
        return out
    if isinstance(node, list):
        return [_canonical(labels, v) for v in node]
    return node


def canonical_model(model: cp_model.CpModel, proto: cp_model_pb2.CpModelProto | None = None) -> dict:
    proto = proto if proto is not None else to_pb(model)
    labels = _var_labels(proto)
    constraints = sorted(
        json.dumps(_canonical(labels, MessageToDict(c)), sort_keys=True) for c in proto.constraints
    )
    objective = None
    if proto.HasField("objective"):
        objective = json.dumps(_canonical(labels, MessageToDict(proto.objective)), sort_keys=True)
    return {"variables": sorted(labels), "constraints": constraints, "objective": objective}


def fingerprint(model: cp_model.CpModel) -> dict:
    proto = to_pb(model)
    proto.ClearField("solution_hint")   # warm-start hints are not part of the equations
    canon = canonical_model(model, proto)
    digest = hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()
    types: dict[str, int] = {}
    for c in proto.constraints:
        kind = c.WhichOneof("constraint")
        types[kind] = types.get(kind, 0) + 1
    return {
        "sha256": digest,
        # Also identical when the variables and constraints are created in the same order.
        "raw_sha256": hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest(),
        "num_variables": len(canon["variables"]),
        "num_constraints": len(canon["constraints"]),
        "constraint_types": dict(sorted(types.items())),
        "has_objective": canon["objective"] is not None,
    }


def dump_canonical(model: cp_model.CpModel, path: Path) -> None:
    """Write the canonical model (one item per line) so two versions can be compared with diff."""
    canon = canonical_model(model)
    lines = [f"VAR {v}" for v in canon["variables"]] + [f"CON {c}" for c in canon["constraints"]]
    lines.append(f"OBJ {canon['objective']}")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


class RecordingSolver:
    """Stand-in for cp_model.CpSolver: does not solve, records the model at every Solve() call.

    Every expression evaluates to 0, so the lexicographic phases fix their objective to 0. This is
    enough to capture which objective each phase minimises and which constraints it adds.
    """

    calls: list[dict] = []

    def __init__(self):
        from ortools.sat import sat_parameters_pb2
        self.parameters = sat_parameters_pb2.SatParameters()

    def Solve(self, model, *args, **kwargs):  # noqa: N802 - CP-SAT API
        RecordingSolver.calls.append(fingerprint(model))
        return cp_model.FEASIBLE

    def Value(self, expr):  # noqa: N802
        return 0

    def BooleanValue(self, lit):  # noqa: N802
        return False

    def ObjectiveValue(self):  # noqa: N802
        return 0.0

    def BestObjectiveBound(self):  # noqa: N802
        return 0.0

    def StatusName(self, status=None):  # noqa: N802
        return "FEASIBLE"

    def NumConflicts(self):  # noqa: N802
        return 0

    def NumBranches(self):  # noqa: N802
        return 0

    def WallTime(self):  # noqa: N802
        return 0.0

    def ResponseProto(self):  # noqa: N802
        class _Response:
            solution: list = []
        return _Response()
