"""Shared cross-language vectors — the Python half.

Same ../vectors.json drives js/vectors.test.js. Keeping one expected-value
file means a divergence between the two ports fails a test instead of
waiting to be noticed during the next hand-port (which is how the 0.1.1
atan sign bug surfaced).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from emltree.compiler import compile_formula
from emltree.evaluator import evaluate

_DOC = json.loads((Path(__file__).parent.parent / "vectors.json").read_text())
_TOL = _DOC["tolerance"]


@pytest.mark.parametrize(
    "case", _DOC["cases"], ids=[f"{c['formula']} @ {c['bindings']}" for c in _DOC["cases"]]
)
def test_vector(case):
    tree = compile_formula(case["formula"], variables=list(case["bindings"]))
    got = complex(evaluate(tree, case["bindings"]))
    expected = case["expected"]
    scale = max(1.0, abs(expected))
    assert abs(got - expected) < _TOL * scale, f"{case['formula']}: {got} != {expected}"
