"""Numerical tests: compile formulas and compare to sympy's evaluation."""
from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import sympy as sp

from emltree.compiler import compile_formula
from emltree.evaluator import evaluate


def _equal(a: complex, b: complex, tol: float = 1e-9) -> bool:
    scale = max(1.0, abs(b))
    return abs(a - b) < tol * scale


def _roundtrip(expression: str, bindings: dict[str, complex]):
    tree = compile_formula(expression, variables=list(bindings.keys()))
    got = complex(evaluate(tree, bindings))
    expr = sp.sympify(expression, locals={k: sp.Symbol(k) for k in bindings})
    expected = complex(expr.evalf(subs=bindings))
    return got, expected


@pytest.mark.parametrize(
    "formula, bindings",
    [
        ("x", {"x": 2.5}),
        ("x + y", {"x": 1.5, "y": 0.7}),
        ("x - y", {"x": 4.0, "y": 1.25}),
        ("x * y", {"x": 3.0, "y": 2.0}),
        ("x / y", {"x": 7.0, "y": 4.0}),
        ("1/x", {"x": 0.8}),
        ("-x", {"x": 3.14}),
        ("exp(x)", {"x": 1.2}),
        ("log(x)", {"x": 3.0}),
        ("sqrt(x)", {"x": 5.0}),
        ("x**3", {"x": 1.3}),
        ("x**y", {"x": 2.0, "y": 3.5}),
        ("exp(x) - log(y)", {"x": 0.3, "y": 2.5}),
    ],
)
def test_algebra(formula, bindings):
    got, expected = _roundtrip(formula, bindings)
    assert _equal(got, expected), f"{formula}: got {got}, expected {expected}"


@pytest.mark.parametrize(
    "formula, bindings",
    [
        ("sinh(x)", {"x": 0.7}),
        ("cosh(x)", {"x": 0.7}),
        ("tanh(x)", {"x": 0.7}),
        ("asinh(x)", {"x": 0.4}),
        ("acosh(x)", {"x": 1.8}),
        ("atanh(x)", {"x": 0.3}),
    ],
)
def test_hyperbolic(formula, bindings):
    got, expected = _roundtrip(formula, bindings)
    assert _equal(got, expected, tol=1e-8)


@pytest.mark.parametrize(
    "formula, bindings",
    [
        ("sin(x)", {"x": 0.9}),
        ("cos(x)", {"x": 0.9}),
        ("tan(x)", {"x": 0.4}),
        ("sin(x)**2 + cos(x)**2", {"x": 1.37}),
    ],
)
def test_trig(formula, bindings):
    got, expected = _roundtrip(formula, bindings)
    assert _equal(got, expected, tol=1e-7)


def test_constants():
    tree_e = compile_formula("E")
    tree_pi = compile_formula("pi")
    e_val = complex(evaluate(tree_e))
    pi_val = complex(evaluate(tree_pi))
    assert _equal(e_val, complex(math.e), tol=1e-12)
    assert _equal(pi_val, complex(math.pi), tol=1e-7)


def test_paper_ln_identity():
    """Verify paper formula (5):  ln(x) = eml(1, eml(eml(1, x), 1))."""
    from emltree.primitives import ln_
    from emltree.core import one, var

    tree = ln_(var("x"))
    assert tree.to_nested() == "eml(1, eml(eml(1, x), 1))"
    # depth 3, 4 leaves (three 1s and x)
    assert tree.leaf_count() == 4
    assert tree.depth() == 3


def test_paper_exp_identity():
    from emltree.primitives import exp_
    from emltree.core import var

    tree = exp_(var("x"))
    assert tree.to_nested() == "eml(x, 1)"


def test_rpn_encoding():
    """paper: RPN for ln is  1 1 x E 1 E E"""
    from emltree.primitives import ln_
    from emltree.core import var

    tree = ln_(var("x"))
    assert tree.to_rpn() == "1 1 x E 1 E E"
