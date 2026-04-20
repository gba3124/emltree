"""Compile a sympy expression (or formula string) into an EML tree.

The strategy: dispatch on the sympy node type, recurse on children, then
glue the results with the matching builder from `primitives`.

Unknown nodes raise `EMLCompileError` with a clear message.
"""
from __future__ import annotations

from fractions import Fraction

import sympy as sp

from . import primitives as P
from .core import EMLNode, var


class EMLCompileError(ValueError):
    pass


def compile_formula(expression: str, variables: list[str] | None = None) -> EMLNode:
    """Parse a string formula with sympy, then compile it."""
    local = {name: sp.Symbol(name) for name in (variables or [])}
    expr = sp.sympify(expression, locals=local, convert_xor=True)
    return compile_sympy(expr)


def compile_sympy(expr: sp.Basic) -> EMLNode:
    return _compile(sp.sympify(expr))


# ---------------------------------------------------------------------------

def _compile(expr: sp.Basic) -> EMLNode:
    # Integer / Rational / Float
    if isinstance(expr, sp.Integer):
        return P.integer(int(expr))
    if isinstance(expr, sp.Rational):
        return P.rational(int(expr.p), int(expr.q))
    if isinstance(expr, sp.Float):
        # Best-effort: snap to a Fraction with reasonable denominator
        frac = Fraction(float(expr)).limit_denominator(10**9)
        return P.rational(frac.numerator, frac.denominator)

    # Named constants
    if expr is sp.S.Exp1:
        return P.e_const()
    if expr is sp.S.Pi:
        return P.pi_const()
    if expr is sp.S.ImaginaryUnit:
        return P.i_const()
    if expr is sp.S.NegativeOne:
        return P.integer(-1)
    if expr is sp.S.One:
        return P.one() if False else P.integer(1)
    if expr is sp.S.Zero:
        return P.zero()
    if expr is sp.S.Half:
        return P.half()

    # Symbol
    if isinstance(expr, sp.Symbol):
        return var(expr.name)

    # n-ary add/mul flattened by sympy
    if isinstance(expr, sp.Add):
        args = [_compile(a) for a in expr.args]
        return _fold(args, P.add_)
    if isinstance(expr, sp.Mul):
        args = [_compile(a) for a in expr.args]
        return _fold(args, P.mul_)

    # Power
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        # Special-case integer exponents where it helps readability
        if exponent == sp.S.NegativeOne:
            return P.inv_(_compile(base))
        if exponent == sp.S.Half:
            return P.sqrt_(_compile(base))
        return P.pow_(_compile(base), _compile(exponent))

    # Unary-ish: exp, log/ln, sqrt, trig, hyperbolic, sigmoid
    func = type(expr).__name__
    handler = _UNARY.get(func)
    if handler is not None and len(expr.args) == 1:
        return handler(_compile(expr.args[0]))

    # log(x, base)  -> two args
    if isinstance(expr, sp.log) and len(expr.args) == 2:
        x_arg, base_arg = expr.args
        return P.log_base(_compile(x_arg), _compile(base_arg))

    raise EMLCompileError(
        f"No EML expansion registered for {type(expr).__name__}: {expr}"
    )


def _fold(args: list[EMLNode], op) -> EMLNode:
    if not args:
        raise EMLCompileError("_fold called with no args")
    acc = args[0]
    for nxt in args[1:]:
        acc = op(acc, nxt)
    return acc


_UNARY: dict[str, callable] = {
    "exp": P.exp_,
    "log": P.ln_,      # sympy's log/ln both resolve to sympy.log
    "ln": P.ln_,
    "sqrt": P.sqrt_,
    "sin": P.sin_,
    "cos": P.cos_,
    "tan": P.tan_,
    "asin": P.asin_,
    "acos": P.acos_,
    "atan": P.atan_,
    "sinh": P.sinh_,
    "cosh": P.cosh_,
    "tanh": P.tanh_,
    "asinh": P.asinh_,
    "acosh": P.acosh_,
    "atanh": P.atanh_,
}
