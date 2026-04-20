"""EML expansions for the standard scientific-calculator primitives.

Each builder takes EML trees for its arguments and returns an EML tree.
All reductions bottom out in `one()` and `eml(...)`.

Canonical identities (Odrzywolek, arXiv:2603.21852):

    exp(x) = eml(x, 1)
    ln(x)  = eml(1, eml(eml(1, x), 1))                    # formula (5)
    x - y  = eml(ln(x), exp(y))

Everything else is built compositionally on top of these three. The
resulting trees are NOT optimised for K — the paper shows direct search
finds much shorter forms. This module prioritises correctness and
coverage.
"""
from __future__ import annotations

from functools import lru_cache

from .core import EMLNode, Eml, Var, eml, one


# ---------------------------------------------------------------------------
# Direct EML identities
# ---------------------------------------------------------------------------

def exp_(x: EMLNode) -> EMLNode:
    """exp(x) = eml(x, 1)"""
    return eml(x, one())


def ln_(x: EMLNode) -> EMLNode:
    """ln(x) = eml(1, eml(eml(1, x), 1))   — paper eq. (5)"""
    return eml(one(), eml(eml(one(), x), one()))


def sub_(x: EMLNode, y: EMLNode) -> EMLNode:
    """x - y = eml(ln(x), exp(y))"""
    return eml(ln_(x), exp_(y))


# ---------------------------------------------------------------------------
# Composed primitives
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def zero() -> EMLNode:
    """0 = ln(1)"""
    return ln_(one())


def neg_(x: EMLNode) -> EMLNode:
    """-x = 0 - x"""
    return sub_(zero(), x)


def add_(x: EMLNode, y: EMLNode) -> EMLNode:
    """x + y = -((-x) - y)"""
    return neg_(sub_(neg_(x), y))


def mul_(x: EMLNode, y: EMLNode) -> EMLNode:
    """x * y = exp(ln(x) + ln(y))"""
    return exp_(add_(ln_(x), ln_(y)))


def div_(x: EMLNode, y: EMLNode) -> EMLNode:
    """x / y = exp(ln(x) - ln(y))"""
    return exp_(sub_(ln_(x), ln_(y)))


def inv_(x: EMLNode) -> EMLNode:
    """1/x = exp(-ln(x))"""
    return exp_(neg_(ln_(x)))


def pow_(x: EMLNode, y: EMLNode) -> EMLNode:
    """x^y = exp(y * ln(x))"""
    return exp_(mul_(y, ln_(x)))


def sqrt_(x: EMLNode) -> EMLNode:
    """sqrt(x) = x^(1/2)"""
    return pow_(x, half())


# ---------------------------------------------------------------------------
# Constants (cached — they are re-used extensively)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def integer(n: int) -> EMLNode:
    """Build a non-negative integer as 1 + 1 + ... + 1.

    Negative integers go through `neg_`.
    """
    if n < 0:
        return neg_(integer(-n))
    if n == 0:
        return zero()
    if n == 1:
        return one()
    return add_(one(), integer(n - 1))


@lru_cache(maxsize=None)
def rational(p: int, q: int) -> EMLNode:
    if q == 0:
        raise ValueError("rational: zero denominator")
    if q < 0:
        p, q = -p, -q
    return div_(integer(p), integer(q))


@lru_cache(maxsize=None)
def half() -> EMLNode:
    return rational(1, 2)


@lru_cache(maxsize=None)
def e_const() -> EMLNode:
    """e = exp(1) = eml(1, 1)"""
    return exp_(one())


@lru_cache(maxsize=None)
def i_const() -> EMLNode:
    """i = sqrt(-1).

    NB: Using the EML-encoded ln (formula 5) on negative reals yields the
    conjugate of the principal branch (see paper §4.1). The evaluator
    compensates so downstream trig/constant identities behave correctly.
    """
    return sqrt_(integer(-1))


@lru_cache(maxsize=None)
def pi_const() -> EMLNode:
    """pi = -i * ln(-1).

    With principal branch, ln(-1) = i*pi, so -i * i*pi = pi.
    """
    return mul_(neg_(i_const()), ln_(integer(-1)))


# ---------------------------------------------------------------------------
# Hyperbolic functions  (clean, real-domain)
# ---------------------------------------------------------------------------

def sinh_(x: EMLNode) -> EMLNode:
    """(exp(x) - exp(-x)) / 2"""
    return div_(sub_(exp_(x), exp_(neg_(x))), integer(2))


def cosh_(x: EMLNode) -> EMLNode:
    """(exp(x) + exp(-x)) / 2"""
    return div_(add_(exp_(x), exp_(neg_(x))), integer(2))


def tanh_(x: EMLNode) -> EMLNode:
    return div_(sinh_(x), cosh_(x))


def asinh_(x: EMLNode) -> EMLNode:
    """ln(x + sqrt(x^2 + 1))"""
    return ln_(add_(x, sqrt_(add_(pow_(x, integer(2)), one()))))


def acosh_(x: EMLNode) -> EMLNode:
    """ln(x + sqrt(x^2 - 1))"""
    return ln_(add_(x, sqrt_(sub_(pow_(x, integer(2)), one()))))


def atanh_(x: EMLNode) -> EMLNode:
    """(1/2) * ln((1 + x)/(1 - x))"""
    return mul_(half(), ln_(div_(add_(one(), x), sub_(one(), x))))


# ---------------------------------------------------------------------------
# Trigonometric functions (via Euler's formula)
# ---------------------------------------------------------------------------

def sin_(x: EMLNode) -> EMLNode:
    """(exp(i*x) - exp(-i*x)) / (2i)"""
    ix = mul_(i_const(), x)
    return div_(sub_(exp_(ix), exp_(neg_(ix))), mul_(integer(2), i_const()))


def cos_(x: EMLNode) -> EMLNode:
    """(exp(i*x) + exp(-i*x)) / 2"""
    ix = mul_(i_const(), x)
    return div_(add_(exp_(ix), exp_(neg_(ix))), integer(2))


def tan_(x: EMLNode) -> EMLNode:
    return div_(sin_(x), cos_(x))


def asin_(x: EMLNode) -> EMLNode:
    """-i * ln(i*x + sqrt(1 - x^2))"""
    return mul_(
        neg_(i_const()),
        ln_(add_(mul_(i_const(), x), sqrt_(sub_(one(), pow_(x, integer(2)))))),
    )


def acos_(x: EMLNode) -> EMLNode:
    """-i * ln(x + i*sqrt(1 - x^2))"""
    return mul_(
        neg_(i_const()),
        ln_(add_(x, mul_(i_const(), sqrt_(sub_(one(), pow_(x, integer(2))))))),
    )


def atan_(x: EMLNode) -> EMLNode:
    """(i/2) * ln((i - x)/(i + x))"""
    return mul_(
        div_(i_const(), integer(2)),
        ln_(div_(sub_(i_const(), x), add_(i_const(), x))),
    )


# ---------------------------------------------------------------------------
# Misc helpers exposed for the compiler
# ---------------------------------------------------------------------------

def log_base(x: EMLNode, base: EMLNode) -> EMLNode:
    """log_base(x) = ln(x) / ln(base)"""
    return div_(ln_(x), ln_(base))


def sigmoid_(x: EMLNode) -> EMLNode:
    """1 / (1 + exp(-x))"""
    return inv_(add_(one(), exp_(neg_(x))))
