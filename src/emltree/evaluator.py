"""Numerical evaluation of EML trees.

All arithmetic is done in complex128 because many identities (trig,
constants like pi / i) flow through complex intermediates even for
real-valued inputs — exactly as the paper discusses.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .core import Eml, EMLNode, One, Var


ComplexLike = complex | float | int | np.complexfloating | np.floating


def evaluate(
    node: EMLNode,
    bindings: Mapping[str, ComplexLike] | None = None,
) -> np.complex128:
    """Evaluate an EML tree with optional variable bindings.

    `bindings` maps variable names to numeric values. Missing bindings
    raise `KeyError`.
    """
    env = {k: np.complex128(v) for k, v in (bindings or {}).items()}
    return _eval(node, env)


def _eval(node: EMLNode, env: dict[str, np.complex128]) -> np.complex128:
    if isinstance(node, One):
        return np.complex128(1.0)
    if isinstance(node, Var):
        if node.name not in env:
            raise KeyError(
                f"unbound variable '{node.name}' — provide it via bindings=..."
            )
        return env[node.name]
    assert isinstance(node, Eml)
    left = _eval(node.left, env)
    right = _eval(node.right, env)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp(left) - np.log(right)
