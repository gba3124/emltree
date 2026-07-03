"""Numerical evaluation of EML trees.

All arithmetic is done in complex128 because many identities (trig,
constants like pi / i) flow through complex intermediates even for
real-valued inputs — exactly as the paper discusses.

Bindings may be scalars or numpy arrays; arrays evaluate elementwise
(vectorised) and broadcast against each other.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .core import Eml, EMLNode, One, Var


ComplexLike = complex | float | int | np.complexfloating | np.floating


def evaluate(
    node: EMLNode,
    bindings: Mapping[str, ComplexLike | np.ndarray] | None = None,
) -> np.complex128 | np.ndarray:
    """Evaluate an EML tree with optional variable bindings.

    `bindings` maps variable names to numbers or numpy arrays. Missing
    bindings raise `KeyError`. Scalar bindings return `np.complex128`;
    array bindings return an elementwise-evaluated array.
    """
    env: dict[str, np.complex128 | np.ndarray] = {}
    for k, v in (bindings or {}).items():
        arr = np.asarray(v, dtype=np.complex128)
        env[k] = arr if arr.ndim else np.complex128(arr)
    return _eval(node, env, {})


def _eval(
    node: EMLNode,
    env: dict[str, np.complex128 | np.ndarray],
    cache: dict[int, np.complex128 | np.ndarray],
) -> np.complex128 | np.ndarray:
    if isinstance(node, One):
        return np.complex128(1.0)
    if isinstance(node, Var):
        if node.name not in env:
            raise KeyError(
                f"unbound variable '{node.name}' — provide it via bindings=..."
            )
        return env[node.name]
    assert isinstance(node, Eml)
    # Trees share subtrees heavily (constants, cached integers); memoising by
    # node identity makes evaluation O(unique nodes) instead of O(unfolded tree).
    key = id(node)
    hit = cache.get(key)
    if hit is not None:
        return hit
    left = _eval(node.left, env, cache)
    right = _eval(node.right, env, cache)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.exp(left) - np.log(right)
    cache[key] = out
    return out
