"""Command-line interface:  eml "sin(x) + 1"  -> nested EML tree."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

from .compiler import EMLCompileError, compile_formula
from .core import ascii_tree
from .evaluator import evaluate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="emltree",
        description=(
            "Compile an elementary formula into pure EML form. "
            "Supports +, -, *, /, **, sqrt, exp, log, sin, cos, tan, "
            "asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, "
            "pi, E, I."
        ),
    )
    p.add_argument("formula", help='e.g. "sin(x) + exp(y)"')
    p.add_argument(
        "--var",
        "-v",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="Declare (and optionally bind) a free variable. Repeatable.",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=("nested", "rpn", "tree"),
        default="nested",
        help="Output style (default: nested).",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print leaf count, node count, depth.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Compare the compiled EML tree's numerical value against sympy's "
            "direct evaluation. Requires all variables to have --var NAME=VALUE."
        ),
    )

    args = p.parse_args(argv)

    declared: dict[str, float | complex | None] = {}
    for spec in args.var:
        if "=" in spec:
            name, value = spec.split("=", 1)
            declared[name.strip()] = complex(sp.sympify(value))
        else:
            declared[spec.strip()] = None

    try:
        tree = compile_formula(args.formula, variables=list(declared.keys()))
    except EMLCompileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.format == "nested":
        print(tree.to_nested())
    elif args.format == "rpn":
        print(tree.to_rpn())
    else:
        print(ascii_tree(tree).rstrip())

    if args.stats:
        print(
            f"[stats] leaves={tree.leaf_count()} "
            f"nodes={tree.node_count()} depth={tree.depth()}",
            file=sys.stderr,
        )

    if args.verify:
        bindings = {k: v for k, v in declared.items() if v is not None}
        missing = [k for k, v in declared.items() if v is None]
        if missing:
            print(
                f"error: --verify needs values for: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2
        got = complex(evaluate(tree, bindings))
        expr = sp.sympify(args.formula, locals={k: sp.Symbol(k) for k in declared})
        expected = complex(expr.evalf(subs=bindings))
        diff = abs(got - expected)
        ok = diff < 1e-8 * max(1.0, abs(expected))
        print(
            f"[verify] eml={got:.10g} expected={expected:.10g} "
            f"|Δ|={diff:.3g} {'OK' if ok else 'MISMATCH'}",
            file=sys.stderr,
        )
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
