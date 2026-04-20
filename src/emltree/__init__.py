"""EML formula generator.

Compile elementary-function formulas into pure EML (Exp-Minus-Log) form,
where the only operator is  eml(x, y) = exp(x) - ln(y)  and the only
constant is 1, per Odrzywolek (arXiv:2603.21852).
"""

from .core import EMLNode, One, Var, Eml, eml, one, var
from .compiler import compile_formula, compile_sympy
from .evaluator import evaluate

__all__ = [
    "EMLNode",
    "One",
    "Var",
    "Eml",
    "eml",
    "one",
    "var",
    "compile_formula",
    "compile_sympy",
    "evaluate",
]
