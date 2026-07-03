"""EML formula generator.

Compile elementary-function formulas into pure EML (Exp-Minus-Log) form,
where the only operator is  eml(x, y) = exp(x) - ln(y)  and the only
constant is 1, per Odrzywolek (arXiv:2603.21852).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .core import EMLNode, One, Var, Eml, eml, one, var, ascii_tree
from .compiler import EMLCompileError, compile_formula, compile_sympy
from .evaluator import evaluate

try:
    __version__ = _version("emltree")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = [
    "EMLNode",
    "One",
    "Var",
    "Eml",
    "eml",
    "one",
    "var",
    "ascii_tree",
    "EMLCompileError",
    "compile_formula",
    "compile_sympy",
    "evaluate",
    "__version__",
]
