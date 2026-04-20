# emltree

Compile elementary-function formulas into pure **EML (Exp-Minus-Log)** form,
where the only operator is

```
eml(x, y) = exp(x) − ln(y)
```

and the only constant is `1`.

## Paper

This project is an implementation of ideas introduced in:

> **Andrzej Odrzywolek.** *All elementary functions from a single operator.*
> arXiv:2603.21852 (2026).
> [https://arxiv.org/abs/2603.21852](https://arxiv.org/abs/2603.21852)

The paper proves that the single binary operator `eml(x, y) = exp(x) − ln(y)`
together with the constant `1` generates the entire scientific-calculator
basis (arithmetic, roots, logs, all trig and hyperbolic functions, the
constants `e`, `π`, `i`). Every elementary expression becomes a binary
tree whose only internal node is `eml`. This package is the inverse
direction: it **takes an ordinary formula and compiles it into that
tree**. A sibling Rust crate, [OxiEML](https://github.com/cool-japan/oxieml),
covers the Rust ecosystem; `emltree` focuses on the Python side.

## Quick start

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

# Compile a formula
uv run emltree "sin(x) + exp(y)" -v x -v y

# ASCII tree
uv run emltree "log(x)" -v x -f tree

# Numerical sanity check
uv run emltree "sin(x)**2 + cos(x)**2" -v x=0.9 --verify
```

## Library use

```python
from emltree import compile_formula, evaluate

tree = compile_formula("sqrt(x**2 + y**2)", variables=["x", "y"])
print(tree.to_nested())
print(evaluate(tree, {"x": 3.0, "y": 4.0}))   # ≈ 5+0j
```

The returned tree is an immutable ADT (`One` / `Var` / `Eml`) — walk it,
hash it, render as RPN (`1 x E 1 E`), or ship it into an FPGA/analog
circuit as the paper suggests.

## Why the trees are large

This compiler is **compositional**, not optimal. Every primitive bottoms
out in `exp / ln / sub`, so `sin(x)` produces a tree with hundreds of
nodes. The paper's direct-search results (Table 4) are vastly shorter —
integrating that search is one of the open directions below.

## Tests

```bash
uv run pytest
```

## Contributing

Contributions are very welcome — this is an early-stage package and
there's plenty of room to improve. A few directions that would make a
great first PR:

- **Shorter trees** — hand-curated or searched EML identities to
  replace the naive compositional expansions (see paper Table 4).
- **More primitives** — `abs`, `sign`, `floor`, `ceil`, `erf`, etc.
  (many require tricks; see the paper's supplementary).
- **Better output** — LaTeX, GraphViz / Mermaid, Jupyter rich repr.
- **Batched evaluation** — a vectorised numpy/torch evaluator for
  large input arrays.
- **Optional Rust backend** — PyO3 bindings to
  [OxiEML](https://github.com/cool-japan/oxieml) for expensive search
  and symbolic regression.
- **Docs & examples** — walk-throughs of the paper's identities,
  notebooks showing symbolic-regression use cases.

Please open an issue to discuss before sending a large PR. Bug reports,
documentation fixes, and additional test cases are also very welcome —
no contribution is too small.

### Development

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest
```

## License

MIT — see [LICENSE](./LICENSE).

## Citation

If you use `emltree` in academic work, please cite **both** the paper and
the software:

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single operator},
  author  = {Odrzywolek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}

@software{hsiao2026emltree,
  author  = {Hsiao, Wei-Chien},
  title   = {emltree: a Python compiler from elementary formulas to EML trees},
  year    = {2026},
  url     = {https://github.com/gba3124/emltree},
  version = {0.1.0}
}
```
