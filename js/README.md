# emltree

Compile elementary functions into pure **EML form** — binary trees built from the
single Sheffer-like operator

```
eml(x, y) = exp(x) - ln(y)
```

plus the constant `1`. Odrzywołek ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852))
showed this one operator generates every function on a scientific calculator — the
NAND gate of continuous mathematics.

Zero dependencies. JS port of the [Python `emltree` package](https://github.com/gba3124/emltree)
(same identities, same tree shapes).

## Install

```
npm install emltree
```

## Usage

```js
import { compile, evaluate, toNested, toRpn, asciiTree } from 'emltree';

const tree = compile('sin(x)^2 + cos(x)^2');
evaluate(tree, { x: 1.37 });   // { re: 1.0000000…, im: ~0 }

toNested(compile('exp(x)'));   // 'eml(x, 1)'
toRpn(compile('log(x)'));      // '1 1 x E 1 E E'   (paper eq. 5)
```

Builders are exported too, if you'd rather skip the parser:

```js
import { variable, sin, pow, add, integer, evaluate } from 'emltree';

const x = variable('x');
const tree = add(pow(sin(x), integer(2)), integer(1));
```

Evaluation is complex throughout (`{ re, im }`) — trig and the constants `pi` / `I`
flow through complex intermediates even for real inputs, exactly as in the paper.

## CLI

```
npx emltree "sin(x)" -f rpn
npx emltree "exp(x) - log(y)" --stats --eval x=0.3,y=2.5
npx emltree "pi" -f tree
```

Supported syntax: `+ - * / ^ **`, `sqrt exp log ln sin cos tan asin acos atan
sinh cosh tanh asinh acosh atanh sigmoid`, two-arg `log(x, base)`, constants
`pi E I`.

## Caveats

- **Branch cuts**: outside their real domains (`asin(2)`, `acosh(-2)`, `log` of
  negatives, …) results flow through complex branch cuts and may land on a
  non-principal branch — or, where float fuzz compounds, off-sheet entirely
  (paper §4.1). On the usual real domains everything matches to ~1e-7.
- **Addition overflow**: `add`'s expansion applies `exp()` to its second
  operand, so adding values past ~709 overflows float64. Integer/decimal
  constants avoid this internally (binary decomposition, multiplicative odd
  step), but `x + y` with huge `y` is an inherent ceiling of the encoding.

## Canonical identities

```
exp(x) = eml(x, 1)
ln(x)  = eml(1, eml(eml(1, x), 1))      # paper eq. (5)
x - y  = eml(ln(x), exp(y))
```

Everything else is built compositionally on top of these three. The resulting
trees are **not** optimised for size — the paper's direct search finds much
shorter forms; this package prioritises correctness and coverage.

## License

MIT
