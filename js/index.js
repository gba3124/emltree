/**
 * emltree — compile elementary functions into pure EML form.
 *
 * Grammar:  S -> 1 | x | eml(S, S)      where  eml(x, y) = exp(x) - ln(y)
 *
 * JS port of the Python `emltree` package (same identities, same tree shapes).
 * Canonical identities (Odrzywolek, arXiv:2603.21852):
 *
 *     exp(x) = eml(x, 1)
 *     ln(x)  = eml(1, eml(eml(1, x), 1))          // paper eq. (5)
 *     x - y  = eml(ln(x), exp(y))
 */

// --- core tree ------------------------------------------------------------

export const ONE = Object.freeze({ t: '1' });

export function variable(name) {
  return Object.freeze({ t: 'var', name });
}

export function eml(left, right) {
  return Object.freeze({ t: 'eml', left, right });
}

export function toNested(n) {
  if (n.t === '1') return '1';
  if (n.t === 'var') return n.name;
  return `eml(${toNested(n.left)}, ${toNested(n.right)})`;
}

/** Reverse Polish notation; EML is encoded as 'E'. */
export function toRpn(n) {
  if (n.t === '1') return '1';
  if (n.t === 'var') return n.name;
  return `${toRpn(n.left)} ${toRpn(n.right)} E`;
}

export function leafCount(n) {
  return n.t === 'eml' ? leafCount(n.left) + leafCount(n.right) : 1;
}

export function nodeCount(n) {
  return n.t === 'eml' ? 1 + nodeCount(n.left) + nodeCount(n.right) : 1;
}

export function depth(n) {
  return n.t === 'eml' ? 1 + Math.max(depth(n.left), depth(n.right)) : 0;
}

export function* walk(n) {
  yield n;
  if (n.t === 'eml') {
    yield* walk(n.left);
    yield* walk(n.right);
  }
}

/** Render an EML tree as ASCII. Left child drawn above right child. */
export function asciiTree(n, prefix = '', isLast = true) {
  const connector = isLast ? '└── ' : '├── ';
  if (n.t !== 'eml') return `${prefix}${connector}${n.t === '1' ? '1' : n.name}\n`;
  const childPrefix = prefix + (isLast ? '    ' : '│   ');
  return `${prefix}${connector}eml\n`
    + asciiTree(n.left, childPrefix, false)
    + asciiTree(n.right, childPrefix, true);
}

// --- primitives -------------------------------------------------------------
// Trees are NOT optimised for size — direct search finds much shorter forms
// (paper §5). These prioritise correctness and coverage, like the Python original.

export const exp = (x) => eml(x, ONE);                       // exp(x) = eml(x, 1)
export const ln = (x) => eml(ONE, eml(eml(ONE, x), ONE));    // paper eq. (5)
export const sub = (x, y) => eml(ln(x), exp(y));             // x - y

export const ZERO = ln(ONE);                                 // 0 = ln(1)
export const neg = (x) => sub(ZERO, x);
export const add = (x, y) => neg(sub(neg(x), y));
export const mul = (x, y) => exp(add(ln(x), ln(y)));
export const div = (x, y) => exp(sub(ln(x), ln(y)));
export const inv = (x) => exp(neg(ln(x)));
export const pow = (x, y) => exp(mul(y, ln(x)));

const intCache = new Map(); // ponytail: memo shares subtrees; huge n still means huge trees, same ceiling as the Python original
export function integer(n) {
  if (!Number.isSafeInteger(n)) throw new Error(`integer: ${n} is not a safe integer`);
  if (n < 0) return neg(integer(-n));
  if (n === 0) return ZERO;
  if (n === 1) return ONE;
  let acc = intCache.get(n);
  if (!acc) {
    acc = ONE;
    for (let i = 2; i <= n; i++) acc = add(ONE, acc);
    intCache.set(n, acc);
  }
  return acc;
}

export function rational(p, q) {
  if (q === 0) throw new Error('rational: zero denominator');
  if (q < 0) { p = -p; q = -q; }
  return div(integer(p), integer(q));
}

export const HALF = rational(1, 2);
export const sqrt = (x) => pow(x, HALF);

export const E = exp(ONE);                                   // e = eml(1, 1)
export const I = sqrt(integer(-1));                          // i — EML ln gives the conjugate branch on negative reals (paper §4.1); identities below all use the same I so it cancels
export const PI = mul(neg(I), ln(integer(-1)));              // pi = -i * ln(-1)

export const sinh = (x) => div(sub(exp(x), exp(neg(x))), integer(2));
export const cosh = (x) => div(add(exp(x), exp(neg(x))), integer(2));
export const tanh = (x) => div(sinh(x), cosh(x));
export const asinh = (x) => ln(add(x, sqrt(add(pow(x, integer(2)), ONE))));
export const acosh = (x) => ln(add(x, sqrt(sub(pow(x, integer(2)), ONE))));
export const atanh = (x) => mul(HALF, ln(div(add(ONE, x), sub(ONE, x))));

export const sin = (x) => {
  const ix = mul(I, x);
  return div(sub(exp(ix), exp(neg(ix))), mul(integer(2), I));
};
export const cos = (x) => {
  const ix = mul(I, x);
  return div(add(exp(ix), exp(neg(ix))), integer(2));
};
export const tan = (x) => div(sin(x), cos(x));
export const asin = (x) => mul(neg(I), ln(add(mul(I, x), sqrt(sub(ONE, pow(x, integer(2)))))));
export const acos = (x) => mul(neg(I), ln(add(x, mul(I, sqrt(sub(ONE, pow(x, integer(2))))))));
// (i/2) * ln((i + x)/(i - x)) — NB the Python original has (i-x)/(i+x), which is -atan(x); fixed here
export const atan = (x) => mul(div(I, integer(2)), ln(div(add(I, x), sub(I, x))));

export const logBase = (x, base) => div(ln(x), ln(base));
export const sigmoid = (x) => inv(add(ONE, exp(neg(x))));

// --- formula parser + compiler ----------------------------------------------
// Replaces sympy: numbers, identifiers, + - * / ^ ** ( ) , and function calls.

const FUNCS = {
  exp, log: ln, ln, sqrt, sin, cos, tan, asin, acos, atan,
  sinh, cosh, tanh, asinh, acosh, atanh, sigmoid,
};
const CONSTS = { E, pi: PI, PI, I };

export function compile(formula) {
  return compileAst(parse(formula));
}

function tokenize(src) {
  const tokens = [];
  const re = /\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\*\*|[-+*/^(),])/y;
  let pos = 0;
  while (pos < src.length) {
    re.lastIndex = pos;
    const m = re.exec(src);
    if (!m) throw new Error(`parse error at position ${pos}: ${JSON.stringify(src.slice(pos, pos + 10))}`);
    tokens.push(m[1]);
    pos = re.lastIndex;
  }
  return tokens;
}

function parse(src) {
  const toks = tokenize(src.trim());
  let i = 0;
  const peek = () => toks[i];
  const next = () => toks[i++];
  const expect = (t) => {
    const got = next();
    if (got !== t) throw new Error(`expected '${t}', got ${got === undefined ? 'end of formula' : `'${got}'`}`);
  };

  function expr() {
    let node = term();
    while (peek() === '+' || peek() === '-') {
      const op = next();
      node = { k: op, a: node, b: term() };
    }
    return node;
  }
  function term() {
    let node = unary();
    while (peek() === '*' || peek() === '/') {
      const op = next();
      node = { k: op, a: node, b: unary() };
    }
    return node;
  }
  function unary() {
    if (peek() === '-') { next(); return { k: 'neg', a: unary() }; }
    if (peek() === '+') { next(); return unary(); }
    return power();
  }
  function power() {
    const base = atom();
    if (peek() === '**' || peek() === '^') {   // right-associative, like sympy with convert_xor
      next();
      return { k: '^', a: base, b: unary() };
    }
    return base;
  }
  function atom() {
    const t = next();
    if (t === undefined) throw new Error('unexpected end of formula');
    if (t === '(') { const node = expr(); expect(')'); return node; }
    if (/^\d/.test(t)) return { k: 'num', v: t };
    if (/^[A-Za-z_]/.test(t)) {
      if (peek() === '(') {
        next();
        const args = [expr()];
        while (peek() === ',') { next(); args.push(expr()); }
        expect(')');
        return { k: 'call', fn: t, args };
      }
      return { k: 'sym', name: t };
    }
    throw new Error(`unexpected token '${t}'`);
  }

  const node = expr();
  if (i < toks.length) throw new Error(`unexpected token '${toks[i]}'`);
  return node;
}

function compileAst(n) {
  switch (n.k) {
    case 'num': {
      const v = Number(n.v);
      if (Number.isInteger(v)) return integer(v);
      const [p, q] = toFraction(v);
      return rational(p, q);
    }
    case 'sym':
      return n.name in CONSTS ? CONSTS[n.name] : variable(n.name);
    case 'neg': return neg(compileAst(n.a));
    case '+': return add(compileAst(n.a), compileAst(n.b));
    case '-': return sub(compileAst(n.a), compileAst(n.b));
    case '*': return mul(compileAst(n.a), compileAst(n.b));
    case '/': return div(compileAst(n.a), compileAst(n.b));
    case '^': {
      if (n.b.k === 'neg' && n.b.a.k === 'num' && Number(n.b.a.v) === 1) return inv(compileAst(n.a));
      if (n.b.k === 'num' && Number(n.b.v) === 0.5) return sqrt(compileAst(n.a));
      return pow(compileAst(n.a), compileAst(n.b));
    }
    case 'call': {
      if (n.fn === 'log' && n.args.length === 2) {
        return logBase(compileAst(n.args[0]), compileAst(n.args[1]));
      }
      const f = FUNCS[n.fn];
      if (!f || n.args.length !== 1) {
        throw new Error(`no EML expansion registered for ${n.fn}/${n.args.length}`);
      }
      return f(compileAst(n.args[0]));
    }
  }
}

// ponytail: plain continued fractions, no best-approx correction between convergents like Fraction.limit_denominator — close enough for float snapping
function toFraction(x, maxDen = 1e9) {
  const sign = x < 0 ? -1 : 1;
  let v = Math.abs(x);
  let [p0, q0, p1, q1] = [0, 1, 1, 0];
  for (;;) {
    const a = Math.floor(v);
    const p2 = a * p1 + p0;
    const q2 = a * q1 + q0;
    if (q2 > maxDen) break;
    [p0, q0, p1, q1] = [p1, q1, p2, q2];
    const frac = v - a;
    if (frac < 1e-15) break;
    v = 1 / frac;
  }
  return [sign * p1, q1];
}

// --- evaluator ----------------------------------------------------------------
// Complex arithmetic throughout: trig and the constants pi / i flow through
// complex intermediates even for real inputs, exactly as the paper discusses.

export function evaluate(node, bindings = {}) {
  const env = new Map();
  for (const [k, v] of Object.entries(bindings)) {
    env.set(k, typeof v === 'number' ? { re: v, im: 0 } : { re: v.re ?? 0, im: v.im ?? 0 });
  }
  return ev(node, env);
}

function ev(n, env) {
  if (n.t === '1') return { re: 1, im: 0 };
  if (n.t === 'var') {
    const v = env.get(n.name);
    if (v === undefined) throw new Error(`unbound variable '${n.name}' — provide it via bindings`);
    return v;
  }
  const a = ev(n.left, env);
  const b = ev(n.right, env);
  // eml(x, y) = exp(x) - ln(y), principal branch (matches numpy semantics)
  // real-axis special case keeps C99 cexp(±inf + 0i) semantics: r * sin(0) must be 0, not inf*0 = NaN
  const r = Math.exp(a.re);
  return {
    re: (a.im === 0 ? r : r * Math.cos(a.im)) - Math.log(Math.hypot(b.re, b.im)),
    im: (a.im === 0 ? 0 : r * Math.sin(a.im)) - Math.atan2(b.im, b.re),
  };
}
