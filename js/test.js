import test from 'node:test';
import assert from 'node:assert/strict';
import {
  compile, evaluate, toNested, toRpn, leafCount, depth,
  ln, exp, variable,
} from './index.js';

function close(got, expected, tol = 1e-9) {
  const e = typeof expected === 'number' ? { re: expected, im: 0 } : expected;
  const scale = Math.max(1, Math.hypot(e.re, e.im));
  assert.ok(
    Math.hypot(got.re - e.re, got.im - e.im) < tol * scale,
    `got ${got.re}+${got.im}i, expected ${e.re}+${e.im}i`,
  );
}

const check = (formula, bindings, expected, tol) =>
  test(formula, () => close(evaluate(compile(formula), bindings), expected, tol));

// algebra (mirrors Python test_algebra)
check('x', { x: 2.5 }, 2.5);
check('x + y', { x: 1.5, y: 0.7 }, 2.2);
check('x - y', { x: 4.0, y: 1.25 }, 2.75);
check('x * y', { x: 3, y: 2 }, 6);
check('x / y', { x: 7, y: 4 }, 1.75);
check('1/x', { x: 0.8 }, 1.25);
check('-x', { x: 3.14 }, -3.14);
check('exp(x)', { x: 1.2 }, Math.exp(1.2));
check('log(x)', { x: 3 }, Math.log(3));
check('sqrt(x)', { x: 5 }, Math.sqrt(5));
check('x**3', { x: 1.3 }, 1.3 ** 3);
check('x^3', { x: 1.3 }, 1.3 ** 3);
check('x**y', { x: 2, y: 3.5 }, 2 ** 3.5);
check('exp(x) - log(y)', { x: 0.3, y: 2.5 }, Math.exp(0.3) - Math.log(2.5));
check('log(x, 2)', { x: 8 }, 3, 1e-8);
check('sigmoid(x)', { x: 0.5 }, 1 / (1 + Math.exp(-0.5)));

// hyperbolic (mirrors Python test_hyperbolic)
check('sinh(x)', { x: 0.7 }, Math.sinh(0.7), 1e-8);
check('cosh(x)', { x: 0.7 }, Math.cosh(0.7), 1e-8);
check('tanh(x)', { x: 0.7 }, Math.tanh(0.7), 1e-8);
check('asinh(x)', { x: 0.4 }, Math.asinh(0.4), 1e-8);
check('acosh(x)', { x: 1.8 }, Math.acosh(1.8), 1e-8);
check('atanh(x)', { x: 0.3 }, Math.atanh(0.3), 1e-8);

// trig (mirrors Python test_trig)
check('sin(x)', { x: 0.9 }, Math.sin(0.9), 1e-7);
check('cos(x)', { x: 0.9 }, Math.cos(0.9), 1e-7);
check('tan(x)', { x: 0.4 }, Math.tan(0.4), 1e-7);
check('sin(x)**2 + cos(x)**2', { x: 1.37 }, 1, 1e-7);
check('asin(x)', { x: 0.6 }, Math.asin(0.6), 1e-7);
check('acos(x)', { x: 0.6 }, Math.acos(0.6), 1e-7);
check('atan(x)', { x: 0.6 }, Math.atan(0.6), 1e-7);

test('constants', () => {
  close(evaluate(compile('E')), Math.E, 1e-12);
  close(evaluate(compile('pi')), Math.PI, 1e-7);
});

test('paper ln identity (eq. 5)', () => {
  const tree = ln(variable('x'));
  assert.equal(toNested(tree), 'eml(1, eml(eml(1, x), 1))');
  assert.equal(leafCount(tree), 4);
  assert.equal(depth(tree), 3);
});

test('paper exp identity', () => {
  assert.equal(toNested(exp(variable('x'))), 'eml(x, 1)');
});

test('rpn encoding', () => {
  // paper: RPN for ln is  1 1 x E 1 E E
  assert.equal(toRpn(ln(variable('x'))), '1 1 x E 1 E E');
});

test('unbound variable throws', () => {
  assert.throws(() => evaluate(compile('x + y'), { x: 1 }), /unbound variable 'y'/);
});

test('unknown function throws', () => {
  assert.throws(() => compile('gamma(x)'), /no EML expansion/);
});
