// Shared cross-language vectors — the JS half.
// Same ../vectors.json drives tests/test_vectors.py. See that file for why.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { compile, evaluate } from './index.js';

const doc = JSON.parse(
  fs.readFileSync(new URL('../vectors.json', import.meta.url), 'utf8'),
);

for (const { formula, bindings, expected } of doc.cases) {
  test(`${formula} @ ${JSON.stringify(bindings)}`, () => {
    const got = evaluate(compile(formula), bindings);
    const scale = Math.max(1, Math.abs(expected));
    assert.ok(
      Math.hypot(got.re - expected, got.im) < doc.tolerance * scale,
      `${formula}: got ${got.re}+${got.im}i, expected ${expected}`,
    );
  });
}
